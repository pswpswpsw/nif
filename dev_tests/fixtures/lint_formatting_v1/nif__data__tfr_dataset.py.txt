"""Read/write original column-list TFRecord Examples without TensorFlow.

Every shard record contains float-list features ``input_0``, ...,
``output_0``, ..., and optional ``weight``. Unlike a row-per-Example format,
this is the original NIF representation, including a shorter final shard.
TFRecord framing and masked CRC32C are checked independently of protobuf.
"""
import glob
import os
import random
import struct
from pathlib import Path

import google_crc32c
import numpy as np
import torch
from tfrecord import example_pb2
from torch.utils.data import DataLoader, IterableDataset, TensorDataset, get_worker_info


def _masked_crc(data):
    crc = google_crc32c.value(data)
    return (((crc >> 15) | (crc << 17)) + 0xa282ead8) & 0xffffffff


def _read_records(filename):
    with open(filename, 'rb') as file:
        while True:
            length = file.read(8)
            if not length:
                break
            if len(length) != 8:
                raise ValueError(f'Truncated TFRecord length: {filename}')
            length_crc = file.read(4)
            if len(length_crc) != 4 or struct.unpack('<I', length_crc)[0] != _masked_crc(length):
                raise ValueError(f'Invalid TFRecord length checksum: {filename}')
            size, = struct.unpack('<Q', length)
            payload = file.read(size)
            crc = file.read(4)
            if len(payload) != size or len(crc) != 4:
                raise ValueError(f'Truncated TFRecord payload: {filename}')
            if struct.unpack('<I', crc)[0] != _masked_crc(payload):
                raise ValueError(f'Invalid TFRecord payload checksum: {filename}')
            yield payload


def _write_record(file, payload):
    length = struct.pack('<Q', len(payload))
    file.write(length)
    file.write(struct.pack('<I', _masked_crc(length)))
    file.write(payload)
    file.write(struct.pack('<I', _masked_crc(payload)))


class _ShardDataset(IterableDataset):
    def __init__(self, filenames, keys, epochs, shuffle_buffer):
        super().__init__()
        self.filenames = filenames
        self.keys = keys
        self.epochs = epochs
        self.shuffle_buffer = shuffle_buffer

    def _epoch(self, filenames):
        for filename in filenames:
            for payload in _read_records(filename):
                example = example_pb2.Example()
                example.ParseFromString(payload)
                features = example.features.feature
                if any(key not in features for key in self.keys):
                    raise ValueError(f'Missing expected feature in TFRecord {filename}')
                columns = []
                for key in self.keys:
                    feature = features[key]
                    if feature.WhichOneof('kind') != 'float_list':
                        raise ValueError(f'{key} must be a float-list feature')
                    columns.append(torch.tensor(list(feature.float_list.value), dtype=torch.float32).unsqueeze(0))
                if len({column.numel() for column in columns}) != 1:
                    raise ValueError(f'Column lengths differ in {filename}')
                yield columns

    def __iter__(self):
        worker = get_worker_info()
        filenames = self.filenames if worker is None else self.filenames[worker.id::worker.num_workers]
        for _ in range(self.epochs):
            stream = self._epoch(filenames)
            # A bounded shuffle keeps memory independent of the total dataset.
            if self.shuffle_buffer > 1:
                buffer = []
                for shard in stream:
                    if len(buffer) < self.shuffle_buffer:
                        buffer.append(shard)
                    else:
                        index = random.randrange(len(buffer))
                        yield buffer[index]
                        buffer[index] = shard
                random.shuffle(buffer)
                yield from buffer
            else:
                yield from stream


class TFRDataset:
    """Native iterable access to legacy TFRecord shards.

    ``get_tfr_meta_dataset`` yields one list of column tensors per record, each
    column shaped ``[1, rows]`` as in the old batched meta-dataset. Pass it to
    ``gen_dataset_from_batch_file`` for an ordinary PyTorch DataLoader yielding
    ``(features, targets[, weight])``, with weights shaped ``[batch, 1]``.
    """
    def __init__(self, n_feature, n_target, area_weight=False):
        if not isinstance(n_feature, int) or not isinstance(n_target, int) or n_feature <= 0 or n_target <= 0:
            raise ValueError('n_feature and n_target must be positive integers')
        self.n_feature = n_feature
        self.n_target = n_target
        self.area_weight = bool(area_weight)
        self.AUTOTUNE = None  # retained attribute; DataLoader controls workers

    @property
    def _keys(self):
        return ([f'input_{j}' for j in range(self.n_feature)] +
                [f'output_{j}' for j in range(self.n_target)] +
                (['weight'] if self.area_weight else []))

    def create_from_npz(self, num_pts_per_file, npz_path, npz_key, tfr_path, prefix):
        num_pts_per_file = int(num_pts_per_file)
        if num_pts_per_file <= 0:
            raise ValueError('num_pts_per_file must be positive')
        with np.load(npz_path) as file:
            data = np.asarray(file[npz_key])
        if data.ndim != 2 or data.shape[1] != len(self._keys):
            raise ValueError('NPZ columns do not match features, targets and optional weight')
        if not np.all(np.isfinite(data)):
            raise ValueError('NPZ data must be finite')
        if self.area_weight and np.any(data[:, -1] < 0):
            raise ValueError('area weights must be nonnegative')
        # Shuffle complete rows so targets and weights never lose alignment.
        data = data[np.random.permutation(len(data))]
        mkdir(tfr_path)
        paths = []
        for index, start in enumerate(range(0, len(data), num_pts_per_file)):
            shard = data[start:start + num_pts_per_file]
            example = example_pb2.Example()
            for column, key in enumerate(self._keys):
                example.features.feature[key].float_list.value.extend(shard[:, column].astype(np.float32).tolist())
            path = Path(tfr_path) / f'{prefix}_{index}.tfrecord'
            with path.open('wb') as file:
                _write_record(file, example.SerializeToString())
            paths.append(str(path))
        return paths

    def gen_dataset_from_batch_file(self, batch_file, batch_size, shuffle=True, **loader_kwargs):
        if int(batch_size) != batch_size or batch_size <= 0:
            raise ValueError('batch_size must be a positive integer')
        if isinstance(batch_file, dict):
            batch_file = [batch_file[key] for key in self._keys]
        if len(batch_file) != len(self._keys):
            raise ValueError('Incorrect number of shard columns')
        columns = [torch.as_tensor(column, dtype=torch.float32).reshape(-1) for column in batch_file]
        if len({column.numel() for column in columns}) != 1:
            raise ValueError('Shard columns must have equal lengths')
        features = torch.stack(columns[:self.n_feature], dim=-1)
        target = torch.stack(columns[self.n_feature:self.n_feature+self.n_target], dim=-1)
        values = (features, target)
        if self.area_weight:
            values += (columns[-1].reshape(-1, 1),)
        return DataLoader(TensorDataset(*values), batch_size=int(batch_size),
                          shuffle=bool(shuffle and len(features)), **loader_kwargs)

    def get_tfr_meta_dataset(self, tfr_path, epoch, tfr_shuffle_buffer_size=1):
        if int(epoch) != epoch or epoch < 0:
            raise ValueError('epoch must be a nonnegative integer')
        if int(tfr_shuffle_buffer_size) != tfr_shuffle_buffer_size or tfr_shuffle_buffer_size <= 0:
            raise ValueError('shuffle buffer must be a positive integer')
        filenames = sorted(glob.glob(os.path.join(str(tfr_path), '*.tfrecord')))
        if not filenames:
            raise FileNotFoundError(f'No TFRecord shards in {tfr_path}')
        self.num_files = len(filenames)
        self.num_pts_per_file = len(filenames)  # legacy attribute (historically misnamed)
        return _ShardDataset(filenames, self._keys, int(epoch), int(tfr_shuffle_buffer_size))


def mkdir(directory):
    os.makedirs(directory, exist_ok=True)
