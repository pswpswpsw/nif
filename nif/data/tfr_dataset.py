import os

import numpy as np
import tensorflow as tf


class TFRDataset(object):
    """A class to handle creating and loading Tensorflow record datasets.

    Args:
        n_feature (int): The number of features.
        n_target (int): The number of targets.
        area_weight (bool, optional): Whether or not to use area weights. Defaults to False.
    """

    def __init__(self, n_feature, n_target, area_weight=False):
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.n_feature = n_feature
        self.n_target = n_target
        self.area_weight = area_weight

    def create_from_npz(self, num_pts_per_file, npz_path, npz_key, tfr_path, prefix):
        """Create Tensorflow record files from a numpy file.

        Args:
            num_pts_per_file (int): The number of points to put into each Tensorflow record file.
            npz_path (str): The path to the numpy file.
            npz_key (str): The key of the numpy array to use.
            tfr_path (str): The path to the output directory for the Tensorflow record files.
            prefix (str): The prefix to add to each Tensorflow record file name.
        """
        num_pts_per_file = int(num_pts_per_file)
        npz_data = np.load(npz_path)[npz_key]
        NUM_TOTAL_PTS, N_COL = npz_data.shape

        if self.area_weight:
            assert N_COL == self.n_feature + self.n_target + 1
            data_weight = npz_data[:, -1:]
        else:
            assert N_COL == self.n_feature + self.n_target
        data_feature = npz_data[:, : self.n_feature]
        data_target = npz_data[:, self.n_feature : self.n_feature + self.n_target]

        total_num_files = int(np.ceil(NUM_TOTAL_PTS / num_pts_per_file))
        print("total number of TFR files = ", total_num_files)

        # shuffle the data before distributing
        np.random.shuffle(npz_data)

        # make dir
        mkdir(tfr_path)

        for i in range(total_num_files):
            print("working in {}-th file... total {}".format(i + 1, total_num_files))
            filename = tfr_path + "/{}_{}.tfrecord".format(prefix, i)

            i0 = i * num_pts_per_file
            i1 = i0 + num_pts_per_file

            data_feature_ = data_feature[i0:i1]
            data_target_ = data_target[i0:i1]

            feature_dict = {}
            for j in range(self.n_feature):
                feature_dict["input_" + str(j)] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=data_feature_[:, j])
                )
            for j in range(self.n_target):
                feature_dict["output_" + str(j)] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=data_target_[:, j])
                )

            if self.area_weight:
                data_weight_ = data_weight[i0:i1]
                feature_dict["weight"] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=data_weight_)
                )

            with tf.io.TFRecordWriter(filename) as writer:
                example = tf.train.Example(
                    features=tf.train.Features(feature=feature_dict)
                )
                writer.write(example.SerializeToString())

    def gen_dataset_from_batch_file(self, batch_file, batch_size):
        """Generate a TensorFlow Dataset from a batch file.

        Args:
            batch_file (np.ndarray): A NumPy array containing the batch data.
            batch_size (int): The batch size.

        Returns:
            tf.data.Dataset: A TensorFlow Dataset object.

        """
        features = tf.transpose(
            tf.squeeze(tf.stack(batch_file[: self.n_feature], 0), 1)
        )
        if self.area_weight:
            target = tf.transpose(
                batch_file[self.n_feature : self.n_feature + self.n_target]
            )
            weight = tf.reshape(batch_file[-1], (-1, 1))
            batch_dataset = tf.data.Dataset.from_tensor_slices(
                (features, target, weight)
            )
        else:
            target = tf.transpose(batch_file[-self.n_target :])
            batch_dataset = tf.data.Dataset.from_tensor_slices((features, target))
        batch_dataset = (
            batch_dataset.shuffle(features.shape[0])
            .batch(batch_size)
            .prefetch(self.AUTOTUNE)
        )
        return batch_dataset

    def get_tfr_meta_dataset(self, tfr_path, epoch, tfr_shuffle_buffer_size=1):
        """Get a meta TensorFlow Dataset object from a folder of TFRecord files.

        Args:
            tfr_path (str): The path to the folder containing the TFRecord files.
            epoch (int): The number of epochs to iterate through.
            tfr_shuffle_buffer_size (int): The shuffle buffer size.

        Returns:
            tf.data.Dataset: A TensorFlow Dataset object.

        """
        # I used an abnormal way to create tfrecord data,
        # I cannot use point wise data line by line for example.
        # because it will end up with an unacceptable create-file time.

        filenames = tf.io.gfile.glob(f"{tfr_path}/*.tfrecord")
        self.num_pts_per_file = len(filenames)

        def prepare_sample(example):
            schema = {}
            for j in range(self.n_feature):
                schema["input_" + str(j)] = tf.io.FixedLenSequenceFeature(
                    [], tf.float32, allow_missing=True
                )
            for j in range(self.n_target):
                schema["output_" + str(j)] = tf.io.FixedLenSequenceFeature(
                    [], tf.float32, allow_missing=True
                )
            if self.area_weight:
                schema["weight"] = tf.io.FixedLenSequenceFeature(
                    [], tf.float32, allow_missing=True
                )
            data_dict = tf.io.parse_single_example(example, schema)
            return list(data_dict.values())

        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(prepare_sample, num_parallel_calls=self.AUTOTUNE)
        if tfr_shuffle_buffer_size > 1:
            dataset = dataset.shuffle(buffer_size=tfr_shuffle_buffer_size)
        # dataset = dataset.shuffle(buffer_size=len(filenames))
        dataset = dataset.repeat(epoch)
        dataset = dataset.batch(
            1
        )  # each time only take on tfrecord out. then we will do sub-batching inside.
        dataset = dataset.prefetch(self.AUTOTUNE)
        return dataset


def mkdir(directory):
    """Create a directory if it does not exist.

    Args:
        directory (str): The directory path.

    """
    if not os.path.exists(directory):
        os.makedirs(directory)
