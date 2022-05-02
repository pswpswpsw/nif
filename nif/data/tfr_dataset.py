import os
import numpy as np
import tensorflow as tf

class TFRDataset(object):
    def __init__(self, n_feature, n_target, area_weight=False):
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.n_feature = n_feature
        self.n_target = n_target
        self.area_weight = area_weight

    def create_from_npz(self, num_pts_per_file, npz_path, npz_key, write_tfr_path):
        num_pts_per_file = int(num_pts_per_file)
        npz_data = np.load(npz_path)[npz_key]
        NUM_TOTAL_PTS, N_COL = npz_data.shape

        if self.area_weight:
            assert N_COL == self.n_feature + self.n_target + 1
        else:
            assert N_COL == self.n_feature + self.n_target
        total_num_files = int(np.ceil(NUM_TOTAL_PTS / num_pts_per_file))
        print('Total number of TFR files = ', total_num_files)

        # shuffle the data before distributing
        np.random.shuffle(npz_data)

        # make dir
        mkdir(write_tfr_path)

        # write each file into `train_i.tfrecord`
        for i in range(total_num_files):
            tfrecords_filename = write_tfr_path + '/' + 'train_' + str(i) + '.tfrecord'
            writer = tf.io.TFRecordWriter(tfrecords_filename)

            tmp_data = np.float32(npz_data[i * num_pts_per_file : (i + 1) * num_pts_per_file, :])

            print('tmp data size = ',tmp_data.shape)

            feature_data = tmp_data[:,:self.n_feature]
            target_data = tmp_data[:,self.n_feature:self.n_feature + self.n_target]
            data_dict = {
                'feature': tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature_data.tobytes()])),
                'target': tf.train.Feature(bytes_list=tf.train.BytesList(value=[target_data.tobytes()]))
            }
            if self.area_weight:
                weight_data = tmp_data[:,-1:]
                data_dict['weight'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[weight_data.tobytes()]))
            example = tf.train.Example(features=tf.train.Features(feature=data_dict))
            writer.write(example.SerializeToString())
            writer.close()

    def get_dataset(self, path, readfile_batch_size, shuffer_buffer=1):
        filenames = tf.io.gfile.glob(f"{path}/*.tfrecord")
        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=self.AUTOTUNE)
        dataset = dataset.map(self.decode_fn, num_parallel_calls=self.AUTOTUNE)
        dataset = dataset.map(self.prepare_sample, num_parallel_calls=self.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=readfile_batch_size * shuffer_buffer)
        # dataset = dataset.repeat(epoch)
        dataset = dataset.batch(readfile_batch_size)
        dataset = dataset.prefetch(self.AUTOTUNE)
        return dataset

    # def get_batched_dataset(filenames):
    #     option_no_order = tf.data.Options()
    #     option_no_order.experimental_deterministic = False
    #
    #     dataset = tf.data.Dataset.list_files(filenames)
    #     dataset = dataset.with_options(option_no_order)
    #     dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=AUTO)
    #     dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)
    #
    #     dataset = dataset.cache() # This dataset fits in RAM
    #     dataset = dataset.repeat()
    #     dataset = dataset.shuffle(2048)
    #     dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    #     dataset = dataset.prefetch(AUTO) #
    #
    #     return dataset

    def prepare_sample(self, data_dict):
        feature = data_dict["feature"]
        feature = tf.io.decode_raw(feature, tf.float32)
        feature = tf.reshape(feature, (-1, self.n_feature))

        target = data_dict["target"]
        target = tf.io.decode_raw(target, tf.float32)
        target = tf.reshape(target, (-1, self.n_target))

        if self.area_weight:
            weight = data_dict["weight"]
            weight = tf.io.decode_raw(weight, tf.float32)
            weight = tf.reshape(weight, (-1, 1))
            return feature, target, weight
        else:
            return feature, target

    def decode_fn(self, example):


        tfrecord_format = {"feature": tf.io.FixedLenFeature([], tf.string),
                           "target": tf.io.FixedLenFeature([], tf.string )}
        if self.area_weight:
            tfrecord_format["weight"] = tf.io.FixedLenFeature([], tf.string )
        return tf.io.parse_single_example(example,tfrecord_format)

        # tfrecord_format = {"feature": tf.io.FixedLenSequenceFeature([], tf.string ,allow_missing=True),
        #                    "target": tf.io.FixedLenSequenceFeature([], tf.string ,allow_missing=True)}
        # if self.area_weight:
        #     tfrecord_format["weight"] = tf.io.FixedLenSequenceFeature([], tf.string ,allow_missing=True)
        # return tf.io.parse_single_example(example,tfrecord_format)


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)