import tensorflow as tf
import numpy as np

# get the numpy datasets
data = np.random.uniform(0, 1, (1000000, 6))
print(data.shape)


def serialize_example(f0, f1):
    return tf.train.Example(features=tf.train.Features(feature={
        'feature0': tf.train.Feature(float_list=tf.train.FloatList(value=f0)),
        'feature1': tf.train.Feature(float_list=tf.train.FloatList(value=f1)),
    })).SerializeToString()


features_dataset = tf.data.Dataset.from_tensor_slices((data[:, :3], data[:, 3:6]))


def generator():
    for features in features_dataset:
        yield serialize_example(*features)


serialized_features_dataset = tf.data.Dataset.from_generator(generator, output_types=tf.string, output_shapes=())

filename = 'test.tfrecord'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(serialized_features_dataset)

filename = 'test2.tfrecord'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(serialized_features_dataset)

# # get the example
# serialized_example = serialize_example(data)
#
# example_proto = tf.train.Example.FromString(serialized_example)
# # print(example_proto)
#
#
# filename = 'test.tfrecord'
# with tf.io.TFRecordWriter(filename) as writer:
#     for i in range(1000):
#         example = serialize_example(data[i,:])
#         writer.write(example)
#         # print(example)
#
# filename = 'test2.tfrecord'
# with tf.io.TFRecordWriter(filename) as writer:
#     for i in range(1000):
#         example = serialize_example(data[i,:])
#         writer.write(example)
#         # print(example)
