import tensorflow as tf
import numpy as np

filenames = ['test.tfrecord', 'test2.tfrecord']

# decoder of the tfrecord
feature_description = {
    'feature0': tf.io.FixedLenFeature([3], tf.float32, default_value=[0.0] * 3),
    'feature1': tf.io.FixedLenFeature([3], tf.float32, default_value=[0.0] * 3)
}


def _parse_function(example_proto):
    example = tf.io.parse_single_example(example_proto, feature_description)
    return example["feature0"], example["feature1"]


# now we get the dataset we knew
raw_dataset = tf.data.TFRecordDataset(filenames)

# print(raw_dataset)
# for parsed_record in raw_dataset.take(10):
#     print(repr(parsed_record))
AUTOTUNE = tf.data.experimental.AUTOTUNE
parsed_dataset = raw_dataset.map(_parse_function, num_parallel_calls=AUTOTUNE)
parsed_dataset = parsed_dataset.shuffle(buffer_size=1024).batch(256).prefetch(AUTOTUNE)

for data_x in parsed_dataset:
    # print(repr(parsed_record))
    print(data_x)
    # print(data_y.shape)

## model training

import tensorflow as tf

x = tf.keras.Input(3)
l1 = tf.keras.layers.Dense(5, activation='swish')
l2 = tf.keras.layers.Dense(5, activation='swish')
l3 = tf.keras.layers.Dense(3)

h = l1(x)
h = l2(h)
y = l3(h)
model = tf.keras.Model(inputs=[x], outputs=[y])

optimizer = tf.keras.optimizers.Adam(1e-3)
model.compile(optimizer=optimizer, loss='mse')

model.fit(parsed_dataset, epochs=10)
