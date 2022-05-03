from nif.data.tfr_dataset import TFRDataset
import sys
import time
import logging

# create tfr from npz
# fh = TFRDataset(n_feature=3, n_target=3)
# fh.create_from_npz(num_pts_per_file=1e6, npz_path='test2.npz', npz_key='data', write_tfr_path='mytest', prefix="case1")
#
# sys.exit()

# get tf.dataset from tfr
fh = TFRDataset(n_feature=3, n_target=3)
meta_dataset = fh.get_tfr_meta_dataset(path='mytest', epoch=2)

import tensorflow as tf

x = tf.keras.Input(3)
l1 = tf.keras.layers.Dense(20, activation='swish')
l2 = tf.keras.layers.Dense(20, activation='swish')
l3 = tf.keras.layers.Dense(3)
h = l1(x)
h = l2(h)
y = l3(h)
model = tf.keras.Model([x], [y])

batch_size = 128


class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass


callbacks = [LossAndErrorPrintingCallback()]

model.compile(optimizer='adam', loss='mse')

for batch_file in meta_dataset:
    batch_dataset = fh.gen_dataset_from_batch_file(batch_file, batch_size)
    history = model.fit(batch_dataset, verbose=1, callbacks=callbacks)
