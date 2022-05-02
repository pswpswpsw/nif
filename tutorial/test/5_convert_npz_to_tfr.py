from nif.data.tfr_dataset import TFRDataset
import sys

# # create tfr from npz
# fh = TFRDataset(n_feature=3, n_target=1)
# fh.create_from_npz(num_pts_per_file=3000, npz_path='test.npz', npz_key='data', write_tfr_path='mytest')
#
# sys.exit()

# get tf.dataset from tfr
fh = TFRDataset(n_feature=3, n_target=1)
dataset = fh.get_dataset(path='mytest', readfile_batch_size=20)


import tensorflow as tf
x = tf.keras.Input(3)
l1 = tf.keras.layers.Dense(20,activation='swish')
l2 = tf.keras.layers.Dense(20,activation='swish')
l3 = tf.keras.layers.Dense(1)
h = l1(x)
h = l2(h)
y = l3(h)
model = tf.keras.Model([x],[y])


model.compile(optimizer='adam',loss='mse')

model.fit(dataset, epochs=10)

# for i in range(10):
#     z1,z2 = next(iter(dataset))
#     print(z1)
#     print(z2)