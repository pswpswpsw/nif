import sys
sys.path.append("../../")

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import nif

# load model
cfg_shape_net = {
    "connectivity": 'full',
    "input_dim": 1,
    "output_dim": 1,
    "units": 30,
    "nlayers": 2,
    "activation": 'swish'
}
cfg_parameter_net = {
    "input_dim": 1,
    "latent_dim": 1,
    "units": 30,
    "nlayers": 2,
    "activation": 'swish',
}

nepoch = 5
batch_size = 128
mixed_policy = 'float32'
lr = 1e-2 # 2e-4
optimizer = tf.keras.optimizers.Adam(lr)
loss_fun = tf.keras.losses.MeanSquaredError()
model_ori = nif.NIF(cfg_shape_net, cfg_parameter_net, mixed_policy)
model = model_ori.model()
model.summary()
model.compile(optimizer, loss_fun)

# get training demo set
train_data = np.load('./data/train.npz')['demo']
input_t = np.linspace(train_data[:,0].min(),train_data[:,0].max(), 100)
num_total_data = train_data.shape[0]
train_dataset = tf.data.Dataset.from_tensor_slices((train_data[:, :2], train_data[:, -1:]))
train_dataset = train_dataset.shuffle(num_total_data).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# restore weights
model.load_weights("./ckpt-50000/ckpt")
# model.load_weights("./ckpt-90000/ckpt")

# output p-lr model
model_p_to_lr = model_ori.model_p_to_lr()
lr_before = model_p_to_lr.predict(input_t)
# model_p_to_lr.save_weights("./p_to_lr/ckpt-50000/ckpt")

model_p_to_lr.save("./p_to_lr_model/")

# output shape model
model_x_to_u = model_ori.model_x_to_u_given_w()
model_x_to_u.save('./x_to_u_model/')

# train a bit
model.fit(train_dataset, epochs=nepoch, batch_size=batch_size, shuffle=False,
          verbose=1, use_multiprocessing=True)
