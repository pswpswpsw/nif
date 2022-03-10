import sys
sys.path.append("../../")

import numpy as np
from matplotlib import pyplot as plt
from nif import *
from nif.optimizers import TFPLBFGS

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
mixed_policy = None
lr = 2e-4
optimizer = tf.keras.optimizers.Adam(lr)
loss_fun = tf.keras.losses.MeanSquaredError()
model_ori = NIF(cfg_shape_net, cfg_parameter_net, mixed_policy)
model = model_ori.model()
latent_model = model_ori.model_pnet()
latent_to_p_model = model_ori.model_latent_to_weights()
model.summary()
model.compile(optimizer, loss_fun)

# load weights
model.load_weights("./ckpt-0/ckpt")
# load_status = model.load_weights("./fine-tuned/ckpt")

# read data from disk
train_data = np.load('./data/train.npz')['data']  # t,x,u
inps, outs = train_data[:, :2], train_data[:, -1:]

# fine tuning
# data + keras model -> function for l-bfgs
fine_tuner = TFPLBFGS(model, loss_fun, inps, outs)
fine_tuner.minimize(2, max_iter=10)
history = fine_tuner.history
model.save_weights("./fine-tuned/ckpt")

plt.figure()
plt.plot(history['iteration'], history['loss'])
plt.xlabel('iteration')
plt.ylabel('loss')
plt.savefig('./fine_tune_loss.png')

