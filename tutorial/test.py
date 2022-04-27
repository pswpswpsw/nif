import tensorflow as tf
import nif
import numpy as np
import time
import logging
import contextlib
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from nif.optimizers import gtcf


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

# mixed precision?
enable_mixed_precision=False
if enable_mixed_precision:
    mixed_policy = "mixed_float16"
    # we might need this for `model.fit` to automatically do loss scaling
    policy = nif.mixed_precision.Policy(mixed_policy)
    nif.mixed_precision.set_global_policy(policy)
else:
    mixed_policy = 'float32'

# mixed_policy = 'float32'
new_model_ori = nif.NIF(cfg_shape_net, cfg_parameter_net, mixed_policy)

#new_model = new_model_ori.model()
new_model = new_model_ori.build()
loss_fun = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(1e-5)
new_model.compile(optimizer, loss_fun)

new_model.load_weights("./saved_weights/ckpt-4999/ckpt")

from nif.optimizers import TFPLBFGS
from nif.demo import TravelingWave
tw = TravelingWave()
train_data = tw.data

data_feature = (train_data[:,:1], train_data[:,1:2])
data_label = train_data[:,-1:]

fine_tuner = TFPLBFGS(new_model, loss_fun, data_feature, data_label, display_epoch=10)

fine_tuner.minimize(rounds=200, max_iter=1000)
new_model.save_weights("./fine-tuned/ckpt")

history = fine_tuner.history
plt.figure(figsize=(8,2))
plt.semilogy(history['iteration'], history['loss'],'k-o')
plt.ylim([1e-5,1e-2])
plt.xlabel('iteration')
plt.ylabel('loss')
plt.savefig('./fine_tune_loss.png')
