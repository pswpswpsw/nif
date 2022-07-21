# if you want single GPU
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import tensorflow as tf

# import tensorflow_model_optimization as tfmot
import nif
import numpy as np
import time
import logging
import contextlib
import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt

lr = 1e-5
nepoch = 1000
batch_size = 1024
enable_multi_gpu = False
enable_mixed_precision = False
checkpt_epoch = 10
display_epoch = 5
print_figure_epoch = 10
enable_sobolov = True  # False
n_output = 9

cfg_shape_net = {
    "use_resblock": False,
    "connectivity": "full",
    "input_dim": 3,
    "output_dim": n_output,
    "units": 32,
    "nlayers": 4,
    "weight_init_factor": 0.01,
    "omega_0": 30.0,
}
cfg_parameter_net = {
    "use_resblock": False,
    "input_dim": 2,
    "latent_dim": 4,
    "units": 32,
    "nlayers": 4,
    "activation": "swish",
}


class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.train_begin_time = time.time()
        self.history_loss = []
        logging.basicConfig(filename="log", level=logging.INFO, format="%(message)s")

    def on_epoch_begin(self, epoch, logs=None):
        self.ts = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if epoch % display_epoch == 0:
            tnow = time.time()
            te = tnow - self.ts
            logging.info(
                "Epoch {:6d}: avg.loss pe = {:4.3e}, {:d} points/sec, time elapsed = {:4.3f} hours".format(
                    epoch,
                    logs["loss"],
                    int(batch_size / te),
                    (tnow - self.train_begin_time) / 3600.0,
                )
            )
            self.history_loss.append(logs["loss"])
        if epoch % print_figure_epoch == 0:
            plt.figure()
            plt.semilogy(self.history_loss)
            plt.xlabel("epoch: per {} epochs".format(print_figure_epoch))
            plt.ylabel("MSE loss")
            plt.savefig("loss.png")
            plt.close()

        if epoch % checkpt_epoch == 0 or epoch == nepoch - 1:
            print("save checkpoint epoch: %d..." % epoch)
            self.model.save_weights("saved_weights/ckpt-{}/ckpt".format(epoch))


# get data
train_data = np.load(
    "/home/shaowu/Desktop/Postdoc/PROJECTS/2022-oliver-pruning/3d-wing-flow/3d-wing-flow.npz"
)["data"]
np.random.shuffle(train_data)

num_total_data = train_data.shape[0]

if enable_sobolov:
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_data[:, :5], train_data[:, 5:])
    )
else:
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_data[:, :5], train_data[:, 5 : 5 + n_output])
    )
train_dataset = (
    train_dataset.shuffle(num_total_data)
    .batch(batch_size)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

# mixed precision?
if enable_mixed_precision:
    mixed_policy = "mixed_float16"
    # we might need this for `model.fit` to automatically do loss scaling
    policy = nif.mixed_precision.Policy(mixed_policy)
    nif.mixed_precision.set_global_policy(policy)
else:
    mixed_policy = "float32"

optimizer = tf.keras.optimizers.Adam(lr)
model_ori = nif.NIFMultiScale(cfg_shape_net, cfg_parameter_net, mixed_policy)
model = model_ori.build()

if enable_sobolov:

    x_index = [2, 3, 4]  # x,y,z
    y_index = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # we have 9 field output

    from nif.layers import JacobianLayer

    y_and_dydx = JacobianLayer(model, y_index, x_index)
    y, dy_dx = y_and_dydx(model.inputs[0])  ##  use[0] to make sure shape is good
    dy_dx_1d = tf.reshape(dy_dx, [-1, 3 * n_output])
    y_and_dydx_1d = tf.concat([y, dy_dx_1d], -1)
    model = tf.keras.Model([model.inputs[0]], [y_and_dydx_1d])

    class Sobolov_MSE(tf.keras.losses.Loss):
        def call(self, y_true, y_pred):
            sd_field = tf.square(y_true[:, :n_output] - y_pred[:, :n_output])
            sd_grad = tf.square(y_true[:, n_output:] - y_pred[:, n_output:])
            return tf.reduce_mean(sd_field, axis=-1) + 0.01 * tf.reduce_mean(
                sd_grad, axis=-1
            )

    model.compile(optimizer, loss=Sobolov_MSE())
else:
    model.compile(optimizer, loss="mse")
model.summary()

callbacks = [LossAndErrorPrintingCallback()]
model.fit(train_dataset, epochs=nepoch, callbacks=callbacks)
