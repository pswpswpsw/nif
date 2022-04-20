import sys
sys.path.append("../../")

# from nif import *
import tensorflow as tf
import nif
import numpy as np
import time
import logging
import contextlib
from matplotlib import pyplot as plt

cfg_shape_net = {
    "use_resblock":False,
    "connectivity": 'full',
    "input_dim": 1,
    "output_dim": 1,
    "units": 50,
    "nlayers": 2,
    "weight_init_factor": 0.001,
    "omega_0":30.0
}
cfg_parameter_net = {
    "use_resblock":False,
    # "nn_type":'mlp',
    "input_dim": 1,
    "latent_dim": 1,
    "units": 50,
    "nlayers": 2,
    "activation": 'swish'
}

enable_multi_gpu = False
enable_mixed_precision = False
nepoch = int(1e5)  # int(1e5)
lr = 1e-5
batch_size = 4096
checkpt_epoch = 10000
display_epoch = 1000
print_figure_epoch = 5000

# get training demo set
train_data = np.load('./data/train.npz')['demo']
num_total_data = train_data.shape[0]
train_dataset = tf.data.Dataset.from_tensor_slices((train_data[:, :2], train_data[:, -1:]))
train_dataset = train_dataset.shuffle(num_total_data).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# mixed precision?
if enable_mixed_precision:
    mixed_policy = "mixed_float16"
    # we might need this for `model.fit` to automatically do loss scaling
    policy = nif.mixed_precision.Policy(mixed_policy)
    nif.mixed_precision.set_global_policy(policy)
else:
    mixed_policy = 'float32'

class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.train_begin_time = time.time()
        self.history_loss = []
        logging.basicConfig(filename='./log', level=logging.INFO, format='%(message)s')

    def on_epoch_begin(self, epoch, logs=None):
        self.ts = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if epoch % display_epoch == 0:
            tnow = time.time()
            te = tnow - self.ts
            logging.info("Epoch {:6d}: avg.loss pe = {:4.3e}, {:d} points/sec, time elapsed = {:4.3f} hours".format(
                epoch, logs['loss'], int(batch_size / te), (tnow - self.train_begin_time) / 3600.0))
            self.history_loss.append(logs['loss'])
        if epoch % print_figure_epoch == 0:
            plt.figure()
            plt.semilogy(self.history_loss)
            plt.xlabel('epoch: per {} epochs'.format(print_figure_epoch))
            plt.ylabel('MSE loss')
            plt.savefig('./loss.png')
            plt.close()
        if epoch % checkpt_epoch == 0 or epoch == nepoch - 1:
            print('save checkpoint epoch: %d...' % epoch)
            self.model.save_weights("./saved_weights/ckpt-{}/ckpt".format(epoch))

def scheduler(epoch, lr):
    if epoch < 2000:
        return lr
    else:
        return 1e-5

scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./tb-logs", update_freq='epoch')

cm = tf.distribute.MirroredStrategy().scope() if enable_multi_gpu else contextlib.nullcontext()
with cm:
    optimizer = tf.keras.optimizers.Adam(lr)
    loss_fun = tf.keras.losses.MeanSquaredError()
    model_ori = nif.NIFMultiScale(cfg_shape_net, cfg_parameter_net, mixed_policy)
    # model_ori = nif.NIF(cfg_shape_net, cfg_parameter_net, mixed_policy)
    model = model_ori.model()
    # model_p_to_lr = model_ori.model_p_to_lr()
    # latent_to_p_model = model_ori.model_lr_to_w()
    model.summary()
    model.compile(optimizer, loss_fun)

callbacks = []
# callbacks = [LossAndErrorPrintingCallback()]
# callbacks = [tensorboard_callback, ]
# callbacks = [tensorboard_callback, LossAndErrorPrintingCallback(), scheduler_callback]
model.fit(train_dataset, epochs=nepoch, batch_size=batch_size, shuffle=False,
          verbose=2, callbacks=callbacks, use_multiprocessing=True)
