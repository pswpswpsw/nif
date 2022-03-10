import sys

sys.path.append("../../")

from nif import *
import numpy as np
import time
import logging
import contextlib
from matplotlib import pyplot as plt

# user specified options
# cfg_shape_net = {
#     "connectivity": 'full',
#     "input_dim": 1,
#     "output_dim": 1,
#     "units": 30,
#     "nlayers": 2,
#     "use_resblock": False,
#     "activation": 'swish'
# }
# cfg_parameter_net = {
#     "input_dim": 1,
#     "latent_dim": 1,
#     "units": 30,
#     "nlayers": 2,
#     "nn_type": 'mlp',
#     "activation": 'swish',
#     "use_resblock": False
# }
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

enable_multi_gpu = False
enable_mixed_precision = False
nepoch = int(1e1)  # int(1e5)
lr = 2e-4
batch_size = 1024
checkpt_epoch = 10000
display_epoch = 1000
print_figure_epoch = 5000

# get training data set
train_data = np.load('./data/train.npz')['data']
num_total_data = train_data.shape[0]
train_dataset = tf.data.Dataset.from_tensor_slices((train_data[:, :2], train_data[:, -1:]))
train_dataset = train_dataset.shuffle(num_total_data).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# mixed precision?
if enable_mixed_precision:
    mixed_policy = "mixed_float16"
    policy = mixed_precision.Policy(mixed_policy)
    mixed_precision.set_global_policy(policy)
else:
    mixed_policy = None

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
    model_ori = NIF(cfg_shape_net, cfg_parameter_net, mixed_policy)
    model = model_ori.model()
    # model_pnet = model_ori.model_pnet()
    # latent_to_p_model = model_ori.model_latent_to_weights()
    model.summary()
    model.compile(optimizer, loss_fun)

# callbacks = []
callbacks = [LossAndErrorPrintingCallback()]
# callbacks = [tensorboard_callback, ]
# callbacks = [tensorboard_callback, LossAndErrorPrintingCallback(),scheduler_callback]
model.fit(train_dataset, epochs=nepoch, batch_size=batch_size, shuffle=False, verbose=0, callbacks=callbacks,
          use_multiprocessing=True)
