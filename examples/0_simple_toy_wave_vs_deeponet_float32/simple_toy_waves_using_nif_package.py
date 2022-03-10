import sys
sys.path.append("../../")

import tensorflow as tf
from tensorflow.keras import mixed_precision
from nif import NIFMultiScale, NIF
import numpy as np
import time
import logging
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

multi_gpu = True
gpus = tf.config.experimental.list_physical_devices('GPU')

if len(gpus) > 0:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_physical_devices('GPU')
    # logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

enable_mixed_precision = False

if enable_mixed_precision:
    mixed_policy = "mixed_float16"
    policy = mixed_precision.Policy(mixed_policy)
    mixed_precision.set_global_policy(policy)
else:
    mixed_policy = None


# tf.keras.utils.plot_model(model,"./framework.png", show_shapes=True) ## currently keras didn't support tf.einsum
# for plotting a diagram.

nepoch = int(8e5)
lr = 2e-4
batch_size = 1024*4
checkpt_epoch = 1000
display_epoch = 500
print_figure_epoch = 500

# get training data set
train_data = np.load('./data/train.npz')['data']
num_total_data = train_data.shape[0]
train_dataset = tf.data.Dataset.from_tensor_slices((train_data[:, :2], train_data[:, -1:]))
train_dataset = train_dataset.shuffle(num_total_data).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

## define loss objective - MSE of course
# loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

## creat loss function
## parallel
# def compute_loss(labels, predictions):
#     per_example_loss = loss_object(labels, predictions)
#     return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

## serial
# compute_loss = loss_object # (labels, predictions)
# loss_object = tf.keras.losses.MeanSquaredError()


# try just pure model subclassing

logging.basicConfig(filename='./logging',level=logging.INFO,format='%(message)s')

class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.train_begin_time = time.time()
        self.history_loss = []
    def on_epoch_begin(self, epoch, logs=None):
        self.ts = time.time()
    def on_epoch_end(self, epoch, logs=None):
        if epoch % display_epoch == 0:
            tnow = time.time()
            te = tnow- self.ts
            logging.info("Epoch {:6d}: avg.loss pe = {:4.3e}, {:d} points/sec, time elapsed = {:4.3f} hours".format(epoch,
                         logs['loss'], int(batch_size/te), (tnow - self.train_begin_time)/3600.0))
            self.history_loss.append(logs['loss'])
        if epoch % print_figure_epoch == 0:
            plt.figure()
            plt.semilogy(self.history_loss)
            plt.xlabel('epoch: per {} epochs'.format(print_figure_epoch))
            plt.ylabel('MSE loss')
            plt.savefig('./loss.png')
            plt.close()

class CustomSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % checkpt_epoch == 0:  # or save after some epoch, each k-th epoch etc.
            self.model.save("./ckpt_{}/".format(epoch))
model_checkpoint_callback = CustomSaver()

# def scheduler(epoch, lr):
#     if epoch < 2000:
#         return lr
#     else:
#         return 1e-5
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./tb-logs",
                                                      update_freq='epoch')
# scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

# model = NIFMultiScale(cfg_shape_net, cfg_parameter_net)
# model = NIFMultiScale(cfg_shape_net, cfg_parameter_net).model()

if multi_gpu:
    strategy = tf.distribute.MirroredStrategy()
    print("number of GPU = {}".format(strategy.num_replicas_in_sync))
    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(lr)
        model = NIF(cfg_shape_net, cfg_parameter_net, mixed_policy).model()
        model.summary()
        model.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
else:
    optimizer = tf.keras.optimizers.Adam(lr)
    model = NIF(cfg_shape_net, cfg_parameter_net, mixed_policy).model()
    model.summary()
    model.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())

# model.fit(train_dataset,
#           epochs=1,
#           batch_size=batch_size,
#           shuffle=True,
#           verbose=0,
#           callbacks=[tensorboard_callback, LossAndErrorPrintingCallback(), scheduler_callback,
#                      model_checkpoint_callback],
#           use_multiprocessing=True)

model.fit(train_dataset,
          epochs=nepoch,
          batch_size=batch_size,
          shuffle=False,
          verbose=0,
          callbacks=[tensorboard_callback, LossAndErrorPrintingCallback(), # scheduler_callback,
                     model_checkpoint_callback],
          use_multiprocessing=True)

# ## construct a single training step
# @tf.function
# def train_step(inputs):
#     features, target = inputs
#     with tf.GradientTape() as tape:
#         predictions = model(features)
#         # loss = compute_loss(target, predictions)
#         loss = loss_object(target, predictions)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     return loss
#
# # add tensorboard callbacks
# tb_callback = tf.keras.callbacks.TensorBoard(log_dir="./tb-logs",update_freq=1, histogram_freq=1, write_graph=False)
# tb_callback.set_model(model)
#
# # training loop
# for epoch in range(0, nepoch):
#     # print("Started of epoch %d" %(epoch,))
#
#     # iterative over batches of dataset
#     avg_epoch_loss = 0
#     n_b = 0
#     for train_feature_target in train_dataset:
#         avg_epoch_loss += train_step(train_feature_target)
#         n_b+=1
#     avg_epoch_loss /= n_b
#
#     if epoch % 100 == 0:
#         print("Epoch %5d - Average epoch loss = %8.7E" % (epoch,avg_epoch_loss))


