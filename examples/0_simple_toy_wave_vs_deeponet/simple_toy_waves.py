
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import GPUtil

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Concatenate
# import tensorflow.compat.v1 as tf

# tf.disable_v2_behavior()
import time
import logging
# from batchup import data_source
import os
import argparse

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# DEVICE_ID_LIST = GPUtil.getFirstAvailable(maxLoad=0.05, maxMemory=0.05)
# os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID_LIST[0])

parser = argparse.ArgumentParser(description='config')
parser.add_argument('--TRAIN_DATA',type=str, help='normalized train data file path')
parser.add_argument('--NETWORK_TYPE',type=str, help='model type')
parser.add_argument('--N_S',type=int, help='width of shapenet/MLP')
parser.add_argument('--RANK',type=int, help='number of ranks')
parser.add_argument('--N_T',type=int, help='width of parameternet')
parser.add_argument('--ACT',type=str, help='activation function type')
parser.add_argument('--L_R',type=float, help='learning rate')
parser.add_argument('--BATCH_SIZE',type=int, help='batch size')
parser.add_argument('--EPOCH',type=int, help='total epoch')
args = parser.parse_args()

TRAIN_DATA = np.load(args.TRAIN_DATA)['data']
NETWORK_TYPE = args.NETWORK_TYPE
N_t = args.N_T
N_s = args.N_S
N_p = args.RANK
ACT_STR = args.ACT
learning_rate = args.L_R
batch_size = args.BATCH_SIZE
nepoch = args.EPOCH

if ACT_STR == 'swish':
    ACT = tf.nn.swish
elif ACT_STR == 'tanh':
    ACT = tf.nn.tanh
elif ACT_STR == 'relu':
    ACT = tf.nn.relu
else:
    raise NotImplementedError

def mkdir(CASE_NAME):
    if not os.path.exists(CASE_NAME):
        os.makedirs(CASE_NAME)

CASE_NAME = '{}_RANK_{}_NSX_{}_NST_{}_ACT_{}'.format(NETWORK_TYPE, N_p, N_s, N_t, ACT_STR)
mkdir(CASE_NAME)
mkdir(CASE_NAME + '/pngs')

# gpu_options = tf.GPUOptions(allow_growth=True)

logging.basicConfig(filename=CASE_NAME+'/log',level=logging.INFO,format='%(message)s')

# print(tf.__version__)

# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
num_gpus = 1
# nepoch = 40001
# learning_rate = 1e-3
# batch_size = 32768 # 2048 # 512
display_epoch = 100
print_figure_epoch = 200
checkpt_epoch = 1000 # nepoch - 1

# Shape Network Parameters
# N_s = 100 # 100
L_s = 3

## TMP
if NETWORK_TYPE == 'DEEPONET':
    Total_para = (L_s-2)*(N_s+1)*N_s + 2*N_s + 1 + N_p + (N_s + 1)*N_p
else:
    Total_para = (L_s-1)*(N_s+1)*N_s + 3*N_s + 1

logging.info('total number of shape net = ' + str(Total_para))

# Logging hyperparameters
logging.info("Number of GPUs = " + str(num_gpus))
logging.info("Total epoch = " + str(nepoch))
logging.info("Learnign rate = " + str(learning_rate))
logging.info("Batch size = " + str(batch_size))
logging.info("\n")


# Build a convolutional neural network
init_weight = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.1)

if NETWORK_TYPE == 'MLP':

    input = keras.Input((2,), name='input_t_x')

    l_1 = Dense(N_s, activation=ACT, kernel_initializer=init_weight, bias_initializer=init_weight, name='layer_1')
    l_2 = Dense(N_s, activation=ACT, kernel_initializer=init_weight, bias_initializer=init_weight, name='layer_2')
    l_3 = Dense(N_s, activation=ACT, kernel_initializer=init_weight, bias_initializer=init_weight, name='layer_3')
    l_4 = Dense(1, activation=None, kernel_initializer=init_weight, bias_initializer=init_weight, name='layer_4')

    u = l_1(input)
    u = u + l_2(u)
    u = u + l_3(u)
    u = l_4(u)

    model = keras.Model(input, u)

elif NETWORK_TYPE == 'DEEPONET':

    # build parameter net
    input_p = keras.Input((1,), name='input_p')
    lt_1 = Dense(N_t, activation=ACT, kernel_initializer=init_weight, bias_initializer=init_weight, name='paranet_layer_1')
    lt_2 = Dense(N_t, activation=ACT, kernel_initializer=init_weight, bias_initializer=init_weight, name='paranet_layer_2')
    lt_3 = Dense(N_p+1, kernel_initializer=init_weight, bias_initializer=init_weight, name='output_paranet')

    # build shapenet
    input_s = keras.Input((1,), name='input_s')
    lx_1 = Dense(N_s, activation=ACT, kernel_initializer=init_weight, bias_initializer=init_weight, name='shapenet_layer_1')
    lx_2 = Dense(N_s, activation=ACT, kernel_initializer=init_weight, bias_initializer=init_weight, name='shapenet_layer_2')
    lx_3 = Dense(N_p, kernel_initializer=init_weight, bias_initializer=init_weight, name='shapenet_layer_3')

    # last layer of shapenet needs to be built
    lx_4 = Dense(1, kernel_initializer=init_weight, bias_initializer=init_weight, name='shapenet_layer_4')


    # lx_4.built = True



    # construct parameter net
    p = lt_1(input_p)
    p = lt_2(p)
    p = lt_3(p)

    # distribute the weight and bias from parameter net
    w0 = p[:,0:N_p]
    b0 = p[:,N_p:N_p+1]
    w0 = tf.reshape(w0, (N_p, 1))
    b0 = tf.reshape(b0, (1,))

    # construct shapenet
    lx_4.kernel = w0
    lx_4.bias = b0

    # shapenet = keras.Sequential([lx_1, lx_2, lx_3, lx_4])

    u = lx_1(input_s)
    u = u + lx_2(u)
    u = lx_3(u)
    u = lx_4(u)

    # shapenet.layers[-1].kernel = w0
    # shapenet.layers[-1].bias = b0


    # input = Concatenate()([input_p, input_s])  # shape = (,2)
    # model = keras.Model(input, u)


    model = keras.Model([input_p,input_s], u)


    # distribute the weight and bias into shapenet
    # model.layers[-1].kernel=w0
    # model.layers[-1].bias=b0
    # lx_4.kernel = w0
    # lx_4.bias = b0

    # def NETWORK(t, x, reuse):
    #     # Define a scope for reusing the variables
    #     with tf.variable_scope('NIN', reuse=reuse):
    #
    #         # Build "parameter net"
    #         para_net = t # tf.concat([t],axis=-1)
    #         para_net = tf.layers.dense(para_net, N_t, activation=ACT,
    #                                    kernel_initializer=init_weight, bias_initializer=init_weight)
    #         para_net = tf.layers.dense(para_net, N_t, activation=ACT,
    #                                    kernel_initializer=init_weight, bias_initializer=init_weight)
    #         # para_net = tf.layers.dense(para_net, 1,  activation=ACT,
    #         #                            kernel_initializer=init_weight, bias_initializer=init_weight)
    #         para_net = tf.layers.dense(para_net, N_p+1, activation=None,
    #                                    kernel_initializer=init_weight, bias_initializer=init_weight)
    #         # # Collect para-net
    #         # para_net = tf.identity(para_net, name="para_net")
    #
    #         phi_x = tf.layers.dense(x, N_s, activation=ACT, kernel_initializer=init_weight, bias_initializer=init_weight)
    #         phi_x = phi_x + tf.layers.dense(phi_x, N_s, activation=ACT, kernel_initializer=init_weight, bias_initializer=init_weight)
    #         phi_x = tf.layers.dense(phi_x, N_p, activation=ACT, kernel_initializer=init_weight, bias_initializer=init_weight)
    #
    #         # final layer
    #         weight_final = para_net[:,0:N_p]
    #         weight_final = tf.reshape(weight_final, shape=[-1, N_p,  1])
    #         bias_final = para_net[:,-1]
    #         bias_final = tf.reshape(bias_final, shape=[-1, 1])
    #
    #         u = tf.einsum('ai,aij->aj', phi_x, weight_final) + bias_final
    #
    #     return u

elif NETWORK_TYPE == 'NIF':
    def NETWORK(t, x, reuse):
        # Define a scope for reusing the variables
        with tf.variable_scope('NIN', reuse=reuse):

            # Build "parameter net"
            para_net = t # tf.concat([t],axis=-1)
            para_net = tf.layers.dense(para_net, N_t, activation=ACT,
                                       kernel_initializer=init_weight, bias_initializer=init_weight)
            para_net = tf.layers.dense(para_net, N_t, activation=ACT,
                                       kernel_initializer=init_weight, bias_initializer=init_weight)
            para_net = tf.layers.dense(para_net, 1,  activation=None,
                                       kernel_initializer=init_weight, bias_initializer=init_weight)
            para_net = tf.layers.dense(para_net, Total_para, activation=None,
                                       kernel_initializer=init_weight, bias_initializer=init_weight)
            # # Collect para-net
            # para_net = tf.identity(para_net, name="para_net")

            # Distribute to weight and biases
            weight_1 = para_net[:,0:N_s];                           weight_1 = tf.reshape(weight_1, shape=[-1, 1,  N_s])
            weight_2 = para_net[:,N_s:((N_s+1)*N_s)];               weight_2 = tf.reshape(weight_2, shape=[-1, N_s, N_s])
            weight_3 = para_net[:,(N_s**2+N_s):(2*N_s**2+N_s)];     weight_3 = tf.reshape(weight_3, shape=[-1, N_s, N_s])
            weight_4 = para_net[:,(2*N_s**2+N_s):(2*N_s**2+2*N_s)]; weight_4 = tf.reshape(weight_4, shape=[-1, N_s, 1 ])
            bias_1   = para_net[:,(2*N_s**2+2*N_s):(2*N_s**2+3*N_s)]; bias_1   = tf.reshape(bias_1,   shape=[-1, N_s])
            bias_2   = para_net[:,(2*N_s**2+3*N_s):(2*N_s**2+4*N_s)]; bias_2   = tf.reshape(bias_2,   shape=[-1, N_s])
            bias_3   = para_net[:,(2*N_s**2+4*N_s):(2*N_s**2+5*N_s)]; bias_3   = tf.reshape(bias_3,   shape=[-1, N_s])
            bias_4   = para_net[:,(2*N_s**2+5*N_s):];                 bias_4   = tf.reshape(bias_4,   shape=[-1, 1])

            # Build "shape net"
            u = ACT(tf.einsum('ai,aij->aj', x, weight_1) + bias_1)
            u = ACT(tf.einsum('ai,aij->aj', u, weight_2) + bias_2) + u
            u = ACT(tf.einsum('ai,aij->aj', u, weight_3) + bias_3) + u
            u = tf.einsum('ai,aij->aj',u, weight_4) + bias_4

        return u

model.summary()
keras.utils.plot_model(model, CASE_NAME+"/framework.png", show_shapes=True)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss = keras.losses.MSE,
    metrics=keras.metrics.MeanSquaredError()
)


class LossAndErrorPrintingCallback(keras.callbacks.Callback):
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
            plt.savefig(CASE_NAME+'/loss.png')
            plt.close()

class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % checkpt_epoch == 0:  # or save after some epoch, each k-th epoch etc.
            self.model.save(CASE_NAME+"/ckpt_{}/".format(epoch))
model_checkpoint_callback = CustomSaver()

def scheduler(epoch, lr):
    if epoch < 2000:
        return lr
    else:
        return 5e-4
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=CASE_NAME+"/tb-logs",update_freq=display_epoch, histogram_freq=1, write_graph=False)
scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

model.fit(x=TRAIN_DATA[:,0:2],
          y=TRAIN_DATA[:,-1],
          epochs=nepoch,
          batch_size=batch_size,
          shuffle=True,
          verbose=0,
          callbacks=[tensorboard_callback, LossAndErrorPrintingCallback(),scheduler_callback,model_checkpoint_callback],
          use_multiprocessing=True)

y_pred = model.predict(TRAIN_DATA[:10,0:2])
print(y_pred)
print(y_pred.shape)


# # tf Graph input
# # X = tf.placeholder(tf.float32, [None, num_input])
# # Y = tf.placeholder(tf.float32, [None, num_classes])
# X  = tf.placeholder(tf.float32, [None, 1],name='input_X')
# # NU = tf.placeholder(tf.float32, [None, 1],name='input_NU')
# T  = tf.placeholder(tf.float32, [None, 1],name='input_T')
# U  = tf.placeholder(tf.float32, [None, 1])
#
#
# # Keep training until reach max iterations
# for epoch in range(1, nepoch + 1):
#     total_batch_size = num_gpus * batch_size
#     # for (batch_nu, batch_x, batch_t, batch_u) in ds.batch_iterator(batch_size=total_batch_size,
#     #                                                                shuffle=np.random.RandomState(epoch)):
#     for batch in iterate_minibatches(TRAIN_DATA[:,0:2], TRAIN_DATA[:,-1], batch_size, shuffle=True):
#         feature_batch, batch_u = batch
#         # batch_nu = feature_batch[:,[0]]
#         batch_x  = feature_batch[:,[0]]
#         batch_t  = feature_batch[:,[1]]
#         batch_u = batch_u.reshape(-1,1)
#
#         ts = time.time()
#         # Get a batch for each GPU
#         # batch_x, batch_y = mnist.train.next_batch(batch_size * num_gpus)
#
#         # Run optimization op (backprop)
#         sess.run(train_op, feed_dict={X: batch_x, T:batch_t, U: batch_u})
#         te = time.time() - ts
#
#     if epoch % display_epoch == 0 or epoch == 1:
#         # Calculate batch loss and accuracy
#         loss = sess.run([loss_op], feed_dict={X: batch_x, T:batch_t, U: batch_u})[0]
#         loss_all = sess.run([loss_op_all], feed_dict={X: TRAIN_DATA[:,[0]], T:TRAIN_DATA[:,[1]], U: TRAIN_DATA[:,[2]]})[0]
#         # print("Epoch " + str(epoch) + ": Minibatch Loss= " + "{:.8f}".format(loss) +" Total Loss ="+ "{:.8f}".format(loss_all) +  ", %i Data Points/sec" % int(len(batch_x)/te))
#         # Record Train mse loss
#         TRAIN_MSE_LIST.append(loss_all)
#
#         # Log to file
#         T_CURRENT = time.time()
#         logging.info("Epoch " + str(epoch) + ": Minibatch Loss= " + "{:.8f}".format(loss) +" Total Loss ="+ "{:.8f}".format(loss_all) +  ", %i Data Points/sec" % int(len(batch_x)/te))
#         logging.info('Time elapsed: '+str((T_CURRENT-T_START)/3600.0)+" Hours..")
#         # logging.info('\n')
#
#         plt.figure()
#         plt.semilogy(np.array(TRAIN_MSE_LIST))
#         plt.xlabel('epoch')
#         plt.ylabel('MSE loss')
#         plt.savefig('loss.png')
#         plt.close()
#
#     if epoch % checkpt_epoch == 0:
#         # save model at check point
#         tf.saved_model.simple_save(sess,
#                                    './saved_model_ckpt_'+str(epoch)+'/',
#                                    inputs={"t": T, "x": X, "u": U},
#                                    outputs={"eval_output_u": eval_output_u})
#
#     epoch += 1
# logging.info("Optimization Finished!")
# T_END = time.time()
#
# logging.info('Time elapsed: ' + str((T_END-T_START)/3600.0)+ " Hours")
# # End of training: save loss and save reconstruction
#
# logging.info('Training data size = '+ str(TRAIN_DATA.shape) )
# u_rec = sess.run([eval_output_u], feed_dict={X: TRAIN_DATA[:,[0]], T:TRAIN_DATA[:,[1]], U: TRAIN_DATA[:,[2]]})[0]
# # print(u_rec)
# # print(u_rec.shape)
# np.savez('mse_saved.npz',loss=np.array(TRAIN_MSE_LIST), time=T_END-T_START, u_rec=u_rec)

