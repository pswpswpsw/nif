__all__ = ["NIFMultiScale", "NIF"]

import tensorflow as tf
from tensorflow.keras import Model, initializers
from .layers import *

class NIF(Model):
    def __init__(self, cfg_shape_net, cfg_parameter_net, mixed_policy=None):
        super(NIF, self).__init__()
        self.cfg_shape_net = cfg_shape_net
        self.si_dim = cfg_shape_net['input_dim']
        self.so_dim = cfg_shape_net['output_dim']
        self.n_sx = cfg_shape_net['units']
        self.l_sx = cfg_shape_net['nlayers']
        self.pi_dim = cfg_parameter_net['input_dim']
        self.pi_hidden = cfg_parameter_net['latent_dim']
        self.n_st = cfg_parameter_net['units']
        self.l_st = cfg_parameter_net['nlayers']
        self.mixed_policy = mixed_policy

        if self.mixed_policy == 'mixed_float16':
            self.Dtype = tf.float16
        elif self.mixed_policy == 'float64':
            self.Dtype = tf.float64
        else:
            self.Dtype = tf.float32

        self.pnet_list = self._initialize_pnet(cfg_parameter_net, cfg_shape_net)

    def _initialize_pnet(self, cfg_parameter_net, cfg_shape_net):
        # just simple implementation of a shortcut connected parameter net with a similar shapenet
        self.po_dim = (self.l_sx)*self.n_sx**2 + (self.si_dim + self.so_dim + 1 + self.l_sx)*self.n_sx + self.so_dim

        # construct parameter_net
        pnet_layers_list = []
        # 1. first layer
        layer_1 = Dense(
            self.n_st,
            cfg_parameter_net['activation'],
            kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
            bias_initializer=initializers.TruncatedNormal(stddev=0.1),
            # dtype=self.mixed_policy
        )
        pnet_layers_list.append(layer_1)

        # 2. hidden layer
        for i in range(self.l_st):
            tmp_layer = MLP_SimpleShortCut(
                self.n_st,
                cfg_parameter_net['activation'],
                kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                bias_initializer=initializers.TruncatedNormal(stddev=0.1),
                mixed_policy=self.mixed_policy
            )
            # identity_layer = Lambda(lambda x: x)
            # tmp_layer =tf.keras.layers.Add()(identity_layer,tmp_layer)
            pnet_layers_list.append(tmp_layer)

        # 3. bottleneck layer
        bottleneck_layer = Dense(
            self.pi_hidden,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
            bias_initializer=initializers.TruncatedNormal(stddev=0.1),
            # dtype=self.mixed_policy
        )
        pnet_layers_list.append(bottleneck_layer)

        # 4. last layer
        last_layer = Dense(
            self.po_dim,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
            bias_initializer=initializers.TruncatedNormal(stddev=0.1),
            # dtype=self.mixed_policy
        )
        pnet_layers_list.append(last_layer)
        return pnet_layers_list

    def call(self, inputs, training=None, mask=None):
        input_p = tf.cast(inputs[:, 0:self.pi_dim],self.Dtype)
        input_s = tf.cast(inputs[:, self.pi_dim:self.pi_dim+self.si_dim],self.Dtype)
        # get parameter from parameter_net
        self.pnet_output = self._call_parameter_net(input_p, self.pnet_list)[0]
        return self._call_shape_net(input_s, self.pnet_output)

    def _call_shape_net(self, input_s, pnet_output):
        w_1 = tf.reshape(pnet_output[:, :self.si_dim*self.n_sx],
                         [-1, self.si_dim, self.n_sx])
        w_hidden_list = []
        for i in range(self.l_sx):
            w_tmp = tf.reshape(pnet_output[:,
                               self.si_dim*self.n_sx + i*self.n_sx**2:
                               self.si_dim*self.n_sx + (i + 1)*self.n_sx**2],
                               [-1, self.n_sx, self.n_sx])
            w_hidden_list.append(w_tmp)
        w_l = tf.reshape(pnet_output[:,
                         self.si_dim*self.n_sx + self.l_sx*self.n_sx**2:
                         self.si_dim*self.n_sx + self.l_sx*self.n_sx**2 + self.so_dim*self.n_sx],
                         [-1, self.n_sx, self.so_dim])
        n_weights = self.si_dim*self.n_sx + self.l_sx*self.n_sx**2 + self.so_dim*self.n_sx

        # distribute bias
        b_1 = tf.reshape(pnet_output[:, n_weights: n_weights + self.n_sx],
                         [-1, self.n_sx])
        b_hidden_list = []
        for i in range(self.l_sx):
            b_tmp = tf.reshape(pnet_output[:, n_weights + self.n_sx + i*self.n_sx:
                                              n_weights + self.n_sx + (i + 1)*self.n_sx], [-1, self.n_sx])
            b_hidden_list.append(b_tmp)
        b_l = tf.reshape(pnet_output[:,
                         n_weights + (self.l_sx + 1)*self.n_sx:],
                         [-1, self.so_dim])

        # construct shape net
        act_fun = tf.keras.activations.get(self.cfg_shape_net['activation'])
        u = act_fun(tf.einsum('ai,aij->aj', input_s, w_1) + b_1)

        for i in range(self.l_sx):
            w_tmp = w_hidden_list[i]
            b_tmp = b_hidden_list[i]
            u = act_fun(tf.einsum('ai,aij->aj', u, w_tmp) + b_tmp) + u
        u = tf.einsum('ai,aij->aj', u, w_l) + b_l
        return tf.cast(u, tf.float32)

    def _call_parameter_net(self, input_p, pnet_list):
        latent = input_p
        for l in pnet_list[:-1]:
            latent = l(latent)
        output_final = pnet_list[-1](latent)
        return output_final, latent

    def model(self):
        input_s = tf.keras.layers.Input(shape=(self.si_dim + self.pi_dim))
        return Model(inputs=[input_s], outputs=[self.call(input_s)])

    def model_pnet(self):
        input_p = tf.keras.layers.Input(shape=(self.pi_dim))
        return Model(inputs=[input_p], outputs=[self._call_parameter_net(input_p, self.pnet_list)[1]])

    def model_latent_to_weights(self):
        input_l = tf.keras.layers.Input(shape=(self.pi_hidden))
        return Model([input_l],[self.pnet_list[-1](input_l)])

# TODO: test if it works...
class NIFMultiScale(NIF):
    def __init__(self, cfg_shape_net, cfg_parameter_net, mixed_policy=None):
        super(NIFMultiScale, self).__init__(cfg_shape_net, cfg_parameter_net, mixed_policy)

    def _initialize_pnet(self, cfg_parameter_net, cfg_shape_net):
        """
        generate the layers for parameter net, given configuration of
        shape_net you will also need the last layer to be consistent with
        the total number of shapenet' weights+biases
        """

        if not isinstance(cfg_parameter_net, dict):
            raise TypeError("cfg_parameter_net must be a dictionary")
        if not isinstance(cfg_shape_net, dict):
            raise TypeError("cfg_shape_net must be a dictionary")
        assert 'use_resblock' in cfg_shape_net.keys(), "use_resblock should be in cfg_shape_net"
        assert type(cfg_shape_net['use_resblock']) == bool, "cfg_shape_net['use_resblock'] must be a bool"

        pnet_layers_list = []
        # very first, determine the output dimension of parameter_net
        # TODO: now this is only for fully parameterized, still need option for last-layer-parameterization
        if cfg_shape_net['use_resblock']:
            self.po_dim = (2*self.l_sx)*self.n_sx**2 + (
                        self.si_dim + self.so_dim + 1 + 2*self.l_sx)*self.n_sx + self.so_dim
        else:
            self.po_dim = (self.l_sx)*self.n_sx**2 + (self.si_dim + self.so_dim + 1 + self.l_sx)*self.n_sx + self.so_dim

        # first, and hidden layers are only dependent on the type of parameter_net
        if cfg_parameter_net['nn_type'] == 'siren':
            assert cfg_parameter_net['activation'] == 'sine', "you should specify activation in cfg_parameter_net as " \
                                                              "sine"
            # 1. first layer
            layer_1 = SIREN(self.pi_dim, self.n_st, 'first',
                            cfg_parameter_net['omega_0'],
                            cfg_shape_net['omega_0'], cfg_shape_net,
                            self.mixed_policy)
            pnet_layers_list.append(layer_1)

            # 2. hidden layers
            if cfg_parameter_net['use_resblock']:
                for i in range(self.l_st):
                    tmp_layer = SIREN_ResNet(
                        self.n_st,
                        self.n_st,
                        cfg_parameter_net['omega_0'],
                        cfg_shape_net['omega_0'],
                        self.mixed_policy
                    )
                    pnet_layers_list.append(tmp_layer)
            else:
                for i in range(self.l_sx):
                    tmp_layer = SIREN(self.n_st, self.n_st, 'hidden',
                                      cfg_parameter_net['omega_0'],
                                      cfg_shape_net['omega_0'], cfg_shape_net,
                                      self.mixed_policy)
                    pnet_layers_list.append(tmp_layer)

            # 3. bottleneck layer
            bottleneck_layer = SIREN(self.n_st, self.pi_hidden, 'bottleneck',
                                     cfg_parameter_net['omega_0'],
                                     cfg_shape_net['omega_0'], cfg_shape_net,
                                     self.mixed_policy)
            pnet_layers_list.append(bottleneck_layer)

            # 4. last layer
            layer_last = SIREN(self.pi_hidden, self.po_dim, 'last',
                               cfg_parameter_net['omega_0'],
                               cfg_shape_net['omega_0'], cfg_shape_net,
                               self.mixed_policy)
            pnet_layers_list.append(layer_last)

        elif cfg_parameter_net['nn_type'] == 'mlp':
            # 1. first layer
            layer_1 = Dense(
                self.n_st,
                cfg_parameter_net['activation'],
                kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                bias_initializer=initializers.TruncatedNormal(stddev=0.1),
                # dtype=self.mixed_policy
            )
            pnet_layers_list.append(layer_1)

            # 2. hidden layer
            if cfg_parameter_net['use_resblock']:
                for i in range(self.l_st):
                    tmp_layer = MLP_ResNet(width=self.n_st,
                                           activation=cfg_parameter_net['activation'],
                                           kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                                           bias_initializer=initializers.TruncatedNormal(stddev=0.1),
                                           mixed_policy=self.mixed_policy)
                    pnet_layers_list.append(tmp_layer)
            else:
                for i in range(self.l_st):
                    tmp_layer = MLP_SimpleShortCut(
                        self.n_st,
                        cfg_parameter_net['activation'],
                        kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                        bias_initializer=initializers.TruncatedNormal(stddev=0.1),
                        mixed_policy=self.mixed_policy
                    )
                    # identity_layer = Lambda(lambda x: x)
                    # tmp_layer =tf.keras.layers.Add()(identity_layer,tmp_layer)
                    pnet_layers_list.append(tmp_layer)

            # 3. bottleneck layer
            bottleneck_layer = Dense(
                self.pi_hidden,
                kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                bias_initializer=initializers.TruncatedNormal(stddev=0.1),
                # dtype=self.mixed_policy
            )
            pnet_layers_list.append(bottleneck_layer)

            # 4. last layer
            last_layer = HyperLinearForSIREN(
                self.pi_hidden,
                self.po_dim,
                cfg_shape_net,
                self.mixed_policy
            )
            pnet_layers_list.append(last_layer)

        else:
            raise NotImplementedError(
                "no implementation for cfg_parameter_net['nn_type']=={}".format(cfg_parameter_net['nn_type']))

        return pnet_layers_list



    def _call_shape_net(self, input_s, pnet_output):
        """
        distribute `pnet_output` into weight and bias, it depends on the type of shapenet.

        For now, we only support shapenet having the following structure,
            - resnet-block
            - plain fnn

        """
        if self.cfg_shape_net['use_resblock']:
            # distribute weights
            w_1 = tf.reshape(pnet_output[:, :self.si_dim*self.n_sx],
                             [-1, self.si_dim, self.n_sx])
            w_hidden_list = []
            for i in range(self.l_sx):
                w1_tmp = tf.reshape(pnet_output[:,
                                    self.si_dim*self.n_sx + 2*i*self.n_sx**2:
                                    self.si_dim*self.n_sx + (2*i + 1)*self.n_sx**2],
                                    [-1, self.n_sx, self.n_sx])
                w2_tmp = tf.reshape(pnet_output[:,
                                    self.si_dim*self.n_sx + (2*i + 1)*self.n_sx**2:
                                    self.si_dim*self.n_sx + (2*i + 2)*self.n_sx**2],
                                    [-1, self.n_sx, self.n_sx])
                w_hidden_list.append([w1_tmp, w2_tmp])
            w_l = tf.reshape(pnet_output[:,
                             self.si_dim*self.n_sx + (2*self.l_sx)*self.n_sx**2:
                             self.si_dim*self.n_sx + (2*self.l_sx)*self.n_sx**2 + self.so_dim*self.n_sx],
                             [-1, self.n_sx, self.so_dim])

            n_weights = self.si_dim*self.n_sx + (2*self.l_sx)*self.n_sx**2 + self.so_dim*self.n_sx

            # distribute bias
            b_1 = tf.reshape(pnet_output[:, n_weights: n_weights + self.n_sx],
                             [-1, self.n_sx])
            b_hidden_list = []
            for i in range(self.l_sx):
                b1_tmp = tf.reshape(pnet_output[:,
                                    n_weights + self.n_sx + 2*i*self.n_sx:
                                    n_weights + self.n_sx + (2*i + 1)*self.n_sx],
                                    [-1, self.n_sx])
                b2_tmp = tf.reshape(pnet_output[:,
                                    n_weights + self.n_sx + (2*i + 1)*self.n_sx:
                                    n_weights + self.n_sx + (2*i + 2)*self.n_sx],
                                    [-1, self.n_sx])
                b_hidden_list.append([b1_tmp, b2_tmp])
            b_l = tf.reshape(pnet_output[:, n_weights + (2*self.l_sx + 1)*self.n_sx:], [-1, self.so_dim])

            # construct shape net
            omega_0 = self.cfg_shape_net['omega_0']
            u = tf.math.sin(omega_0*tf.einsum('ai,aij->aj', tf.cast(input_s,self.Dtype), tf.cast(w_1,self.Dtype)) +
                            tf.cast(b_1,self.Dtype))
            for i in range(self.l_sx):
                h = tf.math.sin(omega_0*tf.einsum('ai,aij->aj', u, w_hidden_list[i][0]) + b_hidden_list[i][0])
                u = 0.5*(u + tf.math.sin(omega_0*tf.einsum('ai,aij->aj', h, w_hidden_list[i][1]) + b_hidden_list[i][1]))
            u = tf.einsum('ai,aij->aj', u, w_l) + b_l

        else:
            # distribute weights
            w_1 = tf.reshape(pnet_output[:, :self.si_dim*self.n_sx],
                             [-1, self.si_dim, self.n_sx])
            w_hidden_list = []
            for i in range(self.l_sx):
                w_tmp = tf.reshape(pnet_output[:,
                                   self.si_dim*self.n_sx + i*self.n_sx**2:
                                   self.si_dim*self.n_sx + (i + 1)*self.n_sx**2],
                                   [-1, self.n_sx, self.n_sx])
                w_hidden_list.append(w_tmp)
            w_l = tf.reshape(pnet_output[:,
                             self.si_dim*self.n_sx + self.l_sx*self.n_sx**2:
                             self.si_dim*self.n_sx + self.l_sx*self.n_sx**2 + self.so_dim*self.n_sx],
                             [-1, self.n_sx, self.so_dim])
            n_weights = self.si_dim*self.n_sx + self.l_sx*self.n_sx**2 + self.so_dim*self.n_sx

            # distribute bias
            b_1 = tf.reshape(pnet_output[:, n_weights: n_weights + self.n_sx],
                             [-1, self.n_sx])
            b_hidden_list = []
            for i in range(self.l_sx):
                b_tmp = tf.reshape(pnet_output[:, n_weights + self.n_sx + i*self.n_sx:
                                                  n_weights + self.n_sx + (i + 1)*self.n_sx], [-1, self.n_sx])
                b_hidden_list.append(b_tmp)
            b_l = tf.reshape(pnet_output[:,
                             n_weights + (self.l_sx + 1)*self.n_sx:],
                             [-1, self.so_dim])

            # construct shape net
            omega_0 = self.cfg_shape_net['omega_0']
            u = tf.math.sin(omega_0*tf.einsum('ai,aij->aj', input_s, w_1) + b_1)
            for i in range(self.l_sx):
                u = tf.math.sin(omega_0*tf.einsum('ai,aij->aj', u, w_hidden_list[i]) + b_hidden_list[i])
            u = tf.einsum('ai,aij->aj', u, w_l) + b_l

        return tf.cast(u, tf.float32)
