"""The code represents a TensorFlow implementation of the Neural Implicit Flow

NIF is a neural network architecture designed for learning and
representing implicit functions. The model consists of two sub-networks :
a shape network and a parameter network. The shape network takes the input
data as input and outputs a point on a manifold, while the parameter network
takes the input data as input and outputs the parameters that determine the
shape of the manifold.

The NIF class is the main class that represents the NIF model. It takes two
configuration dictionaries as inputs, one for the shape network and one for
the parameter network, and initializes the shape and parameter networks.
The call method of the class takes the input data as input, passes it through
the parameter network to obtain the parameters, and then passes it through
the shape network to obtain the output.

The NIF class uses several custom layers defined in the layers module,
including SIREN (Sinusoidal Representation Networks), SIREN_ResNet
(SIREN with residual connections), MLP_ResNet (Multi-Layer Perceptron with
residual connections), MLP_SimpleShortCut (Multi-Layer Perceptron with
shortcut connections), JacRegLatentLayer (a layer that computes the Jacobian
of the output with respect to the latent variables and applies a regularization
term), and EinsumLayer (a layer that computes the dot product of two tensors
using Einstein summation notation). The NIF class also uses various TensorFlow
modules, including tf.keras, tf.keras.layers, tf.keras.regularizers, tf.initializers,
and tf.cast.
"""

__all__ = ["NIFMultiScale", "NIF", "NIFMultiScaleLastLayerParameterized"]

import json

import tensorflow as tf
from tensorflow.keras import Model, initializers
from tensorflow.keras import regularizers

from .layers import Dense
from .layers import EinsumLayer
from .layers import HyperLinearForSIREN
from .layers import JacRegLatentLayer
from .layers import MLP_ResNet
from .layers import MLP_SimpleShortCut
from .layers import SIREN
from .layers import SIREN_ResNet


class NIF(object):
    def __init__(self, cfg_shape_net, cfg_parameter_net, mixed_policy="float32"):
        super(NIF, self).__init__()
        self.cfg_shape_net = cfg_shape_net
        self.si_dim = cfg_shape_net["input_dim"]
        self.so_dim = cfg_shape_net["output_dim"]
        self.n_sx = cfg_shape_net["units"]
        self.l_sx = cfg_shape_net["nlayers"]
        self.pi_dim = cfg_parameter_net["input_dim"]
        self.pi_hidden = cfg_parameter_net["latent_dim"]
        self.n_st = cfg_parameter_net["units"]
        self.l_st = cfg_parameter_net["nlayers"]

        # additional regularization
        self.cfg_parameter_net = cfg_parameter_net
        self.p_jac_reg = cfg_parameter_net.get("jac_reg", None)
        self.p_l1_reg = cfg_parameter_net.get("l1_reg", None)
        self.p_l2_reg = cfg_parameter_net.get("l2_reg", None)
        self.p_act_l1_reg = cfg_parameter_net.get("act_l1_reg", None)
        self.p_act_l2_reg = cfg_parameter_net.get("act_l2_reg", None)

        self.mixed_policy = tf.keras.mixed_precision.Policy(
            mixed_policy
        )  # policy object can be feed into keras.layer
        self.variable_Dtype = self.mixed_policy.variable_dtype
        self.compute_Dtype = self.mixed_policy.compute_dtype

        # setup for standard regularization
        # 1. regularization for kernel in parameter net
        if isinstance(self.p_l2_reg, (float, int)):
            self.pnet_kernel_regularizer = regularizers.L2(self.p_l2_reg)
            self.pnet_bias_regularizer = regularizers.L2(self.p_l2_reg)
        elif isinstance(self.p_l1_reg, (float, int)):
            self.pnet_kernel_regularizer = regularizers.L1(self.p_l1_reg)
            self.pnet_bias_regularizer = regularizers.L1(self.p_l1_reg)
        else:
            self.pnet_kernel_regularizer = None
            self.pnet_bias_regularizer = None

        # 2. output of parameter net regularization
        if isinstance(self.p_act_l2_reg, (float, int)):
            self.pnet_act_regularizer = regularizers.L2(self.p_act_l2_reg)
        elif isinstance(self.p_act_l1_reg, (float, int)):
            self.pnet_act_regularizer = regularizers.L1(self.p_act_l1_reg)
        else:
            self.pnet_act_regularizer = None

        # finally initialize the parameter net structure
        self.pnet_list = self._initialize_pnet(cfg_parameter_net, cfg_shape_net)

    def call(self, inputs, training=None, mask=None):
        input_p = inputs[:, 0 : self.pi_dim]
        input_s = inputs[:, self.pi_dim : self.pi_dim + self.si_dim]
        self.pnet_output = self._call_parameter_net(input_p, self.pnet_list)[0]
        return self._call_shape_net(
            tf.cast(input_s, self.compute_Dtype),
            self.pnet_output,
            si_dim=self.si_dim,
            so_dim=self.so_dim,
            n_sx=self.n_sx,
            l_sx=self.l_sx,
            activation=self.cfg_shape_net["activation"],
            variable_dtype=self.variable_Dtype,
        )

    def _initialize_pnet(self, cfg_parameter_net, cfg_shape_net):
        # just simple implementation of a shortcut connected parameter net with a similar shapenet
        self.po_dim = (
            (self.l_sx) * self.n_sx**2
            + (self.si_dim + self.so_dim + 1 + self.l_sx) * self.n_sx
            + self.so_dim
        )

        # construct parameter_net
        pnet_layers_list = []
        # 1. first layer
        layer_1 = Dense(
            self.n_st,
            cfg_parameter_net["activation"],
            kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
            bias_initializer=initializers.TruncatedNormal(stddev=0.1),
            dtype=self.mixed_policy,
            kernel_regularizer=self.pnet_kernel_regularizer,
            bias_regularizer=self.pnet_bias_regularizer,
            name="first_dense_pnet",
        )
        pnet_layers_list.append(layer_1)

        # 2. hidden layer
        for i in range(self.l_st):
            tmp_layer = MLP_SimpleShortCut(
                self.n_st,
                cfg_parameter_net["activation"],
                kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                bias_initializer=initializers.TruncatedNormal(stddev=0.1),
                kernel_regularizer=self.pnet_kernel_regularizer,
                bias_regularizer=self.pnet_bias_regularizer,
                mixed_policy=self.mixed_policy,
                name="hidden_mlpshortcut_pnet_{}".format(i),
            )
            # identity_layer = Lambda(lambda x: x)
            # tmp_layer =tf.keras.layers.Add()(identity_layer,tmp_layer)
            pnet_layers_list.append(tmp_layer)

        # 3. bottleneck layer
        bottleneck_layer = Dense(
            self.pi_hidden,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
            bias_initializer=initializers.TruncatedNormal(stddev=0.1),
            dtype=self.mixed_policy,
            kernel_regularizer=self.pnet_kernel_regularizer,
            bias_regularizer=self.pnet_bias_regularizer,
            name="bottleneck_pnet",
        )
        pnet_layers_list.append(bottleneck_layer)

        # 4. last layer
        # add regularization
        last_layer = Dense(
            self.po_dim,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
            bias_initializer=initializers.TruncatedNormal(stddev=0.1),
            kernel_regularizer=self.pnet_kernel_regularizer,
            bias_regularizer=self.pnet_bias_regularizer,
            activity_regularizer=self.pnet_act_regularizer,
            dtype=self.mixed_policy,
            name="last_pnet",
        )
        pnet_layers_list.append(last_layer)
        return pnet_layers_list

    @staticmethod
    def _call_shape_net(
        input_s, pnet_output, si_dim, so_dim, n_sx, l_sx, activation, variable_dtype
    ):
        w_1 = tf.reshape(
            pnet_output[:, : si_dim * n_sx], [-1, si_dim, n_sx], name="w_first_snet"
        )
        w_hidden_list = []
        for i in range(l_sx):
            w_tmp = tf.reshape(
                pnet_output[
                    :,
                    si_dim * n_sx + i * n_sx**2 : si_dim * n_sx + (i + 1) * n_sx**2,
                ],
                [-1, n_sx, n_sx],
                name="w_hidden_snet_{}".format(i),
            )
            w_hidden_list.append(w_tmp)
        w_l = tf.reshape(
            pnet_output[
                :,
                si_dim * n_sx
                + l_sx * n_sx**2 : si_dim * n_sx
                + l_sx * n_sx**2
                + so_dim * n_sx,
            ],
            [-1, n_sx, so_dim],
            name="w_last_snet",
        )
        n_weights = si_dim * n_sx + l_sx * n_sx**2 + so_dim * n_sx

        # distribute bias
        b_1 = tf.reshape(
            pnet_output[:, n_weights : n_weights + n_sx],
            [-1, n_sx],
            name="b_first_snet",
        )
        b_hidden_list = []
        for i in range(l_sx):
            b_tmp = tf.reshape(
                pnet_output[
                    :, n_weights + n_sx + i * n_sx : n_weights + n_sx + (i + 1) * n_sx
                ],
                [-1, n_sx],
                name="b_hidden_snet_{}".format(i),
            )
            b_hidden_list.append(b_tmp)
        b_l = tf.reshape(
            pnet_output[:, n_weights + (l_sx + 1) * n_sx :],
            [-1, so_dim],
            name="b_last_snet",
        )

        # construct shape net
        act_fun = tf.keras.activations.get(activation)
        u = act_fun(
            EinsumLayer("ai,aij->aj", name="first_einsum_snet")((input_s, w_1)) + b_1
        )
        # u = act_fun(tf.einsum('ai,aij->aj', input_s, w_1) + b_1)

        for i in range(l_sx):
            w_tmp = w_hidden_list[i]
            b_tmp = b_hidden_list[i]
            u = (
                act_fun(
                    EinsumLayer("ai,aij->aj", name="hidden_einsum_snet_{}".format(i))(
                        (u, w_tmp)
                    )
                    + b_tmp
                )
                + u
            )
            # u = act_fun(tf.einsum('ai,aij->aj', u, w_tmp) + b_tmp) + u
        u = EinsumLayer("ai,aij->aj", name="last_einsum_snet")((u, w_l)) + b_l
        # u = tf.einsum('ai,aij->aj', u, w_l) + b_l
        return tf.cast(u, variable_dtype, name="output_cast_snet")

    @staticmethod
    def _call_parameter_net(input_p, pnet_list):
        latent = input_p
        for layer_ in pnet_list[:-1]:
            latent = layer_(latent)
        output_final = pnet_list[-1](latent)
        return output_final, latent

    def build(self):
        if isinstance(self.p_jac_reg, (float, int)):
            input_tot = tf.keras.layers.Input(
                shape=(self.pi_dim + self.si_dim), name="input_tot"
            )
            input_p = input_tot[:, : self.pi_dim]
            model_augment_latent = Model(
                inputs=[input_tot],
                outputs=[
                    self.call(input_tot),
                    self._call_parameter_net(input_p, self.pnet_list)[1],
                ],
            )
            # we take d latent / d parameter
            y_index = range(0, self.pi_hidden)
            x_index = range(0, self.pi_dim)
            output = JacRegLatentLayer(
                model_augment_latent,
                y_index,
                x_index,
                self.p_jac_reg,
                name="jac_reg_latent",
            )(input_tot)
            return Model(inputs=[input_tot], outputs=[output])
        else:
            return self.model()

    def model(self):
        input_tot = tf.keras.layers.Input(
            shape=(self.pi_dim + self.si_dim), name="input_tot"
        )
        return Model(inputs=[input_tot], outputs=[self.call(input_tot)])

    def model_p_to_w(self):
        input_p = tf.keras.layers.Input(shape=(self.pi_dim), name="input_p_to_w")
        return Model(
            inputs=[input_p],
            outputs=[self._call_parameter_net(input_p, self.pnet_list)[0]],
        )

    def model_p_to_lr(self):
        input_p = tf.keras.layers.Input(shape=(self.pi_dim), name="input_p_to_lr")
        # this model: t, mu -> hidden LR
        return Model(
            inputs=[input_p],
            outputs=[self._call_parameter_net(input_p, self.pnet_list)[1]],
        )

    def model_lr_to_w(self):
        input_lr = tf.keras.layers.Input(shape=(self.pi_hidden), name="input_lr_to_w")
        # this model: hidden LR -> weights and biases of shapenet
        return Model(inputs=[input_lr], outputs=[self.pnet_list[-1](input_lr)])

    def model_x_to_u_given_w(self):
        input_s = tf.keras.layers.Input(
            shape=(self.si_dim), name="input_x_to_u_given_w"
        )
        input_pnet = tf.keras.layers.Input(
            shape=(self.pnet_list[-1].output_shape[1]), name="input_w_and_b_from_pnet"
        )
        return Model(
            inputs=[input_s, input_pnet],
            outputs=[
                self._call_shape_net(
                    tf.cast(input_s, self.compute_Dtype),
                    tf.cast(input_pnet, self.compute_Dtype),
                    si_dim=self.si_dim,
                    so_dim=self.so_dim,
                    n_sx=self.n_sx,
                    l_sx=self.l_sx,
                    activation=self.cfg_shape_net["activation"],
                    variable_dtype=self.variable_Dtype,
                )
            ],
        )

    def save_config(self, filename="config.json"):
        config = {
            "cfg_shape_net": self.cfg_shape_net,
            "cfg_parameter_net": self.cfg_parameter_net,
            "mixed_policy": self.mixed_policy.name,
        }
        with open(filename, "w") as write_file:
            json.dump(config, write_file, indent=4)


class NIFMultiScale(NIF):
    def __init__(self, cfg_shape_net, cfg_parameter_net, mixed_policy="float32"):
        super(NIFMultiScale, self).__init__(
            cfg_shape_net, cfg_parameter_net, mixed_policy
        )

    def call(self, inputs, training=None, mask=None):
        input_p = inputs[:, 0 : self.pi_dim]
        input_s = inputs[:, self.pi_dim : self.pi_dim + self.si_dim]
        # get parameter from parameter_net
        self.pnet_output = self._call_parameter_net(input_p, self.pnet_list)[0]
        return self._call_shape_net_mres(
            tf.cast(input_s, self.compute_Dtype),
            self.pnet_output,
            flag_resblock=self.cfg_shape_net["use_resblock"],
            omega_0=tf.cast(self.cfg_shape_net["omega_0"], self.compute_Dtype),
            si_dim=self.si_dim,
            so_dim=self.so_dim,
            n_sx=self.n_sx,
            l_sx=self.l_sx,
            variable_dtype=self.variable_Dtype,
        )

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
        assert ("use_resblock" in cfg_shape_net.keys()), \
            "`use_resblock` should be in cfg_shape_net"
        # assert 'nn_type' in cfg_parameter_net.keys(), "`nn_type` should
        # be in cfg_parameter_net"
        assert (type(cfg_shape_net["use_resblock"]) == bool), \
            "cfg_shape_net['use_resblock'] must be a bool"

        pnet_layers_list = []
        if cfg_shape_net["connectivity"] == "full":
            # very first, determine the output dimension of parameter_net
            if cfg_shape_net["use_resblock"]:
                self.po_dim = (
                    (2 * self.l_sx) * self.n_sx**2
                    + (self.si_dim + self.so_dim + 1 + 2 * self.l_sx) * self.n_sx
                    + self.so_dim
                )
            else:
                self.po_dim = (
                    (self.l_sx) * self.n_sx**2
                    + (self.si_dim + self.so_dim + 1 + self.l_sx) * self.n_sx
                    + self.so_dim
                )
        elif cfg_shape_net["connectivity"] == "last_layer":
            # only parameterize the last layer
            self.po_dim = self.pi_hidden
        else:
            raise ValueError("cfg_shape_net missing correct `connectivity`")

        # first, and hidden layers are only dependent on the type of parameter_net
        # if cfg_parameter_net['nn_type'] == 'siren':
        if cfg_parameter_net["activation"] == "sine":
            # assert cfg_parameter_net['activation'] == 'sine',
            # "you should specify activation in cfg_parameter_net as "sine"
            # 1. first layer
            layer_1 = SIREN(
                self.pi_dim,
                self.n_st,
                "first",
                cfg_parameter_net["omega_0"],
                cfg_shape_net,
                self.pnet_kernel_regularizer,
                self.pnet_bias_regularizer,
                self.mixed_policy,
                name="siren_first_pnet",
            )
            pnet_layers_list.append(layer_1)

            # 2. hidden layers
            if cfg_parameter_net["use_resblock"]:
                for i in range(self.l_st):
                    tmp_layer = SIREN_ResNet(
                        self.n_st,
                        self.n_st,
                        cfg_parameter_net["omega_0"],
                        self.pnet_kernel_regularizer,
                        self.pnet_bias_regularizer,
                        self.mixed_policy,
                        name="siren_hidden_resblock_pnet_{}".format(i),
                    )
                    pnet_layers_list.append(tmp_layer)
            else:
                for i in range(self.l_st):
                    tmp_layer = SIREN(
                        self.n_st,
                        self.n_st,
                        "hidden",
                        cfg_parameter_net["omega_0"],
                        cfg_shape_net,
                        self.pnet_kernel_regularizer,
                        self.pnet_bias_regularizer,
                        self.mixed_policy,
                        name="siren_hidden_pnet_{}".format(i),
                    )
                    pnet_layers_list.append(tmp_layer)

            # 3. bottleneck layer
            bottleneck_layer = SIREN(
                self.n_st,
                self.pi_hidden,
                "bottleneck",
                cfg_parameter_net["omega_0"],
                cfg_shape_net,
                self.pnet_kernel_regularizer,
                self.pnet_bias_regularizer,
                self.mixed_policy,
                name="siren_bottleneck_pnet",
            )
            pnet_layers_list.append(bottleneck_layer)

            # 4. last layer
            last_layer = HyperLinearForSIREN(
                self.pi_hidden,
                self.po_dim,
                cfg_shape_net,
                self.mixed_policy,
                connectivity=cfg_shape_net["connectivity"],
                kernel_regularizer=self.pnet_kernel_regularizer,
                bias_regularizer=self.pnet_bias_regularizer,
                activity_regularizer=self.pnet_act_regularizer,
                name="HyperLinearForSIREN",
            )

            pnet_layers_list.append(last_layer)

        else:
            # cfg_parameter_net['nn_type'] == 'mlp':
            # 1. first layer
            layer_1 = Dense(
                self.n_st,
                cfg_parameter_net["activation"],
                kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                bias_initializer=initializers.TruncatedNormal(stddev=0.1),
                kernel_regularizer=self.pnet_kernel_regularizer,
                bias_regularizer=self.pnet_bias_regularizer,
                dtype=self.mixed_policy,
                name="mlp_first_pnet",
            )
            pnet_layers_list.append(layer_1)

            # 2. hidden layer
            if cfg_parameter_net["use_resblock"]:
                for i in range(self.l_st):
                    tmp_layer = MLP_ResNet(
                        width=self.n_st,
                        activation=cfg_parameter_net["activation"],
                        kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                        bias_initializer=initializers.TruncatedNormal(stddev=0.1),
                        kernel_regularizer=self.pnet_kernel_regularizer,
                        bias_regularizer=self.pnet_bias_regularizer,
                        mixed_policy=self.mixed_policy,
                        name="mlp_hidden_resblock_pnet_{}".format(i),
                    )
                    pnet_layers_list.append(tmp_layer)
            else:
                for i in range(self.l_st):
                    tmp_layer = MLP_SimpleShortCut(
                        self.n_st,
                        cfg_parameter_net["activation"],
                        kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                        bias_initializer=initializers.TruncatedNormal(stddev=0.1),
                        kernel_regularizer=self.pnet_kernel_regularizer,
                        bias_regularizer=self.pnet_bias_regularizer,
                        mixed_policy=self.mixed_policy,
                        name="mlp_hidden_pnet_{}".format(i),
                    )
                    # identity_layer = Lambda(lambda x: x)
                    # tmp_layer =tf.keras.layers.Add()(identity_layer,tmp_layer)
                    pnet_layers_list.append(tmp_layer)

            # 3. bottleneck layer
            bottleneck_layer = Dense(
                self.pi_hidden,
                kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                bias_initializer=initializers.TruncatedNormal(stddev=0.1),
                kernel_regularizer=self.pnet_kernel_regularizer,
                bias_regularizer=self.pnet_bias_regularizer,
                dtype=self.mixed_policy,
                name="bottleneck_pnet",
            )
            pnet_layers_list.append(bottleneck_layer)

            # 4. last layer
            last_layer = HyperLinearForSIREN(
                self.pi_hidden,
                self.po_dim,
                cfg_shape_net,
                self.mixed_policy,
                connectivity=cfg_shape_net["connectivity"],
                kernel_regularizer=self.pnet_kernel_regularizer,
                bias_regularizer=self.pnet_bias_regularizer,
                activity_regularizer=self.pnet_act_regularizer,
                name="HyperLinearForSIREN",
            )
            pnet_layers_list.append(last_layer)

        return pnet_layers_list

    @staticmethod
    def _call_shape_net_mres(
        input_s,
        pnet_output,
        flag_resblock,
        omega_0,
        si_dim,
        so_dim,
        n_sx,
        l_sx,
        variable_dtype,
    ):
        """
        distribute `pnet_output` into weight and bias, it depends on the type of shapenet.

        For now, we only support shapenet having the following structure,
            - resnet-block
            - plain fnn
        """
        if flag_resblock:
            # distribute weights
            w_1 = tf.reshape(
                pnet_output[:, : si_dim * n_sx], [-1, si_dim, n_sx], name="w_first_snet"
            )
            w_hidden_list = []
            for i in range(l_sx):
                w1_tmp = tf.reshape(
                    pnet_output[
                        :,
                        si_dim * n_sx
                        + 2 * i * n_sx**2 : si_dim * n_sx
                        + (2 * i + 1) * n_sx**2,
                    ],
                    [-1, n_sx, n_sx],
                    name="w1_hidden_snet_{}".format(i),
                )
                w2_tmp = tf.reshape(
                    pnet_output[
                        :,
                        si_dim * n_sx
                        + (2 * i + 1) * n_sx**2 : si_dim * n_sx
                        + (2 * i + 2) * n_sx**2,
                    ],
                    [-1, n_sx, n_sx],
                    name="w2_hidden_snet_{}".format(i),
                )
                w_hidden_list.append([w1_tmp, w2_tmp])
            w_l = tf.reshape(
                pnet_output[
                    :,
                    si_dim * n_sx
                    + (2 * l_sx) * n_sx**2 : si_dim * n_sx
                    + (2 * l_sx) * n_sx**2
                    + so_dim * n_sx,
                ],
                [-1, n_sx, so_dim],
                name="w_last_snet",
            )

            n_weights = si_dim * n_sx + (2 * l_sx) * n_sx**2 + so_dim * n_sx

            # distribute bias
            b_1 = tf.reshape(
                pnet_output[:, n_weights : n_weights + n_sx],
                [-1, n_sx],
                name="b_first_snet",
            )
            b_hidden_list = []
            for i in range(l_sx):
                b1_tmp = tf.reshape(
                    pnet_output[
                        :,
                        n_weights
                        + n_sx
                        + 2 * i * n_sx : n_weights
                        + n_sx
                        + (2 * i + 1) * n_sx,
                    ],
                    [-1, n_sx],
                    name="b1_hidden_snet_{}".format(i),
                )
                b2_tmp = tf.reshape(
                    pnet_output[
                        :,
                        n_weights
                        + n_sx
                        + (2 * i + 1) * n_sx : n_weights
                        + n_sx
                        + (2 * i + 2) * n_sx,
                    ],
                    [-1, n_sx],
                    name="b1_hidden_snet_{}".format(i),
                )
                b_hidden_list.append([b1_tmp, b2_tmp])
            b_l = tf.reshape(
                pnet_output[:, n_weights + (2 * l_sx + 1) * n_sx :],
                [-1, so_dim],
                name="b_last_snet",
            )

            # construct shape net
            u = tf.math.sin(
                omega_0
                * EinsumLayer("ai,aij->aj", name="first_einsum_snet")((input_s, w_1))
                + b_1
            )
            # u = tf.math.sin(omega_0 * tf.einsum('ai,aij->aj', input_s, w_1) + b_1)
            for i in range(l_sx):
                h = tf.math.sin(
                    omega_0
                    * EinsumLayer(
                        "ai,aij->aj", name="hidden_1_einsum_snet_{}".format(i)
                    )((u, w_hidden_list[i][0]))
                    + b_hidden_list[i][0]
                )
                # h = tf.math.sin(omega_0 * tf.einsum('ai,aij->aj', u, w_hidden_list[i][0])
                # + b_hidden_list[i][0])
                u = 0.5 * (
                    u
                    + tf.math.sin(
                        omega_0
                        * EinsumLayer(
                            "ai,aij->aj", name="hidden_2_einsum_snet_{}".format(i)
                        )((h, w_hidden_list[i][1]))
                        + b_hidden_list[i][1]
                    )
                )
                # u = 0.5 * (u + tf.math.sin(omega_0 * tf.einsum('ai,aij->aj', h, w_hidden_list[i][1])
                # + b_hidden_list[i][1]))
            u = EinsumLayer("ai,aij->aj", name="last_einsum_snet")((u, w_l)) + b_l
            # u = tf.einsum('ai,aij->aj', u, w_l) + b_l

        else:
            # disable resblock for parameter net
            # distribute weights
            w_1 = tf.reshape(
                pnet_output[:, : si_dim * n_sx], [-1, si_dim, n_sx], name="w_first_snet"
            )
            w_hidden_list = []
            for i in range(l_sx):
                w_tmp = tf.reshape(
                    pnet_output[
                        :,
                        si_dim * n_sx
                        + i * n_sx**2 : si_dim * n_sx
                        + (i + 1) * n_sx**2,
                    ],
                    [-1, n_sx, n_sx],
                    name="w_hidden_snet_{}".format(i),
                )
                w_hidden_list.append(w_tmp)
            w_l = tf.reshape(
                pnet_output[
                    :,
                    si_dim * n_sx
                    + l_sx * n_sx**2 : si_dim * n_sx
                    + l_sx * n_sx**2
                    + so_dim * n_sx,
                ],
                [-1, n_sx, so_dim],
                name="w_last_snet",
            )
            n_weights = si_dim * n_sx + l_sx * n_sx**2 + so_dim * n_sx

            # distribute bias
            b_1 = tf.reshape(
                pnet_output[:, n_weights : n_weights + n_sx],
                [-1, n_sx],
                name="b_first_snet",
            )
            b_hidden_list = []
            for i in range(l_sx):
                b_tmp = tf.reshape(
                    pnet_output[
                        :,
                        n_weights + n_sx + i * n_sx : n_weights + n_sx + (i + 1) * n_sx,
                    ],
                    [-1, n_sx],
                    name="b_hidden_snet_{}".format(i),
                )
                b_hidden_list.append(b_tmp)
            b_l = tf.reshape(
                pnet_output[:, n_weights + (l_sx + 1) * n_sx :],
                [-1, so_dim],
                name="b_last_snet",
            )

            # construct shape net
            u = tf.math.sin(
                omega_0
                * EinsumLayer("ai,aij->aj", name="first_einsum_snet")((input_s, w_1))
                + b_1
            )
            # u = tf.math.sin(omega_0 * tf.einsum('ai,aij->aj', input_s, w_1) + b_1)
            for i in range(l_sx):
                u = tf.math.sin(
                    omega_0
                    * EinsumLayer("ai,aij->aj", name="hidden_einsum_snet_{}".format(i))(
                        (u, w_hidden_list[i])
                    )
                    + b_hidden_list[i]
                )
                # u = tf.math.sin(omega_0 * tf.einsum('ai,aij->aj', u, w_hidden_list[i]) + b_hidden_list[i])
            u = EinsumLayer("ai,aij->aj", name="last_einsum_snet")((u, w_l)) + b_l
            # u = tf.einsum('ai,aij->aj', u, w_l) + b_l

        return tf.cast(u, variable_dtype, name="output_cast_snet")

    def model_x_to_u_given_w(self):
        input_s = tf.keras.layers.Input(
            shape=(self.si_dim), name="input_x_to_u_given_w"
        )
        input_pnet = tf.keras.layers.Input(
            shape=(self.pnet_list[-1].output_shape[1]), name="input_w_and_b_from_pnet"
        )
        return Model(
            inputs=[input_s, input_pnet],
            outputs=[
                self._call_shape_net_mres(
                    tf.cast(input_s, self.compute_Dtype),
                    tf.cast(input_pnet, self.compute_Dtype),
                    flag_resblock=self.cfg_shape_net["use_resblock"],
                    omega_0=tf.cast(self.cfg_shape_net["omega_0"], self.compute_Dtype),
                    si_dim=self.si_dim,
                    so_dim=self.so_dim,
                    n_sx=self.n_sx,
                    l_sx=self.l_sx,
                    variable_dtype=self.variable_Dtype,
                )
            ],
        )


class NIFMultiScaleLastLayerParameterized(NIFMultiScale):
    def __init__(self, cfg_shape_net, cfg_parameter_net, mixed_policy="float32"):
        super(NIFMultiScaleLastLayerParameterized, self).__init__(
            cfg_shape_net, cfg_parameter_net, mixed_policy
        )
        assert (
            cfg_shape_net["connectivity"] == "last_layer"
        ), "you should assign cfg_shape_net['connectivity'] == 'last_layer'"

        self.s_l1_reg = cfg_shape_net.get("l1_reg", None)
        self.s_l2_reg = cfg_shape_net.get("l2_reg", None)

        if isinstance(self.s_l2_reg, (float, int)):
            self.snet_kernel_regularizer = regularizers.L2(self.p_l2_reg)
            self.snet_bias_regularizer = regularizers.L2(self.p_l2_reg)
        elif isinstance(self.s_l1_reg, (float, int)):
            self.snet_kernel_regularizer = regularizers.L1(self.p_l1_reg)
            self.snet_bias_regularizer = regularizers.L1(self.p_l1_reg)
        else:
            self.snet_kernel_regularizer = None
            self.snet_bias_regularizer = None

        self.snet_list, self.last_layer_bias = self._initialize_snet(cfg_shape_net)

        # last layer einsum dense
        self.einsum_layer = tf.keras.layers.EinsumDense("ijk,ik->ij", bias_axes="j")

    def call(self, inputs, training=None, mask=None):
        input_p = inputs[:, 0 : self.pi_dim]
        input_s = inputs[:, self.pi_dim : self.pi_dim + self.si_dim]
        # get parameter from parameter_net
        self.pnet_output = self._call_parameter_net(input_p, self.pnet_list)[0]
        return self._call_shape_net_mres_only_para_last_layer(
            tf.cast(input_s, self.compute_Dtype),
            self.snet_list,
            tf.cast(self.last_layer_bias, self.compute_Dtype),
            self.pnet_output,
            self.so_dim,
            self.pi_hidden,
            self.variable_Dtype,
        )

    def model_p_to_lr(self):
        input_p = tf.keras.layers.Input(shape=(self.pi_dim), name="input_p_to_lr")
        # this model: t, mu -> hidden LR
        return Model(
            inputs=[input_p],
            outputs=[self._call_parameter_net(input_p, self.pnet_list)[0]],
        )

    def model_x_to_phi(self):
        input_s = tf.keras.layers.Input(shape=(self.si_dim), name="input_x_to_phi")
        return Model(
            inputs=[input_s],
            outputs=[
                tf.cast(
                    self._call_shape_net_get_phi_x(
                        input_s, self.snet_list, self.so_dim, self.pi_hidden
                    ),
                    self.variable_Dtype,
                )
            ],
        )

    def model_lr_to_w(self):
        raise ValueError(
            "In this class: NIFMultiScaleLastLayerParameterization, `w` is the same as `lr`"
        )

    def model_x_to_u_given_w(self):
        input_s = tf.keras.layers.Input(
            shape=(self.si_dim), name="input_x_to_u_given_w"
        )
        input_pnet = tf.keras.layers.Input(
            shape=(self.pnet_list[-1].output_shape[1]), name="input_w_and_b_from_pnet"
        )
        return Model(
            inputs=[input_s, input_pnet],
            outputs=[
                self._call_shape_net_mres_only_para_last_layer(
                    tf.cast(input_s, self.compute_Dtype),
                    self.snet_list,
                    tf.cast(self.last_layer_bias, self.compute_Dtype),
                    tf.cast(input_pnet, self.compute_Dtype),
                    self.so_dim,
                    self.pi_hidden,
                    self.variable_Dtype,
                )
            ],
        )

    def _initialize_snet(self, cfg_shape_net):
        # create a simple feedfowrard, with resblock or not, that maps self.si_dim to
        # self.so_dim*self.pi_hidden

        snet_layers_list = []
        # 1. first layer
        layer_1 = SIREN(
            self.si_dim,
            self.n_sx,
            "first",
            cfg_shape_net["omega_0"],
            cfg_shape_net,
            self.snet_kernel_regularizer,
            self.snet_bias_regularizer,
            self.mixed_policy,
            name="siren_first_snet",
        )
        snet_layers_list.append(layer_1)

        # 2. hidden layers
        if cfg_shape_net["use_resblock"]:
            for i in range(self.l_sx):
                tmp_layer = SIREN_ResNet(
                    self.n_sx,
                    self.n_sx,
                    cfg_shape_net["omega_0"],
                    self.snet_kernel_regularizer,
                    self.snet_bias_regularizer,
                    self.mixed_policy,
                    name="siren_hidden_resblock_snet_{}".format(i),
                )
                snet_layers_list.append(tmp_layer)
        else:
            for i in range(self.l_sx):
                tmp_layer = SIREN(
                    self.n_sx,
                    self.n_sx,
                    "hidden",
                    cfg_shape_net["omega_0"],
                    cfg_shape_net,
                    self.snet_kernel_regularizer,
                    self.snet_bias_regularizer,
                    self.mixed_policy,
                    name="siren_hidden_snet_{}".format(i),
                )
                snet_layers_list.append(tmp_layer)

        # 3. bottleneck AND the same time, last layer for spatial basis
        bottle_last_layer = SIREN(
            self.n_sx,
            self.po_dim * self.so_dim, # todo: change this to self.sl_dim * self.so_dim
            "bottleneck",
            cfg_shape_net["omega_0"],
            cfg_shape_net,
            self.snet_kernel_regularizer,
            self.snet_bias_regularizer,
            self.mixed_policy,
            name="siren_bottleneck_snet",
        )
        snet_layers_list.append(bottle_last_layer)

        # create bias for the last layer
        # last_layer_init = initializers.TruncatedNormal(stddev=1e-16)
        last_layer_init = initializers.TruncatedNormal(stddev=0.1)
        last_layer_bias = tf.Variable(
            last_layer_init([self.so_dim]),
            dtype=self.mixed_policy.variable_dtype,
            name="last_layer_bias_snet",
        )

        return snet_layers_list, last_layer_bias

    def _call_shape_net_get_phi_x(self, input_s, snet_layers_list, so_dim, pi_hidden):
        # 1. x -> phi_x
        phi_x = input_s
        for layer_ in snet_layers_list:
            phi_x = layer_(phi_x)
        # 2. phi_x * a_t + bias
        phi_x_matrix = tf.reshape(phi_x, [-1, so_dim, pi_hidden], name="phi_snet")
        return phi_x_matrix

    def _call_shape_net_mres_only_para_last_layer(
        self,
        input_s,
        snet_layers_list,
        last_layer_bias,
        pnet_output,
        so_dim,
        pi_hidden,
        variable_dtype,
    ):
        phi_x_matrix = self._call_shape_net_get_phi_x(input_s,
                                                      snet_layers_list,
                                                      so_dim,
                                                      pi_hidden)
        # u = tf.keras.layers.Add()[
        #     tf.keras.layers.Dot(axes=(2, 1))([phi_x_matrix, pnet_output]),
        #     last_layer_bias
        # ]

        # u = tf.keras.layers.Add()[
        #     tf.keras.layers.Dot(axes=(2, 1))([phi_x_matrix, pnet_output]),
        #     last_layer_bias
        # ]



        # Perform matrix multiplication using EinsumDense layer
        u = self.einsum_layer([phi_x_matrix, pnet_output])

        # layer = tf.keras.layers.EinsumDense("ab,bc->ac",
        #                                     output_shape=64,
        #                                     bias_axes="c")

        return tf.cast(u, variable_dtype, name="output_cast")
