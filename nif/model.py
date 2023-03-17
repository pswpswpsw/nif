"""The code represents a TensorFlow implementation of the Neural Implicit Flow

NIF is a neural network architecture designed for learning and
representing implicit flow. The model consists of two sub-networks :
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
from .layers import BiasAddLayer


class NIF(object):
    """
    Neural Implicit Flow class represents a network with two sub-networks to reduce
    the dimensionality of spatial temporal fields

    Attributes:
        cfg_shape_net (dict): Configuration dictionary for the shape network.
        cfg_parameter_net (dict): Configuration dictionary for the parameter network.
        mixed_policy (str): The data type for mixed precision training (default is 'float32').

    Methods:
        call(self, inputs, training=None, mask=None): Forward pass for the NIF model.
        build(self): Builds and returns the NIF model with a Jacobian regularization layer.
        model(self): Builds and returns the NIF model.
        model_p_to_w(self): Builds and returns a model that maps input parameters to weights and
            biases of the shape net.
        model_p_to_lr(self): Builds and returns a model that maps input parameters to the hidden
            layer representation.
        model_lr_to_w(self): Builds and returns a model that maps the hidden layer representation
            to shape net weights and biases.
        model_x_to_u_given_w(self): Builds and returns a model that maps input states to output,
            given shape net weights and biases.
        save_config(self, filename="config.json"): Saves the NIF configuration to a JSON file.
    """

    def __init__(self, cfg_shape_net, cfg_parameter_net, mixed_policy="float32"):
        """
        Initializes the NIF object with the given configurations and mixed precision policy.

        Args:
            cfg_shape_net (dict): Configuration dictionary for the shape network.
            cfg_parameter_net (dict): Configuration dictionary for the parameter network.
            mixed_policy (str, optional): The data type for mixed precision training. Defaults to "float32".
        """
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
        """
        Performs the forward pass for the NIF model, given input parameters and states.

        Args:
            inputs (tf.Tensor): A tensor containing input parameters and states.
            training (bool, optional): Whether the model is in training mode. Defaults to None.
            mask (tf.Tensor, optional): A tensor representing masked elements. Defaults to None.

        Returns:
            tf.Tensor: The output tensor after passing through the shape network.
        """
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
        """
        Initializes the parameter network structure based on the given configuration.

        Args:
            cfg_parameter_net (dict): Configuration dictionary for the parameter network.
            cfg_shape_net (dict): Configuration dictionary for the shape network.

        Returns:
            list: A list of layers that make up the parameter network.
        """
        # just simple implementation of a shortcut connected parameter net with
        # a similar shapenet
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
        """
        Calls the shape network with the given input and parameter network output.

        Args:
            input_s (tf.Tensor): Input tensor for the shape network.
            pnet_output (tf.Tensor): Output tensor of the parameter network.
            si_dim (int): Input dimension of the shape network.
            so_dim (int): Output dimension of the shape network.
            n_sx (int): Number of units in each hidden layer of the shape network.
            l_sx (int): Number of hidden layers in the shape network.
            activation (str): Activation function used in the shape network.
            variable_dtype (str): Data type for the variables in the shape network.

        Returns:
            tf.Tensor: The output tensor of the shape network.
        """
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
        """
        Calls the parameter network with the given input and list of layers.

        Args:
            input_p (tf.Tensor): Input tensor for the parameter network.
            pnet_list (list): List of layers in the parameter network.

        Returns:
            tuple: A tuple containing the output tensor of the parameter network
                   and the hidden layer representation (latent).
        """
        latent = input_p
        for layer_ in pnet_list[:-1]:
            latent = layer_(latent)
        output_final = pnet_list[-1](latent)
        return output_final, latent

    def build(self):
        """
        Builds and returns the NIF model with a Jacobian regularization layer
        if specified in the configuration. Otherwise it is the same as `.model()`

        Returns:
            tf.keras.Model: The NIF model with or without the Jacobian regularization layer.
        """
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
        """
        Builds and returns the NIF model.

        Returns:
            tf.keras.Model: The NIF model.
        """
        input_tot = tf.keras.layers.Input(
            shape=(self.pi_dim + self.si_dim), name="input_tot"
        )
        return Model(inputs=[input_tot], outputs=[self.call(input_tot)])

    def model_p_to_w(self):
        """
        Builds and returns a model that maps input parameters to weights and
        biases of the shape network.

        Returns:
            tf.keras.Model: The model mapping input parameters to shape network
            weights and biases.
        """
        input_p = tf.keras.layers.Input(shape=(self.pi_dim), name="input_p_to_w")
        return Model(
            inputs=[input_p],
            outputs=[self._call_parameter_net(input_p, self.pnet_list)[0]],
        )

    def model_p_to_lr(self):
        """
        Builds and returns a model that maps input parameters to the hidden layer
        representation.

        Returns:
            tf.keras.Model: The model mapping input parameters to the hidden layer
            representation.
        """
        input_p = tf.keras.layers.Input(shape=(self.pi_dim), name="input_p_to_lr")
        # this model: t, mu -> hidden LR
        return Model(
            inputs=[input_p],
            outputs=[self._call_parameter_net(input_p, self.pnet_list)[1]],
        )

    def model_lr_to_w(self):
        """
        Builds and returns a model that maps the hidden layer representation to
        shape network weights and biases.

        Returns:
            tf.keras.Model: The model mapping the hidden layer representation to
            shape network weights and biases.
        """
        input_lr = tf.keras.layers.Input(shape=(self.pi_hidden), name="input_lr_to_w")
        # this model: hidden LR -> weights and biases of shapenet
        return Model(inputs=[input_lr], outputs=[self.pnet_list[-1](input_lr)])

    def model_x_to_u_given_w(self):
        """
        Builds and returns a model that maps input states to output, given shape
        network weights and biases.

        Returns:
            tf.keras.Model: The model mapping input states to output, given shape
            network weights and biases.
        """
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
        """
        Saves the NIF model configuration to a JSON file.

        Args:
            filename (str, optional): The name of the file to save the
            configuration. Defaults to "config.json".
        """
        config = {
            "cfg_shape_net": self.cfg_shape_net,
            "cfg_parameter_net": self.cfg_parameter_net,
            "mixed_policy": self.mixed_policy.name,
        }
        with open(filename, "w") as write_file:
            json.dump(config, write_file, indent=4)


class NIFMultiScale(NIF):
    """
    The NIFMultiScale class is a subclass of the NIF class, extending its functionality
    to support multiscale computations. The class is designed to work with neural
    implicit flow in a multiscale context.

    Attributes:
        cfg_shape_net (dict): Configuration dictionary for the shape network.
        cfg_parameter_net (dict): Configuration dictionary for the parameter network.
        mixed_policy (str, optional): The mixed precision policy to be used for
            TensorFlow computations. Defaults to "float32".
    """

    def __init__(self, cfg_shape_net, cfg_parameter_net, mixed_policy="float32"):
        """
        Initializes an instance of the NIFMultiScale class.

        Args:
            cfg_shape_net (dict): Configuration dictionary for the shape network.
            cfg_parameter_net (dict): Configuration dictionary for the parameter network.
            mixed_policy (str, optional): The mixed precision policy to be used for
                TensorFlow computations. Defaults to "float32".
        """
        super(NIFMultiScale, self).__init__(
            cfg_shape_net, cfg_parameter_net, mixed_policy
        )

    def call(self, inputs, training=None, mask=None):
        """
        Implements the forward pass of the NIFMultiScale model, which takes the
        input tensors and computes the output using the multiscale architecture.

        Args:
            inputs (tensor): Input tensor with concatenated parameter and
                shape information.
            training (bool, optional): Whether the model is in training mode. Defaults
                to None, which will use the model's training mode.
            mask (tensor, optional): Mask tensor for masked input. Defaults to None.

        Returns:
            tensor: Output tensor computed by the multiscale shape network.
        """
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
        Generate the layers for the parameter net, given the configuration of the
        shape net. You will also need the last layer to be consistent with the
        total number of ShapeNet's weights and biases.

        Args:
            cfg_parameter_net (dict): Configuration dictionary for the parameter net.
            cfg_shape_net (dict): Configuration dictionary for the shape net.

        Returns:
            pnet_layers_list (list): List of layers for the parameter net.
        """

        if not isinstance(cfg_parameter_net, dict):
            raise TypeError("cfg_parameter_net must be a dictionary")
        if not isinstance(cfg_shape_net, dict):
            raise TypeError("cfg_shape_net must be a dictionary")
        assert (
            "use_resblock" in cfg_shape_net.keys()
        ), "`use_resblock` should be in cfg_shape_net"
        # assert 'nn_type' in cfg_parameter_net.keys(), "`nn_type` should
        # be in cfg_parameter_net"
        assert (
            type(cfg_shape_net["use_resblock"]) == bool
        ), "cfg_shape_net['use_resblock'] must be a bool"

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
        Distribute `pnet_output` into weight and bias to construct the shape network.

        Args:
            input_s (tf.Tensor): Input tensor for the shape network.
            pnet_output (tf.Tensor): Output tensor from the parameter network.
            flag_resblock (bool): Indicates whether to use a ResNet block structure.
            omega_0 (float): Scaling factor for the sine activation function.
            si_dim (int): Dimension of the input space for the shape network.
            so_dim (int): Dimension of the output space for the shape network.
            n_sx (int): Number of neurons in the shape network's hidden layers.
            l_sx (int): Number of hidden layers in the shape network.
            variable_dtype (tf.DType): Data type for the resulting tensor.

        Returns:
            tf.Tensor: The output tensor of the shape network with the given data type.
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
        """
        Constructs a Keras model for mapping input tensor `x` to output tensor `u`
        given the weights and biases from the parameter network.

        Returns:
            tf.keras.Model: A Keras model that takes two inputs, `input_s` and
                            `input_pnet`, and returns the output tensor `u`.
        """
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
    """
    NIFMultiScaleLastLayerParameterized is a subclass of NIFMultiScale representing a
    Neural Information Flow (NIF) model with a multi-scale architecture that parameterizes
    only the last layer of the shape network. This class is designed to have its shape
    network and parameter network work in conjunction to map the inputs to the outputs.

    Attributes:
        cfg_shape_net (dict): Configuration for the shape network.
        cfg_parameter_net (dict): Configuration for the parameter network.
        mixed_policy (str): Policy to be used for mixed precision calculations.

    Methods:
        call(inputs, training=None, mask=None): Implements the forward pass of the model.
        model_p_to_lr(): Returns a Keras model that maps input parameters to hidden learning rates.
        model_x_to_phi(): Returns a Keras model that maps input spatial data to the output phi_x.
        model_lr_to_w(): Raises a ValueError as 'w' is the same as 'lr' in this class.
        model_x_to_u_given_w(): Returns a Keras model that maps input spatial data to the
            :wq:wq
            :wq
            output 'u' given 'w'.
    """

    def __init__(self, cfg_shape_net, cfg_parameter_net, mixed_policy="float32"):
        """
        Initialize the NIFMultiScaleLastLayerParameterized class.

        Args:
            cfg_shape_net (dict): Configuration dictionary for the shape network.
            cfg_parameter_net (dict): Configuration dictionary for the parameter network.
            mixed_policy (str, optional): Policy for mixed precision training. Defaults to "float32".
        """
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

        self.snet_list = self._initialize_snet(cfg_shape_net)
        self.last_bias_layer = BiasAddLayer(self.so_dim, mixed_policy=self.mixed_policy)

    def call(self, inputs, training=None, mask=None):
        """
        Forward pass of the NIFMultiScaleLastLayerParameterized model.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, input_dim).
            training (bool, optional): Whether the model is in training mode. Defaults to None.
            mask (tf.Tensor, optional): Mask tensor for masked inputs. Defaults to None.

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        input_p = inputs[:, 0 : self.pi_dim]
        input_s = inputs[:, self.pi_dim : self.pi_dim + self.si_dim]
        # get parameter from parameter_net
        self.pnet_output = self._call_parameter_net(input_p, self.pnet_list)[0]
        return self._call_shape_net_mres_only_para_last_layer(
            tf.cast(input_s, self.compute_Dtype),
            self.snet_list,
            # tf.cast(self.last_layer_bias, self.compute_Dtype),
            self.pnet_output,
            self.so_dim,
            self.pi_hidden,
            self.variable_Dtype,
        )

    def model_p_to_lr(self):
        """
        Creates a Keras model for mapping input parameters to hidden layer representation.

        Returns:
            tf.keras.Model: A Keras model that maps input parameters (t, mu) to a hidden
            layer representation (LR).
        """
        input_p = tf.keras.layers.Input(shape=(self.pi_dim), name="input_p_to_lr")
        # this model: t, mu -> hidden LR
        return Model(
            inputs=[input_p],
            outputs=[self._call_parameter_net(input_p, self.pnet_list)[0]],
        )

    def model_x_to_phi(self):
        """
        Creates a Keras model to compute phi_x from input spatial features.

        Returns:
            tf.keras.Model: A Keras model that takes input spatial features and computes
            the phi_x values using the shape network.
        """
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
        """
        Raises an error as 'w' and 'lr' are the same in NIFMultiScaleLastLayerParameterized.

        Raises:
            ValueError: Error stating that 'w' is the same as 'lr' in this class.
        """
        raise ValueError(
            "In this class: NIFMultiScaleLastLayerParameterization, `w` is the same as `lr`"
        )

    def model_x_to_u_given_w(self):
        """
        Creates a Keras model that maps input_s and input_pnet to the output of the
        shape network with the given parameters.

        Returns:
            Model: A Keras model with input_s and input_pnet as inputs, and the output
                   of the shape network as the output.
        """
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
        """
        Initializes the shape network layers based on the configuration.

        Args:
            cfg_shape_net (dict): Configuration dictionary for the shape network.

        Returns:
            List[Layer]: A list of initialized Keras layers for the shape network.
        """
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
            self.po_dim * self.so_dim,  # todo: change this to self.sl_dim * self.so_dim
            "bottleneck",
            cfg_shape_net["omega_0"],
            cfg_shape_net,
            self.snet_kernel_regularizer,
            self.snet_bias_regularizer,
            self.mixed_policy,
            name="siren_bottleneck_snet",
        )
        snet_layers_list.append(bottle_last_layer)

        return snet_layers_list

    def _call_shape_net_get_phi_x(self, input_s, snet_layers_list, so_dim, pi_hidden):
        """
        Compute the phi_x matrix for the given input and shape network layers.

        Args:
            input_s (tf.Tensor): Input tensor for the shape network.
            snet_layers_list (List[Layer]): List of Keras layers for the shape network.
            so_dim (int): Output spatial dimension.
            pi_hidden (int): Hidden dimension for the parameter network.

        Returns:
            tf.Tensor: The computed phi_x matrix.
        """
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
        # last_layer_bias,
        pnet_output,
        so_dim,
        pi_hidden,
        variable_dtype,
    ):
        """
        Compute the output tensor u for the given input and network layers.

        Args:
            input_s (tf.Tensor): Input tensor for the shape network.
            snet_layers_list (List[Layer]): List of Keras layers for the shape network.
            pnet_output (tf.Tensor): Output tensor from the parameter network.
            so_dim (int): Output spatial dimension.
            pi_hidden (int): Hidden dimension for the parameter network.
            variable_dtype (tf.DType): Data type for the output tensor.

        Returns:
            tf.Tensor: The computed output tensor u.
        """
        phi_x_matrix = self._call_shape_net_get_phi_x(
            input_s, snet_layers_list, so_dim, pi_hidden
        )
        u = tf.keras.layers.Dot(axes=(2, 1))([phi_x_matrix, pnet_output])
        u = self.last_bias_layer(u)
        return tf.cast(u, variable_dtype, name="output_cast")
