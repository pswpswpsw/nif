import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot


def gen_hypernetwork_weights_bias_for_siren_shapenet(
    num_inputs,
    num_outputs,
    weight_factor,
    num_weight_first,
    num_weight_hidden,
    num_weight_last,
    input_dim,
    width,
    omega_0,
    variable_dtype,
):
    """
    Generates initial weights and biases for a hypernetwork for a SIREN shape network.

    Args:
        num_inputs (int): Number of inputs to the network.
        num_outputs (int): Number of outputs of the hypernetwork.
        weight_factor (float): Scaling factor for the weight initialization.
        num_weight_first (int): Number of weights in the first layer.
        num_weight_hidden (int): Number of weights in each hidden layer.
        num_weight_last (int): Number of weights in the last layer.
        input_dim (int): Dimensionality of the input.
        width (int): Width of the hidden layers.
        omega_0 (float): Frequency scale factor.
        variable_dtype: Data type for the variables.

    Returns:
        Tuple containing the initial weights and biases.
    """
    w_init = tf.random.uniform(
        (num_inputs, num_outputs),
        -np.sqrt(6.0 / num_inputs) * weight_factor,
        np.sqrt(6.0 / num_inputs) * weight_factor,
    )

    scale_matrix = np.ones((num_outputs), dtype=variable_dtype)
    scale_matrix[:num_weight_first] /= input_dim  # 1st layer weights
    scale_matrix[num_weight_first : num_weight_first + num_weight_hidden] *= (
        np.sqrt(6.0 / width) / omega_0
    )  # hidden layer weights
    scale_matrix[
        num_weight_first
        + num_weight_hidden : num_weight_first
        + num_weight_hidden
        + num_weight_last
    ] *= np.sqrt(
        6.0 / (width + width)
    )  # last layer weights, since it is linear layer and no scaling,
    # we choose GlorotUniform
    scale_matrix[
        num_weight_first + num_weight_hidden + num_weight_last :
    ] /= width  # all biases

    b_init = tf.random.uniform(
        (num_outputs,), -scale_matrix, scale_matrix, dtype=variable_dtype
    )
    return w_init, b_init


def compute_number_of_weightbias_by_its_position_for_shapenet(cfg_shape_net):
    """
    Computes the number of weights and biases for each position in the shape network.

    Args:
        cfg_shape_net (dict): A dictionary containing the configuration parameters for the shape network.

    Returns:
        Tuple: A tuple containing the number of weights for the first layer, the number of
               weights for the hidden layers,
        and the number of weights for the output layer.
    """
    si_dim = cfg_shape_net["input_dim"]
    so_dim = cfg_shape_net["output_dim"]
    n_sx = cfg_shape_net["units"]
    l_sx = cfg_shape_net["nlayers"]

    if cfg_shape_net["connectivity"] == "full":
        num_weight_first = si_dim * n_sx
        if cfg_shape_net["use_resblock"]:
            num_weight_hidden = (2 * l_sx) * n_sx**2
        else:
            num_weight_hidden = l_sx * n_sx**2
    elif cfg_shape_net["connectivity"] == "last":
        num_weight_first = 0
        num_weight_hidden = 0
    else:
        raise ValueError(
            "check cfg_shape_net['connectivity'] value, it can only be 'last' or 'full'"
        )
    num_weight_last = so_dim * n_sx
    return num_weight_first, num_weight_hidden, num_weight_last


class SIREN(tf.keras.layers.Layer, tfmot.sparsity.keras.PrunableLayer):
    """
    A class representing the SIREN layer.

    Args:
        num_inputs (int): The number of input units.
        num_outputs (int): The number of output units.
        layer_position (str): The position of the SIREN layer in the network architecture.
            Possible values are 'first', 'hidden', 'last' and 'bottleneck'.
        omega_0 (float): The cutoff frequency for the initial frequency. Should be set to 1.0
            for most cases.
        cfg_shape_net (dict): A dictionary containing the configuration parameters for the network
            if the layer position is 'last'.
        kernel_regularizer (tf.keras.regularizers.Regularizer): Regularizer function applied to
            the kernel weights matrix.
        bias_regularizer (tf.keras.regularizers.Regularizer): Regularizer function applied to the
            bias vector.
        mixed_policy (tf.keras.mixed_precision.Policy): A mixed precision policy used for the
            weights and biases.
        **kwargs: Additional arguments.

    Attributes:
        w_init (tf.Tensor): The initialized weights.
        b_init (tf.Tensor): The initialized biases.
        w (tf.Variable): The learnable weights.
        b (tf.Variable): The learnable biases.
        compute_Dtype (tf.DType): The computational datatype of the layer.

    Methods:
        call(inputs, **kwargs): Defines the computation performed at every call.
        get_config(): Returns the config of the layer.
        get_prunable_weights(): Returns the prunable weights of the layer.
    """

    def __init__(
        self,
        num_inputs,
        num_outputs,
        layer_position,
        omega_0,
        cfg_shape_net=None,
        kernel_regularizer=None,
        bias_regularizer=None,
        mixed_policy=tf.keras.mixed_precision.Policy("float32"),
        **kwargs
    ):
        """
        Initializes a SIREN layer.

        Args:
            num_inputs (int): The number of input dimensions.
            num_outputs (int): The number of output dimensions.
            layer_position (str): The position of the layer in the neural network, which can be one of
                the following: 'first', 'hidden', 'bottleneck', or 'last'.
            omega_0 (float): The frequency parameter for SIREN activation function.
            cfg_shape_net (Optional[dict]): Configuration dictionary for shapenet.
            kernel_regularizer (Optional[tf.keras.regularizers.Regularizer]): Regularizer function applied
                to the kernel weights.
            bias_regularizer (Optional[tf.keras.regularizers.Regularizer]): Regularizer function applied
                to the bias weights.
            mixed_policy (tf.keras.mixed_precision.Policy): Policy for mixed precision training.
            **kwargs: Additional keyword arguments.

        """
        super(SIREN, self).__init__(**kwargs)
        # self.num_inputs = num_inputs
        # self.num_outputs = num_outputs
        self.layer_position = layer_position
        # self.cfg_shape_net = cfg_shape_net
        # self.mixed_policy = mixed_policy
        self.compute_Dtype = mixed_policy.compute_dtype
        # self.\
        variable_Dtype = mixed_policy.variable_dtype
        self.omega_0 = tf.cast(omega_0, variable_Dtype)
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        # initialize the weights
        if layer_position == "first":
            w_init = tf.random.uniform(
                (num_inputs, num_outputs),
                -1.0 / num_inputs,
                1.0 / num_inputs,
                dtype=variable_Dtype,
            )
            b_init = tf.random.uniform(
                (num_outputs,),
                -1.0 / np.sqrt(num_inputs),
                1.0 / np.sqrt(num_inputs),
                dtype=variable_Dtype,
            )

        elif layer_position == "hidden" or layer_position == "bottleneck":
            w_init = tf.random.uniform(
                (num_inputs, num_outputs),
                -tf.math.sqrt(6.0 / num_inputs) / self.omega_0,
                tf.math.sqrt(6.0 / num_inputs) / self.omega_0,
                dtype=variable_Dtype,
            )
            b_init = tf.random.uniform(
                (num_outputs,),
                -1.0 / np.sqrt(num_inputs),
                1.0 / np.sqrt(num_inputs),
                dtype=variable_Dtype,
            )

        elif layer_position == "last":
            if isinstance(cfg_shape_net, dict):
                raise ValueError(
                    "No value for dictionary: cfg_shape_net for {} SIREN layer {}".format(
                        layer_position, self.name
                    )
                )

            # compute the indices needed for generating weights for shapenet
            # if connectivity_e == 'full':
            (
                num_weight_first,
                num_weight_hidden,
                num_weight_last,
            ) = compute_number_of_weightbias_by_its_position_for_shapenet(cfg_shape_net)
            # elif connectivity_e == 'last_layer':
            #     num_weight_first = 0
            #     num_weight_hidden = 0
            #     num_weight_last = num_outputs // cfg_shape_net['output_dim']
            # # note that po_dim*so_dim / so_dim = po_dim
            # else:
            #     raise ValueError("`connectivity_e` has an invalid value {}".format(connectivity_e))

            w_init, b_init = gen_hypernetwork_weights_bias_for_siren_shapenet(
                num_inputs=num_inputs,
                num_outputs=num_outputs,
                weight_factor=cfg_shape_net["weight_init_factor"],
                num_weight_first=num_weight_first,
                num_weight_hidden=num_weight_hidden,
                num_weight_last=num_weight_last,
                input_dim=cfg_shape_net["input_dim"],
                width=cfg_shape_net["units"],
                omega_0=self.omega_0,
                variable_dtype=variable_Dtype,
            )

        else:
            raise NotImplementedError(
                "No implementation of layer_position for {}".format(layer_position)
            )

        self.w_init = w_init
        self.b_init = b_init
        self.w = tf.Variable(
            w_init, dtype=variable_Dtype, name=kwargs.get("name", "siren") + "_w"
        )
        self.b = tf.Variable(
            b_init, dtype=variable_Dtype, name=kwargs.get("name", "siren") + "_b"
        )

    def call(self, x, **kwargs):
        """
        Compute the output of the layer given an input tensor x.

        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        if self.kernel_regularizer is not None:
            self.add_loss(self.kernel_regularizer(self.w))
        if self.bias_regularizer is not None:
            self.add_loss(self.bias_regularizer(self.b))

        if self.layer_position == "last" or self.layer_position == "bottleneck":
            y = tf.matmul(x, tf.cast(self.w, self.compute_Dtype)) + tf.cast(
                self.b, self.compute_Dtype
            )
        else:
            y = tf.math.sin(
                tf.cast(self.omega_0, self.compute_Dtype)
                * tf.matmul(x, tf.cast(self.w, self.compute_Dtype))
                + tf.cast(self.b, self.compute_Dtype)
            )
        return y

    def get_config(self):
        """Returns the configuration of the layer.

        Returns:
            dict: The configuration of the layer.
        """
        config = super().get_config()
        config.update(
            {
                "layer_position": self.layer_position,
                "omega_0": self.omega_0,
            }
        )
        return config

    def get_prunable_weights(self):
        """Returns the list of prunable weights of the layer.

        Returns:
            list: The list of prunable weights of the layer.
        """
        return [self.w]


class SIREN_ResNet(SIREN):
    """
    A subclass of the SIREN class implementing a residual block.

    Args:
        num_inputs (int): Number of input features.
        num_outputs (int): Number of output features.
        omega_0 (float): Frequency parameter for the SIREN activation function.
        kernel_regularizer (tf.keras.regularizers.Regularizer): Regularizer function
            applied to the layer's weights.
        bias_regularizer (tf.keras.regularizers.Regularizer): Regularizer function
            applied to the layer's biases.
        mixed_policy (tf.keras.mixed_precision.Policy): Policy to use for mixed
            precision computation. Defaults to "float32".
        **kwargs: Additional keyword arguments to pass to the parent class constructor.

    Attributes:
        w2 (tf.Variable): Weight variable for the second layer of the residual block.
        b2 (tf.Variable): Bias variable for the second layer of the residual block.

    Methods:
        call(x, training=None, mask=None):
            Performs a forward pass through the layer.
        get_prunable_weights():
            Returns a list of prunable weight variables.

    """

    def __init__(
        self,
        num_inputs,
        num_outputs,
        omega_0,
        kernel_regularizer=None,
        bias_regularizer=None,
        mixed_policy=tf.keras.mixed_precision.Policy("float32"),
        **kwargs
    ):
        """
        Constructs a new instance of the SIREN_ResNet class.

        Args:
            num_inputs (int): Number of input features.
            num_outputs (int): Number of output features.
            omega_0 (float): Frequency parameter for the SIREN activation function.
            kernel_regularizer (tf.keras.regularizers.Regularizer): Regularizer function
                applied to the layer's weights.
            bias_regularizer (tf.keras.regularizers.Regularizer): Regularizer function
                applied to the layer's biases.
            mixed_policy (tf.keras.mixed_precision.Policy): Policy to use for mixed
                precision computation. Defaults to "float32".
            **kwargs: Additional keyword arguments to pass to the parent class constructor.
        """
        super(SIREN_ResNet, self).__init__(
            num_inputs,
            num_outputs,
            layer_position="hidden",
            omega_0=omega_0,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            mixed_policy=mixed_policy,
            **kwargs
        )
        self.w2 = tf.Variable(
            self.w_init,
            dtype=mixed_policy.variable_dtype,
            name=kwargs.get("name", "siren_ResNet") + "_w2",
        )
        self.b2 = tf.Variable(
            self.b_init,
            dtype=mixed_policy.variable_dtype,
            name=kwargs.get("name", "siren_ResNet") + "_b2",
        )

    def call(self, x, training=None, mask=None):
        """
        Performs a forward pass through the layer.

        Args:
            x (tf.Tensor): Input tensor.
            training (bool): Whether the layer is in training mode.
            mask: Ignored.

        Returns:
            The output tensor of the layer.
        """
        if self.kernel_regularizer is not None:
            self.add_loss(self.kernel_regularizer(self.w))
            self.add_loss(self.kernel_regularizer(self.w2))
        if self.bias_regularizer is not None:
            self.add_loss(self.bias_regularizer(self.b))
            self.add_loss(self.bias_regularizer(self.b2))

        h = tf.math.sin(
            self.omega_0 * tf.matmul(x, tf.cast(self.w, self.compute_Dtype))
            + tf.cast(self.b, self.compute_Dtype)
        )
        return 0.5 * (
            x
            + tf.math.sin(
                self.omega_0 * tf.matmul(h, tf.cast(self.w2, self.compute_Dtype))
                + tf.cast(self.b2, self.compute_Dtype)
            )
        )

    def get_prunable_weights(self):
        """
        Returns the list of prunable weights in the layer, i.e., the weights that can be
            pruned during training.

        Returns:
            List of `tf.Variable` objects representing the prunable weights in the layer.
        """
        return [self.w, self.w2]


class HyperLinearForSIREN(tf.keras.layers.Layer, tfmot.sparsity.keras.PrunableLayer):
    """
    Implements a hypernetwork that generates weights and biases for the SIREN layer.

    Args:
        num_inputs (int): Number of input units.
        num_outputs (int): Number of output units.
        cfg_shape_net (dict): Configuration dictionary of the shape network.
        mixed_policy (tf.keras.mixed_precision.Policy): Policy for mixed precision computation.
        connectivity (str): Connectivity type of the SIREN layer. Should be set to `full` or `last_layer`.
        kernel_regularizer (tf.keras.regularizers.Regularizer): Regularizer for the kernel.
        bias_regularizer (tf.keras.regularizers.Regularizer): Regularizer for the bias.
        activity_regularizer (tf.keras.regularizers.Regularizer): Regularizer for the layer activity.
        **kwargs: Additional layer arguments.

    Attributes:
        kernel_regularizer (tf.keras.regularizers.Regularizer): Regularizer for the kernel.
        bias_regularizer (tf.keras.regularizers.Regularizer): Regularizer for the bias.
        compute_Dtype (tf.dtypes.DType): Data type for computation.
        w (tf.Variable): Variable for the weights.
        b (tf.Variable): Variable for the biases.

    Methods:
        call(self, x, **kwargs): Computes the output of the layer for the input `x`.
        get_config(self): Returns the configuration dictionary of the layer.
        get_prunable_weights(self): Returns the prunable weights of the layer.

    """

    def __init__(
        self,
        num_inputs,
        num_outputs,
        cfg_shape_net,
        mixed_policy,
        connectivity="full",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        **kwargs
    ):
        """
        Initializes the `HyperLinearForSIREN` class.
        """
        super(HyperLinearForSIREN, self).__init__(
            activity_regularizer=activity_regularizer, **kwargs
        )
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        # self.num_inputs = num_inputs
        # self.num_outputs = num_outputs
        # self.mixed_policy = mixed_policy
        self.compute_Dtype = mixed_policy.compute_dtype
        # self.variable_Dtype = mixed_policy.variable_dtype
        # self.connectivity = connectivity
        # compute the indices needed for generating weights for shapenet
        if connectivity == "full":
            (
                num_weight_first,
                num_weight_hidden,
                num_weight_last,
            ) = compute_number_of_weightbias_by_its_position_for_shapenet(cfg_shape_net)
        elif connectivity == "last_layer":
            num_weight_first, num_weight_hidden, num_weight_last = 0, 0, num_outputs
        else:
            raise ValueError("connectivity should be set to `full` or `last_layer`")

        w_init, b_init = gen_hypernetwork_weights_bias_for_siren_shapenet(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            weight_factor=cfg_shape_net["weight_init_factor"],
            num_weight_first=num_weight_first,
            num_weight_hidden=num_weight_hidden,
            num_weight_last=num_weight_last,
            input_dim=cfg_shape_net["input_dim"],
            width=cfg_shape_net["units"],
            omega_0=cfg_shape_net["omega_0"],
            variable_dtype=mixed_policy.variable_dtype,
        )

        self.w = tf.Variable(
            w_init,
            mixed_policy.variable_dtype,
            name=kwargs.get("name", "hyper_siren") + "_w",
        )
        self.b = tf.Variable(
            b_init,
            mixed_policy.variable_dtype,
            name=kwargs.get("name", "hyper_siren") + "_b",
        )

    def call(self, x, **kwargs):
        if self.kernel_regularizer is not None:
            self.add_loss(self.kernel_regularizer(self.w))
        if self.bias_regularizer is not None:
            self.add_loss(self.bias_regularizer(self.b))
        y = tf.matmul(x, tf.cast(self.w, self.compute_Dtype)) + tf.cast(
            self.b, self.compute_Dtype
        )
        return y

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                # "num_inputs": self.num_inputs,
                # "num_outputs": self.num_outputs,
                # "mixed_policy": self.mixed_policy,
                # "connectivity": self.connectivity
            }
        )
        return config

    def get_prunable_weights(self):
        # Prune bias also, though that usually harms model accuracy too much.
        return [self.w]
