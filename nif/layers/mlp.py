import tensorflow as tf
import tensorflow_model_optimization as tfmot


class MLP_ResNet(tf.keras.layers.Layer, tfmot.sparsity.keras.PrunableLayer):
    """
    A fully connected neural network with residual connections.

    Attributes:
        compute_Dtype (tf.dtypes.DType): The floating-point precision used for computation.
        variable_Dtype (tf.dtypes.DType): The floating-point precision used for variables.
        act (function): The activation function to use.
        L1 (tf.keras.layers.Dense): The first fully connected layer.
        L2 (tf.keras.layers.Dense): The second fully connected layer.

    Args:
        width (int): The width of the fully connected layers.
        activation (str): The name of the activation function to use.
        kernel_initializer (str): The name of the initializer to use for the kernel weights.
        bias_initializer (str): The name of the initializer to use for the bias weights.
        kernel_regularizer (str): The name of the regularizer to use for the kernel weights.
        bias_regularizer (str): The name of the regularizer to use for the bias weights.
        mixed_policy (tf.keras.mixed_precision.Policy): The policy to use for mixed-precision training.
        **kwargs: Additional keyword arguments to pass to the base class constructor.
    """

    def __init__(
        self,
        width,
        activation,
        kernel_initializer,
        bias_initializer,
        kernel_regularizer,
        bias_regularizer,
        mixed_policy,
        **kwargs
    ):
        super(MLP_ResNet, self).__init__(**kwargs)
        self.compute_Dtype = mixed_policy.compute_dtype
        self.variable_Dtype = mixed_policy.variable_dtype
        self.act = tf.keras.activations.get(activation)
        self.L1 = tf.keras.layers.Dense(
            width,
            activation=tf.keras.activations.get(activation),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            dtype=mixed_policy,
            name=kwargs.get("name", "MLP_ResNet") + "_dense_1",
        )
        self.L2 = tf.keras.layers.Dense(
            width,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            dtype=mixed_policy,
            name=kwargs.get("name", "MLP_ResNet") + "_dense_1",
        )

    def call(self, x, **kwargs):
        """
        Forward pass of the neural network with residual connections.

        Args:
            x (tf.Tensor): The input tensor to the network.

        Returns:
            tf.Tensor: The output tensor of the network.
        """
        h1 = self.L1(x)
        h2 = self.L2(h1)
        y = self.act(x + tf.cast(h2, self.compute_Dtype))
        return tf.cast(
            y,
            self.variable_Dtype,
            name=kwargs.get("name", "MLP_ResNet") + "_output_cast",
        )

    def get_config(self):
        """
        Returns the configuration of the layer.

        Returns:
            dict: The configuration of the layer.
        """
        config = super().get_config()
        config.update({})
        return config

    def get_prunable_weights(self):
        """
        Returns the list of prunable weights of the layer.

        Returns:
            list: The list of prunable weights of the layer.
        """
        return self.L1.weights + self.L2.weights


class MLP_SimpleShortCut(tf.keras.layers.Layer, tfmot.sparsity.keras.PrunableLayer):
    """
    A fully connected layer with a skip connection that adds the input tensor to the output tensor.
    Inherits from tf.keras.layers.Layer and tfmot.sparsity.keras.PrunableLayer.

    Args:
        width (int): The number of neurons in the layer.
        activation (str): The activation function to use for the layer.
        kernel_initializer: Initializer for the kernel weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the kernel weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        mixed_policy: Floating-point precision to use for computing the layer.
        **kwargs: Additional keyword arguments to pass to the base class constructor.
    """

    def __init__(
        self,
        width,
        activation,
        kernel_initializer,
        bias_initializer,
        kernel_regularizer,
        bias_regularizer,
        mixed_policy,
        **kwargs
    ):
        super(MLP_SimpleShortCut, self).__init__(**kwargs)
        self.width = width
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.mixed_policy = mixed_policy
        self.L1 = tf.keras.layers.Dense(
            width,
            activation=tf.keras.activations.get(activation),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            dtype=mixed_policy,
            name=kwargs.get("name", "MLP_SimpleShortCut") + "_dense",
        )

    def call(self, x, **kwargs):
        """
        Applies the fully connected layer with a skip connection to the input tensor.

        Args:
            x: Input tensor to the layer.
            **kwargs: Additional keyword arguments to pass to the call method.

        Returns:
            The result of applying the layer to the input tensor with a skip connection added.
        """
        y = x + self.L1(x)
        return y

    def get_config(self):
        """
        Returns the configuration of the layer.

        Returns:
            A dictionary containing the configuration of the layer.
        """
        config = super().get_config().copy()
        config.update(
            {
                "width": self.width,
                "activation": self.activation,
                "kernel_initializer": self.kernel_initializer,
                "bias_initializer": self.bias_initializer,
                "kernel_regularizer": self.kernel_regularizer,
                "bias_regularizer": self.bias_regularizer,
                "mixed_policy": self.mixed_policy,
            }
        )
        return config

    def get_prunable_weights(self):
        """
        Returns the weights of the layer that can be pruned.

        Returns:
            The weights of the layer that can be pruned.
        """
        return self.L1.weights


class EinsumLayer(tf.keras.layers.Layer):
    """
    A custom layer that wraps a single tf.einsum operation.

    Usage:
    x = EinsumLayer("bmhwf,bmoh->bmowf")((x1, x2))

    Args:
        equation (str): The Einstein summation notation equation to use for the operation.
        **kwargs: Additional keyword arguments to pass to the base class constructor.
    """

    def __init__(self, equation: str, **kwargs):
        super().__init__(**kwargs)
        self.equation = equation

    def call(self, inputs, *args, **kwargs):
        """
        Performs the tf.einsum operation on the inputs.

        Args:
            inputs (tuple[tf.Tensor]): A tuple of input tensors to the operation.

        Returns:
            tf.Tensor: The output tensor of the operation.
        """
        return tf.einsum(self.equation, *inputs)

    def get_config(self):
        """
        Returns the configuration of the layer.

        Returns:
            dict: A dictionary containing the configuration of the layer.
        """
        return {"equation": self.equation}


class BiasAddLayer(tf.keras.layers.Layer):
    """
    A custom layer that adds a bias vector to the inputs.

    Args:
        output_dim (int): The dimensionality of the output space.
        mixed_policy (str): The floating-point precision to use for the bias vector.
        **kwargs: Additional keyword arguments to pass to the base class constructor.
    """

    def __init__(self, output_dim, mixed_policy, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.mixed_policy = mixed_policy
        last_layer_init = tf.keras.initializers.TruncatedNormal(stddev=0.1)
        self.last_layer_bias = tf.Variable(
            last_layer_init([output_dim]),
            dtype=self.mixed_policy.variable_dtype,
            name="last_layer_bias_snet",
        )

    def call(self, inputs):
        """
        Adds the bias vector to the input tensor.

        Args:
            inputs (tf.Tensor): The input tensor to add the bias vector to.

        Returns:
            tf.Tensor: The input tensor with the bias vector added to it.
        """
        return inputs + self.last_layer_bias

    def get_config(self):
        """
        Returns the configuration of the layer.

        Returns:
            dict: A dictionary containing the configuration of the layer.
        """
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "mixed_policy": self.mixed_policy,
            }
        )
        return config
