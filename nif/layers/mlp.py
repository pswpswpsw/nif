import tensorflow as tf
import tensorflow_model_optimization as tfmot


class MLP_ResNet(tf.keras.layers.Layer, tfmot.sparsity.keras.PrunableLayer):
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
        # dtype = tf.float16 if mixed_policy == 'mixed_float16' else tf.float32
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
        # classic ResNet, replace ReLU with Swish
        h1 = self.L1(x)
        h2 = self.L2(h1)
        y = self.act(
            x + tf.cast(h2, self.compute_Dtype)
        )  # , name=kwargs.get('name', 'MLP_ResNet') + '_act')
        return tf.cast(
            y,
            self.variable_Dtype,
            name=kwargs.get("name", "MLP_ResNet") + "_output_cast",
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                # "width": self.width,
                # "activation": self.activation,
                # "kernel_initializer": self.kernel_initializer,
                # "bias_initializer": self.bias_initializer,
                # "mixed_policy": self.mixed_policy
            }
        )
        return config

    def get_prunable_weights(self):
        return self.L1.weights + self.L2.weights


class MLP_SimpleShortCut(tf.keras.layers.Layer, tfmot.sparsity.keras.PrunableLayer):
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
        # dtype = tf.float16 if mixed_policy == 'mixed_float16' else tf.float32
        self.width = width
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        # self.compute_Dtype = mixed_policy.compute_dtype
        # self.variable_Dtype = mixed_policy.variable_dtype
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
        # classic ResNet, replace ReLU with Swish
        y = x + self.L1(x)
        return y

    def get_config(self):
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
        return self.L1.weights


class EinsumLayer(tf.keras.layers.Layer):
    """
    Layer wrapping a single tf.einsum operation.

    Usage:
    x = EinsumLayer("bmhwf,bmoh->bmowf")((x1, x2))
    """

    def __init__(self, equation: str, **kwargs):
        super().__init__(**kwargs)
        self.equation = equation

    def call(self, inputs, *args, **kwargs):
        return tf.einsum(self.equation, *inputs)

    def get_config(self):
        return {"equation": self.equation}


class BiasAddLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, mixed_policy, **kwargs):
        super().__init__(**kwargs)
        # self.scale = tf.Variable(1.)
        self.output_dim = output_dim
        self.mixed_policy = mixed_policy
        # create bias for the last layer
        # last_layer_init = initializers.TruncatedNormal(stddev=1e-16)
        last_layer_init = tf.keras.initializers.TruncatedNormal(stddev=0.1)
        self.last_layer_bias = tf.Variable(
            last_layer_init([output_dim]),
            dtype=self.mixed_policy.variable_dtype,
            name="last_layer_bias_snet",
        )

    def call(self, inputs):
        return inputs + self.last_layer_bias
    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "mixed_policy": self.mixed_policy,
        })
        return config