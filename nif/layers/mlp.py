import tensorflow as tf

class MLP_ResNet(tf.keras.layers.Layer):
    def __init__(self, width, activation, kernel_initializer, bias_initializer, mixed_policy):
        super(MLP_ResNet, self).__init__()
        self.compute_Dtype = mixed_policy.compute_dtype
        self.variable_Dtype = mixed_policy.variable_dtype
        # dtype = tf.float16 if mixed_policy == 'mixed_float16' else tf.float32
        self.act = tf.keras.activations.get(activation)
        self.L1 = tf.keras.layers.Dense(width, activation=self.act,
                                        kernel_initializer=kernel_initializer,
                                        bias_initializer=bias_initializer,
                                        dtype=mixed_policy
                                        )
        self.L2 = tf.keras.layers.Dense(width,
                                        kernel_initializer=kernel_initializer,
                                        bias_initializer=bias_initializer,
                                        dtype=mixed_policy
                                        )

    def call(self, x, **kwargs):
        # classic ResNet, replace ReLU with Swish
        h1 = self.L1(x)
        h2 = self.L2(h1)
        y = self.act(x + tf.cast(h2, self.compute_Dtype))
        return tf.cast(y, self.variable_Dtype)


class MLP_SimpleShortCut(tf.keras.layers.Layer):
    def __init__(self, width, activation, kernel_initializer, bias_initializer, mixed_policy):
        super(MLP_SimpleShortCut, self).__init__()
        # dtype = tf.float16 if mixed_policy == 'mixed_float16' else tf.float32
        self.compute_Dtype = mixed_policy.compute_dtype
        self.variable_Dtype = mixed_policy.variable_dtype
        act = tf.keras.activations.get(activation)
        self.L1 = tf.keras.layers.Dense(width, activation=act,
                                        kernel_initializer=kernel_initializer,
                                        bias_initializer=bias_initializer,
                                        dtype=mixed_policy
                                        )

    def call(self, x, **kwargs):
        # classic ResNet, replace ReLU with Swish
        y = x + self.L1(x)
        return y