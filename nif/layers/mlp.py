import tensorflow as tf

class MLP_ResNet(tf.keras.layers.Layer):
    def __init__(self, width, activation, kernel_initializer, bias_initializer, mixed_policy):
        super(MLP_ResNet, self).__init__()
        dtype = tf.float16 if mixed_policy == 'mixed_float16' else tf.float32
        self.act = tf.keras.activations.get(activation)
        self.L1 = tf.keras.layers.Dense(width, activation=self.act,
                                        kernel_initializer=kernel_initializer,
                                        bias_initializer=bias_initializer,
                                        # dtype=dtype
                                        )
        self.L2 = tf.keras.layers.Dense(width,
                                        kernel_initializer=kernel_initializer,
                                        bias_initializer=bias_initializer,
                                        # dtype=dtype
                                        )

    def call(self, x, **kwargs):
        # classic ResNet, replace ReLU with Swish
        h1 = self.L1(x)
        h2 = self.L2(h1)
        y = self.act(x + h2)
        return y


class MLP_SimpleShortCut(tf.keras.layers.Layer):
    def __init__(self, width, activation, kernel_initializer, bias_initializer, mixed_policy):
        super(MLP_SimpleShortCut, self).__init__()
        dtype = tf.float16 if mixed_policy == 'mixed_float16' else tf.float32
        act = tf.keras.activations.get(activation)
        self.L1 = tf.keras.layers.Dense(width, activation=act,
                                        kernel_initializer=kernel_initializer,
                                        bias_initializer=bias_initializer,
                                        # dtype=dtype
                                        )

    def call(self, x, **kwargs):
        # classic ResNet, replace ReLU with Swish
        y = x + self.L1(x)
        return y