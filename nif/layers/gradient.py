import tensorflow as tf

class InputOutputJacobianLayer(tf.keras.layers.Layer):
    def __init__(self, model, l1=1e-3, diff_target_index=None,
                 mixed_policy=tf.keras.mixed_precision.Policy('float32'),
                 **kwargs):

        self.model = model
        self.mixed_policy = mixed_policy
        self.l1 = tf.cast(l1, self.mixed_policy.compute_dtype)
        self.diff_target_index = diff_target_index
        super().__init__(**kwargs)

    def call(self, x, **kwargs):
        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            y = self.model(x)

        dy0_dx0 = g.gradient(y[0],x[0])
        dy_dx = g.gradient(y,x)
        # jac_tensor = g.batch_jacobian(y, x)
        # jac_loss = self.l1*tf.reduce_sum(tf.reduce_mean(tf.square(jac_tensor),axis=1))
        # self.add_loss(jac_loss)
        return y


class GradientLayerV2(tf.keras.layers.Layer):
    """
    Custom layer to compute 1st and 2nd derivatives for Burgers' equation.
    Attributes:
        model: keras network model.
    """

    def __init__(self, layer_list, l1=1e-3,
                 mixed_policy=tf.keras.mixed_precision.Policy('float32'), **kwargs):
        """
        Args:
            model: keras network model.
        """

        self.layer_list = layer_list
        self.mixed_policy = mixed_policy
        self.l1 = tf.cast(l1, self.mixed_policy.compute_dtype)
        super().__init__(**kwargs)

    def call(self, x, **kwargs):
        """
        Computing 1st and 2nd derivatives for Burgers' equation.
        Args:
            x: input variable.
        Returns:
            model output, 1st and 2nd derivatives.
        """

        with tf.GradientTape(watch_accessed_variables=True) as g:
            g.watch(x)
            latent = x
            for l in self.layer_list[:-1]:
                latent = l(latent)
        jac_tensor = g.batch_jacobian(latent, x)
        tf.print(jac_tensor)
        jac_loss = self.l1*tf.reduce_sum(tf.reduce_mean(tf.square(jac_tensor),axis=1))
        self.add_loss(jac_loss)
        y = self.layer_list[-1](latent)
        return y, latent