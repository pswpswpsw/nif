import tensorflow as tf

class GradientLayer(tf.keras.layers.Layer):
    def __init__(self, model, y_index, x_index,
                 mixed_policy=tf.keras.mixed_precision.Policy('float32')):
        super().__init__()
        self.model = model
        # self.l1 = tf.cast(l1, self.mixed_policy.compute_dtype)
        self.y_index = y_index
        self.x_index = x_index
        self.mixed_policy = mixed_policy

    def call(self, x, **kwargs):
        y, dys_dxs = compute_output_and_gradient(self.model, x, self.x_index, self.y_index)
        return y, dys_dxs

class JacobianRegLayer(GradientLayer):
    def __init__(self, model, y_index, x_index, l1=1e-2,
                 mixed_policy=tf.keras.mixed_precision.Policy('float32')):
        super().__init__(model, y_index, x_index, mixed_policy)
        self.l1 = tf.cast(l1, self.mixed_policy.compute_dtype)

    def call(self, x, **kwargs):
        y, dys_dxs = compute_output_and_gradient(self.model, x, self.x_index, self.y_index)
        jac_reg_loss = self.l1 * tf.reduce_mean(tf.square(dys_dxs))
        self.add_loss(jac_reg_loss)
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



def compute_output_and_gradient(model, x, x_index, y_index):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = model(x)
        ys = tf.gather(y, y_index, axis=-1)
    dys_dx = tape.batch_jacobian(ys,x)
    dys_dxs = tf.gather(dys_dx, x_index, axis=-1)
    return y, dys_dxs