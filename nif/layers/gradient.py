import tensorflow as tf


class JacobianLayer(tf.keras.layers.Layer):
    def __init__(self, model, y_index, x_index, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.y_index = y_index
        self.x_index = x_index

    @tf.function
    def call(self, x, **kwargs):
        y, dys_dxs = compute_output_and_grad(self.model, x, self.x_index, self.y_index)
        return y, dys_dxs


class JacRegLatentLayer(JacobianLayer):
    """
    Jacobian regularization of parameter net:
        y = f_{para}(x)
        Loss = mean((df/dx)^2)
    """

    def __init__(
        self,
        model,
        y_index,
        x_index,
        l1=1e-2,
        mixed_policy=tf.keras.mixed_precision.Policy("float32"),
        **kwargs
    ):
        super().__init__(model, y_index, x_index, mixed_policy=mixed_policy, **kwargs)
        self.l1 = tf.cast(l1, self.mixed_policy.compute_dtype).numpy()

    @tf.function
    def call(self, x, **kwargs):
        y, dls_dxs = compute_output_and_augment_grad(
            self.model, x, self.x_index, self.y_index
        )
        jac_reg_loss = self.l1 * tf.reduce_mean(tf.square(dls_dxs))
        self.add_loss(jac_reg_loss)
        return y

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "l1": self.l1,
            }
        )
        return config


class HessianLayer(tf.keras.layers.Layer):
    def __init__(self, model, y_index, x_index):
        super().__init__()
        self.model = model
        self.y_index = y_index
        self.x_index = x_index

    @tf.function
    def call(self, x, **kwargs):
        y, dys_dxs, dys2_dxs2 = compute_output_and_grad_and_hessian(
            self.model, x, self.x_index, self.y_index
        )
        return y, dys_dxs, dys2_dxs2


def compute_output_and_augment_grad(model, x, x_index, y_index):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y, layer_ = model(x)
        ls = tf.gather(layer_, y_index, axis=-1)
    dls_dx = tape.batch_jacobian(ls, x)
    dls_dxs = tf.gather(dls_dx, x_index, axis=-1)
    return y, dls_dxs


def compute_output_and_grad(model, x, x_index, y_index):
    ys_list = []
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = model(x)
        for i in y_index:
            ys_list.append(y[:, i])
    dys_dx = tf.stack([tape.gradient(q, x) for q in ys_list], 1)
    dys_dxs = tf.gather(dys_dx, x_index, axis=-1)
    del tape
    return y, dys_dxs


def compute_output_and_grad_and_hessian(model, x, x_index, y_index):
    with tf.GradientTape() as g:
        g.watch(x)
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = model(x)
            ys = tf.gather(y, y_index, axis=-1)
        dys_dx = tape.batch_jacobian(ys, x)
        dys_dxs = tf.gather(dys_dx, x_index, axis=-1)
    dys_dx2 = g.batch_jacobian(dys_dxs, x)
    dys_dxs2 = tf.gather(dys_dx2, x_index, axis=-1)
    return y, dys_dxs, dys_dxs2
