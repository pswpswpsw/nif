import tensorflow as tf


class JacobianLayer(tf.keras.layers.Layer):
    """
    A custom Keras layer that computes the Jacobian matrix of a TensorFlow model.

    Args:
        model (tf.keras.Model): The TensorFlow model for which to compute the Jacobian.
        y_index (int or List[int]): The index or indices of the output variable(s) to compute the Jacobian
                                    with respect to.
        x_index (int or List[int]): The index or indices of the input variable(s) to compute the Jacobian
                                    with respect to.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: A tuple containing the output of the model and the Jacobian matrix.
    """

    def __init__(self, model, y_index, x_index, **kwargs):
        """
        Initializes a new instance of the JacobianLayer class.

        Args:
            model (tf.keras.Model): The TensorFlow model for which to compute the Jacobian.
            y_index (int or List[int]): The index or indices of the output variable(s) to compute the
                                        Jacobian with respect to.
            x_index (int or List[int]): The index or indices of the input variable(s) to compute the
                                        Jacobian with respect to.
            **kwargs: Additional keyword arguments to pass to the base class constructor.
        """
        super().__init__(**kwargs)
        self.model = model
        self.y_index = y_index
        self.x_index = x_index

    @tf.function
    def call(self, x, **kwargs):
        """
        Computes the output of the model and the Jacobian matrix.

        Args:
            x (tf.Tensor): The input tensor(s) to the model.
            **kwargs: Additional keyword arguments to pass to the underlying TensorFlow function.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple containing the output of the model and the Jacobian matrix.
        """
        y, dys_dxs = compute_output_and_grad(self.model, x, self.x_index, self.y_index)
        return y, dys_dxs


class JacRegLatentLayer(JacobianLayer):
    """
    A custom Keras layer that applies Jacobian regularization to a TensorFlow model's output.

    This layer computes the Jacobian matrix of the output with respect to the input, and
    adds a regularization term to the loss function to encourage the Jacobian to be
    small. The regularization term is given by:

        L = l1 * mean((df/dx)^2)

    where `l1` is a hyperparameter controlling the strength of the regularization,
    `f` is the TensorFlow model, `y` is its output, and `x` is its input.

    Args:
        model (tf.keras.Model): The TensorFlow model for which to compute the Jacobian.
        y_index (int or List[int]): The index or indices of the output variable(s) to compute the
                                    Jacobian with respect to.
        x_index (int or List[int]): The index or indices of the input variable(s) to compute the
                                    Jacobian with respect to.
        l1 (float): The weight of the Jacobian regularization term in the loss function.
        mixed_policy (str): The floating-point precision to use for computing the Jacobian.
        **kwargs: Additional keyword arguments to pass to the base class constructor.
    """

    def __init__(
        self, model, y_index, x_index, l1=1e-2, mixed_policy="float32", **kwargs
    ):
        """
        Initializes a new instance of the JacRegLatentLayer class.

        Args:
            model (tf.keras.Model): The TensorFlow model for which to compute the Jacobian.
            y_index (int or List[int]): The index or indices of the output variable(s) to compute the
                                        Jacobian with respect to.
            x_index (int or List[int]): The index or indices of the input variable(s) to compute the
                                        Jacobian with respect to.
            l1 (float): The weight of the Jacobian regularization term in the loss function.
            mixed_policy (str): The floating-point precision to use for computing the Jacobian.
            **kwargs: Additional keyword arguments to pass to the base class constructor.
        """
        super().__init__(model, y_index, x_index, dtype=mixed_policy, **kwargs)
        self.l1 = tf.cast(l1, self.mixed_policy.compute_dtype).numpy()

    @tf.function
    def call(self, x, **kwargs):
        """
        Computes the output of the model and the Jacobian regularization loss.

        Args:
            x (tf.Tensor): The input tensor(s) to the model.
            **kwargs: Additional keyword arguments to pass to the underlying TensorFlow function.

        Returns:
            tf.Tensor: The output of the model.
        """
        y, dls_dxs = compute_output_and_augment_grad(
            self.model, x, self.x_index, self.y_index
        )
        jac_reg_loss = self.l1 * tf.reduce_mean(tf.square(dls_dxs))
        self.add_loss(jac_reg_loss)
        return y

    def get_config(self):
        """
        Returns the configuration of the layer.

        Returns:
            Dict[str, Any]: The configuration of the layer.
        """
        config = super().get_config().copy()
        config.update(
            {
                "l1": self.l1,
            }
        )
        return config


class HessianLayer(tf.keras.layers.Layer):
    """
    A custom Keras layer that computes the Hessian matrix of a TensorFlow model's output.

    This layer computes the Hessian matrix of the output with respect to the input, and
    returns it along with the first and second derivatives of the output with respect to
    the input. The first derivative is given by the Jacobian matrix, and the second
    derivative is the Hessian matrix.

    Args:
        model (tf.keras.Model): The TensorFlow model for which to compute the Hessian.
        y_index (int or List[int]): The index or indices of the output variable(s) to compute
                                    the Hessian with respect to.
        x_index (int or List[int]): The index or indices of the input variable(s) to compute
                                    the Hessian with respect to.
    """

    def __init__(self, model, y_index, x_index):
        """
        Initializes a new instance of the HessianLayer class.

        Args:
            model (tf.keras.Model): The TensorFlow model for which to compute the Hessian.
            y_index (int or List[int]): The index or indices of the output variable(s) to
                                        compute the Hessian with respect to.
            x_index (int or List[int]): The index or indices of the input variable(s) to
                                        compute the Hessian with respect to.
        """
        super().__init__()
        self.model = model
        self.y_index = y_index
        self.x_index = x_index

    @tf.function
    def call(self, x, **kwargs):
        """
        Computes the output of the model, the Jacobian matrix, and the Hessian matrix.

        Args:
            x (tf.Tensor): The input tensor(s) to the model.
            **kwargs: Additional keyword arguments to pass to the underlying TensorFlow function.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: A tuple containing the output of the model,
                                                    the Jacobian matrix, and the
                                                    Hessian matrix.
        """
        y, dys_dxs, dys2_dxs2 = compute_output_and_grad_and_hessian(
            self.model, x, self.x_index, self.y_index
        )
        return y, dys_dxs, dys2_dxs2


def compute_output_and_augment_grad(model, x, x_index, y_index):
    """
    Computes the output of a model and the Jacobian matrix.

    Args:
        model (tf.keras.Model): The TensorFlow model for which to compute the Jacobian.
        x (tf.Tensor): The input tensor(s) to the model.
        x_index (int or List[int]): The index or indices of the input variable(s)
                                    to compute the Jacobian with respect to.
        y_index (int or List[int]): The index or indices of the output variable(s)
                                    to compute the Jacobian with respect to.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: A tuple containing the output of the model and the Jacobian matrix.
    """
    with tf.GradientTape() as tape:
        tape.watch(x)
        y, layer_ = model(x)
        ls = tf.gather(layer_, y_index, axis=-1)
    dls_dx = tape.batch_jacobian(ls, x)
    dls_dxs = tf.gather(dls_dx, x_index, axis=-1)
    return y, dls_dxs


def compute_output_and_grad(model, x, x_index, y_index):
    """
    Computes the output of a model and the Jacobian matrix.

    Args:
        model (tf.keras.Model): The TensorFlow model for which to compute the Jacobian.
        x (tf.Tensor): The input tensor(s) to the model.
        x_index (int or List[int]): The index or indices of the input variable(s) to
                                    compute the Jacobian with respect to.
        y_index (int or List[int]): The index or indices of the output variable(s) to
                                    compute the Jacobian with respect to.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: A tuple containing the output of the model and the Jacobian matrix.
    """
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
    """
    Computes the output of a model, the Jacobian matrix, and the Hessian matrix.

    Args:
        model (tf.keras.Model): The TensorFlow model for which to compute the Jacobian and Hessian.
        x (tf.Tensor): The input tensor(s) to the model.
        x_index (int or List[int]): The index or indices of the input variable(s) to
                                    compute the Jacobian and Hessian with respect to.
        y_index (int or List[int]): The index or indices of the output variable(s) to
                                    compute the Jacobian and Hessian with respect to.

    Returns:
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: A tuple containing the output of the model,
                                                the Jacobian matrix, and the Hessian
                                                matrix.
    """
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
