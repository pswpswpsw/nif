import tensorflow as tf


class ParameterOutputL1ActReg(tf.keras.layers.Layer):
    """
    Custom Keras layer for parameter output L1 activation regularization.

    Args:
        model (tf.keras.Model): The TensorFlow model whose parameter output activations are to be regularized.
        l1 (float): The weight of the L1 regularization term in the loss function.
    """

    def __init__(self, model, l1=0.1):
        super().__init__()
        self.model = model
        self.l1 = l1

    @tf.function
    def call(self, x, **kwargs):
        """
        Computes the output of the layer.

        Args:
            x (tf.Tensor): The input tensor.

        Returns:
            The output tensor with parameter output L1 activation regularization applied.
        """
        y, po = self.model(x)
        po_reg_loss = self.l1 * tf.norm(po, ord=1)
        self.add_loss(po_reg_loss)
        return y
