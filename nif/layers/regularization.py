import tensorflow as tf


class ParameterOutputL1ActReg(tf.keras.layers.Layer):
    def __init__(self, model, l1=0.1):
        super().__init__()
        self.model = model
        self.l1 = l1

    @tf.function
    def call(self, x, **kwargs):
        y, po = self.model(x)
        po_reg_loss = self.l1 * tf.norm(po, ord=1)
        self.add_loss(po_reg_loss)
        return y
