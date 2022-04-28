import tensorflow as tf

class InputOutputGradientLayer(tf.keras.layers.Layer):
    def __init__(self, model, y_index, x_index,
                 mixed_policy=tf.keras.mixed_precision.Policy('float32'),
                 **kwargs):

        super().__init__(**kwargs)
        self.model = model
        # self.l1 = tf.cast(l1, self.mixed_policy.compute_dtype)
        self.y_index = y_index
        self.x_index = x_index
        self.mixed_policy = mixed_policy

    def call(self, x, **kwargs):
        with tf.GradientTape() as g:
            g.watch(x)
            y = self.model(x)

            ys = y[:,self.y_index]

        dys_dxs_list = []
        for j in range(len(self.y_index)):
            dys_dx = g.gradient(ys[:,j],x)
            dys_dxs = dys_dx[:,self.x_index]
            dys_dxs_list.append(dys_dxs)

        return y, dys_dxs_list

# with tf.GradientTape(persistent=True) as g:
#
#     g.watch(x2)
#
#     # y = model(x_list)
#
#
#
#     h = tf.concat([x,x2],-1)
#     h = Dense(4,activation='sigmoid')(h)
#     y = h
#
#
#     z = []
#     for i in range(1,4):
#         q = y[:,i]
#         z.append(q)
#
# print(y)
# print(z)
#
# for i in range(1,4):
#     print('this is the dy{}/dx'.format(i))
#     # z2=x2
#     # z2 = x2[:-1,0:2]
#     # z2 = tf.slice(x2, [0,0], [10,2])
#     # print(z2)
#     dy_dx = g.gradient(z[i-1],x2)
#     print(dy_dx)
#     print("==========")
#     print('')


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