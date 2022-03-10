import tensorflow as tf
import numpy as np

def gen_hypernetwork_weights_bias_for_siren_shapenet(
        num_inputs,
        num_outputs,
        weight_factor,
        num_weight_first,
        num_weight_hidden,
        num_weight_last,
        input_dim,
        width,
        omega_0
):
    w_init = tf.random.uniform((num_inputs, num_outputs),
                               -np.sqrt(6.0/num_inputs)*weight_factor,
                               np.sqrt(6.0/num_inputs)*weight_factor)

    scale_matrix = np.ones((num_outputs), dtype=np.float32)
    scale_matrix[:num_weight_first] /= input_dim  # 1st layer weights
    scale_matrix[num_weight_first:
                 num_weight_first + num_weight_hidden] *= np.sqrt(6.0/width)/omega_0  # hidden layer weights
    scale_matrix[num_weight_first + num_weight_hidden:
                 num_weight_first + num_weight_hidden + num_weight_last] *= np.sqrt(6.0/(width + width))  # last layer weights, since it is linear layer and no scaling,
    # we choose GlorotUniform
    scale_matrix[num_weight_first + num_weight_hidden + num_weight_last:] /= width  # all biases

    b_init = tf.random.uniform((num_outputs,), -scale_matrix, scale_matrix)
    return w_init, b_init


def compute_weight_index_for_shapenet(cfg_shape_net):
    si_dim = cfg_shape_net['input_dim']
    so_dim = cfg_shape_net['output_dim']
    n_sx = cfg_shape_net['units']
    l_sx = cfg_shape_net['nlayers']

    if cfg_shape_net['connectivity'] == 'full':
        num_weight_first = si_dim*n_sx
        if cfg_shape_net['use_resblock']:
            num_weight_hidden = (2*l_sx)*n_sx**2
        else:
            num_weight_hidden = l_sx*n_sx**2
    elif cfg_shape_net['connectivity'] == 'last':
        num_weight_first = 0
        num_weight_hidden = 0
    else:
        raise ValueError("check cfg_shape_net['connectivity'] value, it can only be 'last' or 'full'")
    num_weight_last = so_dim*n_sx
    return num_weight_first, num_weight_hidden, num_weight_last


class SIREN(tf.keras.layers.Layer):
    def __init__(self, num_inputs, num_outputs, layer_position,
                 omega_0=30., omega_0_e=30., cfg_shape_net=None,
                 mixed_policy=None):
        super(SIREN, self).__init__()
        self.omega_0 = omega_0
        self.layer_position = layer_position
        self.TF_DTYPE = tf.float16 if mixed_policy == 'mixed_float16' else tf.float32

        # initialize the weights
        if layer_position == 'first':
            w_init = tf.random.uniform((num_inputs, num_outputs),
                                       -1./num_inputs,
                                       1./num_inputs)
            b_init = tf.random.uniform((num_outputs,),
                                       -1./np.sqrt(num_inputs),
                                       1./np.sqrt(num_inputs))

        elif layer_position == 'hidden' or layer_position == 'bottleneck':
            w_init = tf.random.uniform((num_inputs, num_outputs),
                                       -tf.math.sqrt(
                                           6.0/num_inputs)/omega_0_e,
                                       tf.math.sqrt(
                                           6.0/num_inputs)/omega_0_e)
            b_init = tf.random.uniform((num_outputs,),
                                       -1./np.sqrt(num_inputs),
                                       1./np.sqrt(num_inputs))

        elif layer_position == 'last':
            if isinstance(cfg_shape_net, dict):
                raise ValueError(
                    "No value for dictionary: cfg_shape_net for {} SIREN layer {}".format(layer_position, self.name))

            # compute the indices needed for generating weights for shapenet
            num_weight_first, num_weight_hidden, num_weight_last = compute_weight_index_for_shapenet(cfg_shape_net)

            w_init, b_init = gen_hypernetwork_weights_bias_for_siren_shapenet(
                num_inputs=num_inputs,
                num_outputs=num_outputs,
                weight_factor=cfg_shape_net['weight_init_factor'],
                num_weight_first=num_weight_first,
                num_weight_hidden=num_weight_hidden,
                num_weight_last=num_weight_last,
                input_dim=cfg_shape_net['input_dim'],
                width=cfg_shape_net['units'],
                omega_0=self.omega_0
            )

        else:
            raise NotImplementedError("No implementation of layer_position for {}".format(layer_position))

        self.w_init = w_init
        self.b_init = b_init
        self.w = tf.Variable(self.w_init)
        self.b = tf.Variable(self.b_init)

    # def _compute_number_weights_for_hypernetwork(self, cfg_shape_net):
    #     si_dim = cfg_shape_net['input_dim']
    #     so_dim = cfg_shape_net['output_dim']
    #     n_sx = cfg_shape_net['units']
    #     l_sx = cfg_shape_net['nlayers']
    #
    #     num_weight_first = si_dim*n_sx
    #     num_weight_hidden = (2*self.l_sx)*self.n_sx**2 + self.so_dim*self.n_sx
    #     num_weight_hidden = self.l_sx*self.n_sx**2 + self.so_dim*self.n_sx
    #
    #     return num_weight_first, num_weight_hidden

    def call(self, inputs, training=None, mask=None):
        if self.layer_position == 'last' or self.layer_position == 'bottleneck':
            y = tf.matmul(inputs, tf.cast(self.w, self.TF_DTYPE)) + tf.cast(
                self.b, self.TF_DTYPE)
            # y = tf.matmul(inputs, self.w) + self.b
        else:
            y = tf.math.sin(
                self.omega_0*tf.matmul(inputs,tf.cast(self.w, self.TF_DTYPE)) + tf.cast(self.b, self.TF_DTYPE)
            )
            # y = tf.math.sin(self.omega_0 * tf.matmul(inputs, self.w) + self.b)
        return y


class SIREN_ResNet(SIREN):
    def __init__(self, num_inputs,
                 num_outputs,
                 omega_0=30.,
                 omega_0_e=30.,
                 mixed_policy=None):
        super(SIREN_ResNet, self).__init__(num_inputs, num_outputs,
                                           layer_position='hidden',
                                           omega_0=omega_0, omega_0_e=omega_0_e,
                                           mixed_policy=mixed_policy)
        self.w2 = tf.Variable(self.w_init)
        self.b2 = tf.Variable(self.b_init)

    def call(self, x, training=None, mask=None):
        h = tf.math.sin(
            self.omega_0*tf.matmul(x, tf.cast(self.w, self.TF_DTYPE)) + tf.cast(self.b, self.TF_DTYPE))
        return 0.5*(x + tf.math.sin(
            self.omega_0*tf.matmul(h, tf.cast(self.w2, self.TF_DTYPE)) + tf.cast(self.b2, self.TF_DTYPE)))


class HyperLinearForSIREN(tf.keras.layers.Layer):
    def __init__(self,
                 num_inputs,
                 num_outputs,
                 cfg_shape_net,
                 mixed_policy
                 ):
        super(HyperLinearForSIREN, self).__init__()
        self.TF_DTYPE = tf.float16 if mixed_policy == 'mixed_float16' else tf.float32

        # compute the indices needed for generating weights for shapenet
        num_weight_first, num_weight_hidden, num_weight_last = compute_weight_index_for_shapenet(cfg_shape_net)

        w_init, b_init = gen_hypernetwork_weights_bias_for_siren_shapenet(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            weight_factor=cfg_shape_net['weight_init_factor'],
            num_weight_first=num_weight_first,
            num_weight_hidden=num_weight_hidden,
            num_weight_last=num_weight_last,
            input_dim=cfg_shape_net['input_dim'],
            width=cfg_shape_net['units'],
            omega_0=cfg_shape_net['omega_0']
        )

        self.w = tf.Variable(w_init)
        self.b = tf.Variable(b_init)

    def call(self, x, **kwargs):
        y = tf.matmul(x, tf.cast(self.w, self.TF_DTYPE)) + tf.cast(self.b, self.TF_DTYPE)
        return y

