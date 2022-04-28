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
        omega_0,
        variable_dtype
):
    w_init = tf.random.uniform((num_inputs, num_outputs),
                               -np.sqrt(6.0/num_inputs)*weight_factor,
                               np.sqrt(6.0/num_inputs)*weight_factor)

    scale_matrix = np.ones((num_outputs), dtype=variable_dtype)
    scale_matrix[:num_weight_first] /= input_dim  # 1st layer weights
    scale_matrix[num_weight_first:
                 num_weight_first + num_weight_hidden] *= np.sqrt(6.0/width)/omega_0  # hidden layer weights
    scale_matrix[num_weight_first + num_weight_hidden:
                 num_weight_first + num_weight_hidden + num_weight_last] *= np.sqrt(6.0/(width + width))  # last layer weights, since it is linear layer and no scaling,
    # we choose GlorotUniform
    scale_matrix[num_weight_first + num_weight_hidden + num_weight_last:] /= width  # all biases

    b_init = tf.random.uniform((num_outputs,), -scale_matrix, scale_matrix, dtype=variable_dtype)
    return w_init, b_init

def compute_number_of_weightbias_by_its_position_for_shapenet(cfg_shape_net):
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
                 omega_0=30., cfg_shape_net=None,
                 mixed_policy=tf.keras.mixed_precision.Policy('float32')):
        super(SIREN, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.layer_position = layer_position
        self.omega_0 = tf.cast(omega_0, self.variable_Dtype)
        self.cfg_shape_net = cfg_shape_net
        self.mixed_policy = mixed_policy
        self.compute_Dtype = self.mixed_policy.compute_dtype
        self.variable_Dtype = self.mixed_policy.variable_dtype

        # initialize the weights
        if layer_position == 'first':
            w_init = tf.random.uniform((num_inputs, num_outputs),
                                       -1./num_inputs,
                                       1./num_inputs,
                                       dtype=self.variable_Dtype)
            b_init = tf.random.uniform((num_outputs,),
                                       -1./np.sqrt(num_inputs),
                                       1./np.sqrt(num_inputs),
                                       dtype=self.variable_Dtype)

        elif layer_position == 'hidden' or layer_position == 'bottleneck':
            w_init = tf.random.uniform((num_inputs, num_outputs),
                                       -tf.math.sqrt(6.0/num_inputs)/self.omega_0,
                                       tf.math.sqrt(6.0/num_inputs)/self.omega_0,
                                       dtype=self.variable_Dtype)
            b_init = tf.random.uniform((num_outputs,),
                                       -1./np.sqrt(num_inputs),
                                       1./np.sqrt(num_inputs),
                                       dtype=self.variable_Dtype)

        elif layer_position == 'last':
            if isinstance(cfg_shape_net, dict):
                raise ValueError(
                    "No value for dictionary: cfg_shape_net for {} SIREN layer {}".format(layer_position, self.name))

            # compute the indices needed for generating weights for shapenet
            # if connectivity_e == 'full':
            num_weight_first, num_weight_hidden, num_weight_last = compute_number_of_weightbias_by_its_position_for_shapenet(cfg_shape_net)
            # elif connectivity_e == 'last_layer':
            #     num_weight_first = 0
            #     num_weight_hidden = 0
            #     num_weight_last = num_outputs // cfg_shape_net['output_dim']  # po_dim*so_dim / so_dim = po_dim
            # else:
            #     raise ValueError("`connectivity_e` has an invalid value {}".format(connectivity_e))

            w_init, b_init = gen_hypernetwork_weights_bias_for_siren_shapenet(
                num_inputs=num_inputs,
                num_outputs=num_outputs,
                weight_factor=cfg_shape_net['weight_init_factor'],
                num_weight_first=num_weight_first,
                num_weight_hidden=num_weight_hidden,
                num_weight_last=num_weight_last,
                input_dim=cfg_shape_net['input_dim'],
                width=cfg_shape_net['units'],
                omega_0=self.omega_0,
                variable_dtype=self.variable_Dtype
            )

        else:
            raise NotImplementedError("No implementation of layer_position for {}".format(layer_position))

        self.w_init = w_init
        self.b_init = b_init
        self.w = tf.Variable(self.w_init, dtype=self.variable_Dtype)
        self.b = tf.Variable(self.b_init, dtype=self.variable_Dtype)

    def call(self, x, **kwargs):
        if self.layer_position == 'last' or self.layer_position == 'bottleneck':
            y = tf.matmul(x,  tf.cast(self.w, self.compute_Dtype)) + tf.cast(self.b, self.compute_Dtype)
        else:
            y = tf.math.sin(self.omega_0*tf.matmul(x, tf.cast(self.w, self.compute_Dtype)) +
                            tf.cast(self.b,self.compute_Dtype))
        return y

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "layer_position": self.layer_position,
            "omega_0": self.omega_0,
            "cfg_shape_net": self.cfg_shape_net,
            "mixed_policy": self.mixed_policy
        })
        return config

class SIREN_ResNet(SIREN):
    def __init__(self, num_inputs,
                 num_outputs,
                 omega_0=30.,
                 mixed_policy=tf.keras.mixed_precision.Policy('float32')):
        super(SIREN_ResNet, self).__init__(num_inputs, num_outputs,
                                           layer_position='hidden',
                                           omega_0=omega_0,
                                           mixed_policy=mixed_policy)
        self.w2 = tf.Variable(self.w_init, dtype=self.variable_Dtype)
        self.b2 = tf.Variable(self.b_init, dtype=self.variable_Dtype)

    def call(self, x, training=None, mask=None):
        h = tf.math.sin(self.omega_0*tf.matmul(x, tf.cast(self.w, self.compute_Dtype)) +
                        tf.cast(self.b,self.compute_Dtype))
        return 0.5*(x + tf.math.sin(self.omega_0*tf.matmul(h, tf.cast(self.w2, self.compute_Dtype)) +
                                            tf.cast(self.b2, self.compute_Dtype)))


class HyperLinearForSIREN(tf.keras.layers.Layer):
    def __init__(self, num_inputs, num_outputs, cfg_shape_net, mixed_policy, connectivity='full'):
        super(HyperLinearForSIREN, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.mixed_policy = mixed_policy
        self.compute_Dtype = self.mixed_policy.compute_dtype
        self.variable_Dtype = self.mixed_policy.variable_dtype
        self.connectivity = connectivity

        # compute the indices needed for generating weights for shapenet
        if connectivity == 'full':
            num_weight_first, num_weight_hidden, num_weight_last = compute_number_of_weightbias_by_its_position_for_shapenet(cfg_shape_net)
        elif connectivity == 'last_layer':
            num_weight_first, num_weight_hidden, num_weight_last = 0, 0, num_outputs
        else:
            raise ValueError("connectivity should be set to `full` or `last_layer`")

        w_init, b_init = gen_hypernetwork_weights_bias_for_siren_shapenet(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            weight_factor=cfg_shape_net['weight_init_factor'],
            num_weight_first=num_weight_first,
            num_weight_hidden=num_weight_hidden,
            num_weight_last=num_weight_last,
            input_dim=cfg_shape_net['input_dim'],
            width=cfg_shape_net['units'],
            omega_0=cfg_shape_net['omega_0'],
            variable_dtype=self.variable_Dtype
        )

        self.w = tf.Variable(w_init, self.variable_Dtype)
        self.b = tf.Variable(b_init, self.variable_Dtype)

    def call(self, x, **kwargs):
        y = tf.matmul(x, tf.cast(self.w, self.compute_Dtype)) + tf.cast(self.b, self.compute_Dtype)
        return y

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "cfg_shape_net": self.cfg_shape_net,
            "mixed_policy": self.mixed_policy,
            "connectivity": self.connectivity
        })
        return config
