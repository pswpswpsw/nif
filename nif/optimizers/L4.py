import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import gen_training_ops
from tensorflow.python.util.tf_export import keras_export

#
# def n_inner_product(list_of_tensors1, list_of_tensors2):
#     return tf.add_n([tf.reduce_sum(t1*t2) for t1, t2 in zip(list_of_tensors1, list_of_tensors2)])
#
# def time_factor(time_step):
#     """ Routine used for bias correction in exponential moving averages, as in (Kingma, Ba, 2015) """
#     global_step = 1 + tf.train.get_or_create_global_step()
#     decay = 1.0 - 1.0 / time_step
#     return 1.0 - tf.exp((tf.cast(global_step, tf.float32)) * tf.log(decay))
#
# class AdamTransform(object):
#     """
#     Class implementing Adam (Kingma, Ba 2015) transform of the gradient.
#     """
#     def __init__(self, time_scale_grad=10.0, time_scale_var=1000.0, epsilon=1e-4):
#         self.time_scale_grad = time_scale_grad
#         self.time_scale_var = time_scale_var
#         self.epsilon = epsilon
#
#         self.EMAgrad = tf.train.ExponentialMovingAverage(decay=1.0 - 1.0 / self.time_scale_grad)
#         self.EMAvar = tf.train.ExponentialMovingAverage(decay=1.0 - 1.0 / self.time_scale_var)
#
#     def __call__(self, grads):
#         shadow_op_gr = self.EMAgrad.apply(grads)
#         vars = [tf.square(grad) for grad in grads]
#         shadow_op_var = self.EMAvar.apply(vars)
#
#         with tf.control_dependencies([shadow_op_gr, shadow_op_var]):
#             correction_term_1 = time_factor(self.time_scale_grad)
#             avg_grads = [self.EMAgrad.average(grad) / correction_term_1 for grad in grads]
#
#             correction_term_2 = time_factor(self.time_scale_var)
#             avg_vars = [self.EMAvar.average(var) / correction_term_2 for var in vars]
#             return [(grad / (tf.sqrt(var) + self.epsilon)) for grad, var in zip(avg_grads, avg_vars)]
#
#
# class MomentumTransform(object):
#     """
#     Class implementing momentum transform of the gradient (here in the form of exponential moving average)
#     """
#     def __init__(self, time_momentum=10.0):
#         self.time_momentum = time_momentum
#         self.EMAgrad = tf.train.ExponentialMovingAverage(decay=1.0-1.0/self.time_momentum)
#
#     def __call__(self, grads):
#         shadow_op_gr = self.EMAgrad.apply(grads)
#         with tf.control_dependencies([shadow_op_gr]):
#             correction_term = time_factor(self.time_momentum)
#             new_grads = [self.EMAgrad.average(grad) / correction_term for grad in grads]
#             return [tf.identity(grad) for grad in new_grads]
#
#
# string_to_transform = {'momentum': MomentumTransform,
#                        'adam': AdamTransform}
#
#
# class L4General(tf.train.GradientDescentOptimizer):
#     def __init__(self, fraction=0.15, minloss_factor=0.9, init_factor=0.75,
#                  minloss_forget_time=1000.0, epsilon=1e-12,
#                  gradient_estimator='momentum', gradient_params=None,
#                  direction_estimator='adam', direction_params=None):
#         tf.train.GradientDescentOptimizer.__init__(self, 1.0)
#         with tf.variable_scope('L4Optimizer', reuse=tf.AUTO_REUSE):
#             self.min_loss = tf.get_variable(name='min_loss', shape=(),
#                                             initializer=tf.constant_initializer(0.0), trainable=False)
#         self.fraction = fraction
#         self.minloss_factor = minloss_factor
#         self.minloss_increase_rate = 1.0 + 1.0 / minloss_forget_time
#         self.epsilon = epsilon
#         self.init_factor = init_factor
#
#         if not direction_params:
#             direction_params = {}
#         if not gradient_params:
#             gradient_params = {}
#
#         self.grad_direction = string_to_transform[direction_estimator](**direction_params)
#         self.deriv_estimate = string_to_transform[gradient_estimator](**gradient_params)
#
#     def compute_gradients(self, loss, *args, **kwargs):
#         self.loss = loss
#         return super(L4General, self).compute_gradients(loss, *args, **kwargs)
#
#     def apply_gradients(self, grads_and_vars, global_step=None, name=None):
#         if not global_step:
#             global_step = tf.train.get_or_create_global_step()
#         # Filter variables without a gradient.
#         grads_and_vars = [(grad, var) for grad, var in grads_and_vars if grad is not None]
#
#         grads, vars = zip(*grads_and_vars)
#
#         ml_newval = tf.cond(tf.equal(global_step, 0), lambda: self.init_factor*self.loss,
#                             lambda: tf.minimum(self.min_loss, self.loss))
#         ml_update = self.min_loss.assign(ml_newval)
#
#         with tf.control_dependencies([ml_update]):
#             directions = self.grad_direction(grads)
#             derivatives = self.deriv_estimate(grads)
#
#             min_loss_to_use = self.minloss_factor * self.min_loss
#             l_rate = self.fraction*(self.loss - min_loss_to_use) / (n_inner_product(directions, derivatives)+self.epsilon)
#             new_grads = [direction*l_rate for direction in directions]
#             tf.summary.scalar('effective_learning_rate', l_rate)
#             tf.summary.scalar('min_loss_estimate', self.min_loss)
#             ml_update2 = self.min_loss.assign(self.minloss_increase_rate * self.min_loss)
#
#             with tf.control_dependencies([ml_update2]):
#                 return tf.train.GradientDescentOptimizer.apply_gradients(self, zip(new_grads, vars), global_step, name)
#
#     def minimize(self, loss, global_step=None, var_list=None, name=None):
#         if not var_list:
#             var_list = tf.trainable_variables()
#
#         grads_and_vars = self.compute_gradients(loss, var_list)
#         return self.apply_gradients(grads_and_vars, global_step, name)
#
#
# class L4Adam(L4General):
#     """
#     Specialization of the L4 stepsize adaptation with Adam used for gradient updates and Mom for gradient estimation.
#     """
#     def __init__(self, fraction=0.15, minloss_factor=0.9, init_factor=0.75, minloss_forget_time=1000.0,
#                  epsilon=1e-12, adam_params=None):
#         L4General.__init__(self, fraction, minloss_factor, init_factor, minloss_forget_time,
#                            epsilon, gradient_estimator='momentum', direction_estimator='adam',
#                            direction_params=adam_params)
#
#
# class L4Mom(L4General):
#     """
#     Specialization of the L4 stepsize adaptation with Mom used for both gradient estimation and an update direction.
#     """
#     def __init__(self, fraction=0.15, minloss_factor=0.9, init_factor=0.75, minloss_forget_time=1000.0,
#                  epsilon=1e-12, mom_params=None):
#         L4General.__init__(self, fraction, minloss_factor, init_factor, minloss_forget_time,
#                            epsilon, gradient_estimator='momentum', direction_estimator='momentum',
#                            direction_params=mom_params)
#
#
#

#
# class Adam(optimizer_v2.OptimizerV2):
#     _HAS_AGGREGATE_GRAD = True
#     def __init__(self,
#                  learning_rate=0.001,
#                  beta_1=0.9,
#                  beta_2=0.999,
#                  epsilon=1e-7,
#                  amsgrad=False,
#                  name='Adam',
#                  **kwargs):
#         super(Adam, self).__init__(name, **kwargs)
#         self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
#         self._set_hyper('decay', self._initial_decay)
#         self._set_hyper('beta_1', beta_1)
#         self._set_hyper('beta_2', beta_2)
#         self.epsilon = epsilon or backend_config.epsilon()
#         self.amsgrad = amsgrad
#
#     def _create_slots(self, var_list):
#         # Create slots for the first and second moments.
#         # Separate for-loops to respect the ordering of slot variables from v1.
#         for var in var_list:
#             self.add_slot(var, 'm')
#         for var in var_list:
#             self.add_slot(var, 'v')
#         if self.amsgrad:
#             for var in var_list:
#                 self.add_slot(var, 'vhat')
#
#     def _prepare_local(self, var_device, var_dtype, apply_state):
#         super(Adam, self)._prepare_local(var_device, var_dtype, apply_state)
#
#         local_step = math_ops.cast(self.iterations + 1, var_dtype)
#         beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
#         beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
#         beta_1_power = math_ops.pow(beta_1_t, local_step)
#         beta_2_power = math_ops.pow(beta_2_t, local_step)
#         lr = (apply_state[(var_device, var_dtype)]['lr_t'] *
#               (math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)))
#         apply_state[(var_device, var_dtype)].update(
#             dict(
#                 lr=lr,
#                 epsilon=ops.convert_to_tensor_v2_with_dispatch(
#                     self.epsilon, var_dtype),
#                 beta_1_t=beta_1_t,
#                 beta_1_power=beta_1_power,
#                 one_minus_beta_1_t=1 - beta_1_t,
#                 beta_2_t=beta_2_t,
#                 beta_2_power=beta_2_power,
#                 one_minus_beta_2_t=1 - beta_2_t))
#
#     def set_weights(self, weights):
#         params = self.weights
#         # If the weights are generated by Keras V1 optimizer, it includes vhats
#         # even without amsgrad, i.e, V1 optimizer has 3x + 1 variables, while V2
#         # optimizer has 2x + 1 variables. Filter vhats out for compatibility.
#         num_vars = int((len(params) - 1) / 2)
#         if len(weights) == 3 * num_vars + 1:
#             weights = weights[:len(params)]
#         super(Adam, self).set_weights(weights)
#
#     def _resource_apply_dense(self, grad, var, apply_state=None):
#         var_device, var_dtype = var.device, var.dtype.base_dtype
#         coefficients = ((apply_state or {}).get((var_device, var_dtype))
#                         or self._fallback_apply_state(var_device, var_dtype))
#
#         m = self.get_slot(var, 'm')
#         v = self.get_slot(var, 'v')
#
#         return gen_training_ops.ResourceApplyAdam(
#             var=var.handle,
#             m=m.handle,
#             v=v.handle,
#             beta1_power=coefficients['beta_1_power'],
#             beta2_power=coefficients['beta_2_power'],
#             lr=coefficients['lr_t'],
#             beta1=coefficients['beta_1_t'],
#             beta2=coefficients['beta_2_t'],
#             epsilon=coefficients['epsilon'],
#             grad=grad,
#             use_locking=self._use_locking)
#
#     def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
#         raise NotImplementedError("Sparse update not implemented.")
#
#     def get_config(self):
#         config = super(Adam, self).get_config()
#         config.update({
#             'learning_rate': self._serialize_hyperparameter('learning_rate'),
#             'decay': self._serialize_hyperparameter('decay'),
#             'beta_1': self._serialize_hyperparameter('beta_1'),
#             'beta_2': self._serialize_hyperparameter('beta_2'),
#             'epsilon': self.epsilon,
#             'amsgrad': self.amsgrad,
#         })
#         return config


class SGOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.01, name="SGOptimizer", **kwargs):
        """Call super().__init__() and use _set_hyper() to store hyperparameters"""
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate)) # handle lr=learning_rate
        self._is_first = True

    def _create_slots(self, var_list):
        """For each model variable, create the optimizer variable associated with it.
        TensorFlow calls these optimizer variables "slots".
        For momentum optimization, we need one momentum slot per model variable.
        """
        for var in var_list:
            self.add_slot(var, "pv") #previous variable i.e. weight or bias
        for var in var_list:
            self.add_slot(var, "pg") #previous gradient

    @tf.function
    def _resource_apply_dense(self, grad, var):
        """Update the slots and perform one optimization step for one model variable
        """
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype) # handle learning rate decay
        new_var_m = var - grad * lr_t
        pv_var = self.get_slot(var, "pv")
        pg_var = self.get_slot(var, "pg")

        if self._is_first:
            self._is_first = False
            new_var = new_var_m
        else:
            cond = grad*pg_var >= 0
            print(cond)
            avg_weights = (pv_var + var)/2.0
            new_var = tf.where(cond, new_var_m, avg_weights)
        pv_var.assign(var)
        pg_var.assign(grad)
        var.assign(new_var)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        raise NotImplementedError

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
        }


#     def _resource_apply_sparse(self, grad, var):
#         raise NotImplementedError
#
#     def get_config(self):
#         base_config = super().get_config()
#         return {
#             **base_config,
#             "learning_rate": self._serialize_hyperparameter("learning_rate"),
#             "decay": self._serialize_hyperparameter("decay"),
#             "momentum": self._serialize_hyperparameter("momentum"),
#         }
# #
# class Fromage(optimizer_v2.OptimizerV2):
#
#     _HAS_AGGREGATE_GRAD = True
#
#     def __init__(self,
#                  learning_rate=0.01,
#                  name="Fromage"):
#
#         super(Fromage, self).__init__(name)
#         self._set_hyper("learning_rate", learning_rate)
#         self._set_hyper("decay", self._initial_decay)
#
#     def _resource_apply_dense(self, grad, var, apply_state=None):
#         var_device, var_dtype = var.device, var.dtype.base_dtype
#         coefficients = ((apply_state or {}).get((var_device, var_dtype))
#                         or self._fallback_apply_state(var_device, var_dtype))
#
#         var_t = (var - coefficients['lr_t']*grad/tf.norm(grad)*tf.norm(var) ) / tf.sqrt(1 + coefficients['lr_t']**2)
#         return state_ops.assign(var, var_t, use_locking=self._use_locking)
#
#     def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
#         raise NotImplementedError("Sparse update not implemented.")
#
#     def get_config(self):
#         config = super(Fromage, self).get_config()
#         config.update({
#             "learning_rate": self._serialize_hyperparameter("learning_rate")
#         })
#         return config