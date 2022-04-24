import functools
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.keras.optimizer_v2 import utils as optimizer_utils
from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import values as ds_values
from tensorflow.python.keras.utils import tf_utils

# import sys
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

@keras_export('keras.optimizers.L4Adam')
class L4Adam(tf.keras.optimizers.Optimizer):
    # _HAS_AGGREGATE_GRAD = True
    def __init__(self, learning_rate=0.15, tau_m=10., tau_s=1000., tau=1000., gamma_0=0.75,
                 gamma=0.9, epsilon=1e-7, name="L4Adam", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("learning_rate", learning_rate))
        self._set_hyper("tau_m", kwargs.get("tau_m", tau_m))
        self._set_hyper("tau_s", kwargs.get("tau_s", tau_s))
        self._set_hyper("tau", kwargs.get("tau", tau))
        self._set_hyper("gamma", kwargs.get("gamma", gamma))
        self.epsilon = epsilon or backend_config.epsilon()
        self.l_min = 1e6
        self.gamma_0 = kwargs.get("gamma_0", gamma_0)
        self._is_first = True

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'g')
        for var in var_list:
            self.add_slot(var, 'g2')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(L4Adam, self)._prepare_local(var_device, var_dtype, apply_state)

        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        lr = self._decayed_lr(var_dtype)
        tau_m = array_ops.identity(self._get_hyper('tau_m', var_dtype))
        tau_s = array_ops.identity(self._get_hyper('tau_s', var_dtype))
        tau = array_ops.identity(self._get_hyper('tau', var_dtype))
        gamma = array_ops.identity(self._get_hyper('gamma', var_dtype))

        # derived quantity
        one_over_tau_s = 1./tau_s
        one_over_tau_m = 1./tau_m
        normal_factor_m = 1.0 - math_ops.pow(1. - one_over_tau_m, local_step)
        normal_factor_s = 1.0 - math_ops.pow(1. - one_over_tau_s, local_step)

        apply_state[(var_device, var_dtype)].update(dict(
            lr=lr,
            epsilon=ops.convert_to_tensor(self.epsilon, var_dtype),
            tau_m=tau_m,
            tau_s=tau_s,
            normal_factor_m=normal_factor_m,
            normal_factor_s=normal_factor_s,
            one_over_tau_s=one_over_tau_s,
            one_over_tau_m=one_over_tau_m,
            gamma=gamma,
            one_plus_one_over_tau=1. + 1./tau,
            one_minus_one_over_tau_s=1. - one_over_tau_s,
            one_minus_one_over_tau_m=1. - one_over_tau_m
        ))

    def _momentum_add(self, m, x, one_minus_one_over_tau, one_over_tau, factor):
        return (one_minus_one_over_tau*m + one_over_tau*x)/factor

    def _resource_apply_dense(self, grad, var, apply_state=None, loss=None):
        """Update the slots and perform one optimization step for one model variable
        """
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        # t = math_ops.cast(self.iterations + 1, var_dtype)
        # tau_m = self._get_hyper('tau_m', var_dtype)
        # tau_s = self._get_hyper('tau_s', var_dtype)
        #
        # epsilon = self._get_hyper('epsilon', var_dtype)

        g = self.get_slot(var, "g")
        g2 = self.get_slot(var, "g2")
        g_new = self._momentum_add(g, grad,
                                   coefficients['one_minus_one_over_tau_m'],
                                   coefficients['one_over_tau_m'],
                                   coefficients['normal_factor_m'])
        g2_new = self._momentum_add(g2, grad*grad,
                                    coefficients['one_minus_one_over_tau_s'],
                                    coefficients['one_over_tau_s'],
                                    coefficients['normal_factor_s'])
        nu_new = g_new / (math_ops.sqrt(g2_new) + coefficients['epsilon'])

        new_var = None
        #TODO(shaowu: it seems that implementing L4 can be quite troublesome in tensorflow 2)..
        # mostly because of the g^T * nu term, maybe it is easier in pytorch
        # new_var = var - coefficients['lr'] * nu_new * (loss - coefficients["gamma"]*self.l_min)\
        #           /(coefficients["epsilon"] + )

        # create the ops for updating
        g_update = state_ops.assign(g, g_new, use_locking=self._use_locking)
        g2_update = state_ops.assign(g2, g2_new, use_locking=self._use_locking)
        var_update = state_ops.assign(var, new_var, use_locking=self._use_locking)

        return control_flow_ops.group(*[var_update, g_update, g2_update])

    def minimize(self, loss, var_list, grad_loss=None, name=None, tape=None):
        grads_and_vars = self._compute_gradients(
            loss, var_list=var_list, grad_loss=grad_loss, tape=tape)
        return self.apply_gradients(grads_and_vars, name=name, loss=loss)

    def apply_gradients(self,
                        grads_and_vars,
                        name=None,
                        experimental_aggregate_gradients=True,
                        loss=None):
        # update l_min
        if self.iterations == 0:
            self.l_min = ops.convert_to_tensor(self.gamma_0) * loss
        else:
            self.l_min = math_ops.minimum(self.l_min, loss)

        grads_and_vars = optimizer_utils.filter_empty_gradients(grads_and_vars)
        var_list = [v for (_, v) in grads_and_vars]

        with ops.name_scope_v2(self._name):
            # Create iteration if necessary.
            with ops.init_scope():
                self._create_all_weights(var_list)

            if not grads_and_vars:
                # Distribution strategy does not support reducing an empty list of
                # gradients
                return control_flow_ops.no_op()

            if distribute_ctx.in_cross_replica_context():
                raise RuntimeError(
                    "`apply_gradients() cannot be called in cross-replica context. "
                    "Use `tf.distribute.Strategy.run` to enter replica "
                    "context.")

            strategy = distribute_ctx.get_strategy()
            if (not experimental_aggregate_gradients and strategy and isinstance(
                    strategy.extended,
                    parameter_server_strategy.ParameterServerStrategyExtended)):
                raise NotImplementedError(
                    "`experimental_aggregate_gradients=False is not supported for "
                    "ParameterServerStrategy and CentralStorageStrategy")

            apply_state = self._prepare(var_list)
            if experimental_aggregate_gradients:
                grads_and_vars = self._transform_unaggregated_gradients(grads_and_vars)
                grads_and_vars = self._aggregate_gradients(grads_and_vars)
            grads_and_vars = self._transform_gradients(grads_and_vars)

            return distribute_ctx.get_replica_context().merge_call(
                functools.partial(self._distributed_apply, apply_state=apply_state, loss=loss),
                args=(grads_and_vars,),
                kwargs={
                    "name": name,
                })

    def _distributed_apply(self, distribution, grads_and_vars, name, apply_state, loss=None):
        """`apply_gradients` using a `DistributionStrategy`."""

        def apply_grad_to_update_var(var, grad):
            """Apply gradient to variable."""
            if isinstance(var, ops.Tensor):
                raise NotImplementedError("Trying to update a Tensor ", var)

            apply_kwargs = {}
            if isinstance(grad, ops.IndexedSlices):
                if var.constraint is not None:
                    raise RuntimeError(
                        "Cannot use a constraint function on a sparse variable.")
                if "apply_state" in self._sparse_apply_args:
                    apply_kwargs["apply_state"] = apply_state
                return self._resource_apply_sparse_duplicate_indices(
                    grad.values, var, grad.indices, **apply_kwargs)

            if "apply_state" in self._dense_apply_args:
                apply_kwargs["apply_state"] = apply_state
            apply_kwargs["loss"] = loss
            update_op = self._resource_apply_dense(grad, var, **apply_kwargs)
            if var.constraint is not None:
                with ops.control_dependencies([update_op]):
                    return var.assign(var.constraint(var))
            else:
                return update_op

        eagerly_outside_functions = ops.executing_eagerly_outside_functions()
        update_ops = []
        with ops.name_scope(name or self._name, skip_on_eager=True):
            for grad, var in grads_and_vars:
                # TODO(crccw): It's not allowed to assign PerReplica value to
                # MirroredVariable.  Remove this after we relax this restriction.
                def _assume_mirrored(grad):
                    if isinstance(grad, ds_values.PerReplica):
                        return ds_values.Mirrored(grad.values)
                    return grad

                grad = nest.map_structure(_assume_mirrored, grad)
                # Colocate the update with variables to avoid unnecessary communication
                # delays. See b/136304694.
                with distribution.extended.colocate_vars_with(var):
                    with ops.name_scope("update" if eagerly_outside_functions else
                                        "update_" + var.op.name, skip_on_eager=True):
                        update_ops.extend(distribution.extended.update(
                            var, apply_grad_to_update_var, args=(grad,), group=False))

            any_symbolic = any(isinstance(i, ops.Operation) or
                               tf_utils.is_symbolic_tensor(i) for i in update_ops)
            if not context.executing_eagerly() or any_symbolic:
                # If the current context is graph mode or any of the update ops are
                # symbolic then the step update should be carried out under a graph
                # context. (eager updates execute immediately)
                with ops._get_graph_from_inputs(update_ops).as_default():  # pylint: disable=protected-access
                    with ops.control_dependencies([control_flow_ops.group(update_ops)]):
                        return self._iterations.assign_add(1, read_value=False)

            return self._iterations.assign_add(1)


    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        raise NotImplementedError

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "beta_1": self._serialize_hyperparameter("beta_1"),
            "beta_2": self._serialize_hyperparameter("beta_2"),
            "epsilon": self._serialize_hyperparameter("epsilon"),
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