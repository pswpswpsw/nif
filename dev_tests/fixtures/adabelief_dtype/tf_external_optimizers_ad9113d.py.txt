import functools

import tensorflow as tf
from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import values as ds_values
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import utils as optimizer_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import keras_export


# from tensorflow.python.eager import backprop
# from tensorflow.python.framework import dtypes


@keras_export("keras.optimizers.L4Adam")
class L4Adam(tf.keras.optimizers.Optimizer):
    """Implements the L4Adam optimizer.

    This optimizer is an implementation of the L4 optimization algorithm with
    an adaptive learning rate that is based on the Adam optimizer.

    Attributes:
        learning_rate: A float, the initial learning rate.
        tau_m: A float, decay rate for first moment estimates.
        tau_s: A float, decay rate for second moment estimates.
        tau: A float, decay rate for the l_min estimate.
        gamma_0: A float, initial proportion of the loss to be considered as l_min.
        gamma: A float, parameter to control the proportion of l_min in the update.
        epsilon: A float, small constant for numerical stability.
        name: Optional string, the name for the optimizer.

    Methods:
        _create_slots: Creates slots for the optimizer's state.
        _prepare_local: Prepares the local hyperparameters and derived quantities.
        _momentum_add: Computes the momentum addition for a given variable.
        _resource_apply_dense: Applies the dense gradients to the model variables.
        minimize: Minimizes the loss function for the given model variables.
        apply_gradients: Applies the gradients to the model variables.
        _distributed_apply: Applies the gradients in a distributed setting.
        _resource_apply_sparse: NotImplemented, raises NotImplementedError.
        get_config: Returns the config dictionary for the optimizer instance.
    """

    def __init__(
        self,
        learning_rate=0.15,
        tau_m=10.0,
        tau_s=1000.0,
        tau=1000.0,
        gamma_0=0.75,
        gamma=0.9,
        epsilon=1e-7,
        name="L4Adam",
        **kwargs
    ):
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
            self.add_slot(var, "g")
        for var in var_list:
            self.add_slot(var, "g2")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(L4Adam, self)._prepare_local(var_device, var_dtype, apply_state)

        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        lr = self._decayed_lr(var_dtype)
        tau_m = array_ops.identity(self._get_hyper("tau_m", var_dtype))
        tau_s = array_ops.identity(self._get_hyper("tau_s", var_dtype))
        tau = array_ops.identity(self._get_hyper("tau", var_dtype))
        gamma = array_ops.identity(self._get_hyper("gamma", var_dtype))

        # derived quantity
        one_over_tau_s = 1.0 / tau_s
        one_over_tau_m = 1.0 / tau_m
        normal_factor_m = 1.0 - math_ops.pow(1.0 - one_over_tau_m, local_step)
        normal_factor_s = 1.0 - math_ops.pow(1.0 - one_over_tau_s, local_step)

        apply_state[(var_device, var_dtype)].update(
            dict(
                lr=lr,
                epsilon=ops.convert_to_tensor(self.epsilon, var_dtype),
                tau_m=tau_m,
                tau_s=tau_s,
                normal_factor_m=normal_factor_m,
                normal_factor_s=normal_factor_s,
                one_over_tau_s=one_over_tau_s,
                one_over_tau_m=one_over_tau_m,
                gamma=gamma,
                one_plus_one_over_tau=1.0 + 1.0 / tau,
                one_minus_one_over_tau_s=1.0 - one_over_tau_s,
                one_minus_one_over_tau_m=1.0 - one_over_tau_m,
            )
        )

    def _momentum_add(self, m, x, one_minus_one_over_tau, one_over_tau, factor):
        return (one_minus_one_over_tau * m + one_over_tau * x) / factor

    def _resource_apply_dense(self, grad, var, apply_state=None, loss=None):
        """Update the slots and perform one optimization step for one model variable"""
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        # t = math_ops.cast(self.iterations + 1, var_dtype)
        # tau_m = self._get_hyper('tau_m', var_dtype)
        # tau_s = self._get_hyper('tau_s', var_dtype)
        #
        # epsilon = self._get_hyper('epsilon', var_dtype)

        g = self.get_slot(var, "g")
        g2 = self.get_slot(var, "g2")
        g_new = self._momentum_add(
            g,
            grad,
            coefficients["one_minus_one_over_tau_m"],
            coefficients["one_over_tau_m"],
            coefficients["normal_factor_m"],
        )
        g2_new = self._momentum_add(
            g2,
            grad * grad,
            coefficients["one_minus_one_over_tau_s"],
            coefficients["one_over_tau_s"],
            coefficients["normal_factor_s"],
        )
        # nu_new = g_new / (math_ops.sqrt(g2_new) + coefficients["epsilon"])

        new_var = None
        # TODO(shaowu: it seems that implementing L4 can be quite troublesome in tensorflow 2)..
        # mostly because of the g^T * nu term, maybe it is easier in pytorch
        # new_var = var - coefficients['lr'] * nu_new * (loss - coefficients["gamma"]*self.l_min)\
        #           /(coefficients["epsilon"] + )
        # indeed, I can implement it in customized training loop. but anyway, it maybe not worth it.

        # create the ops for updating
        g_update = state_ops.assign(g, g_new, use_locking=self._use_locking)
        g2_update = state_ops.assign(g2, g2_new, use_locking=self._use_locking)
        var_update = state_ops.assign(var, new_var, use_locking=self._use_locking)

        return control_flow_ops.group(*[var_update, g_update, g2_update])

    def minimize(self, loss, var_list, grad_loss=None, name=None, tape=None):
        grads_and_vars = self._compute_gradients(
            loss, var_list=var_list, grad_loss=grad_loss, tape=tape
        )
        return self.apply_gradients(grads_and_vars, name=name, loss=loss)

    def apply_gradients(
        self,
        grads_and_vars,
        name=None,
        experimental_aggregate_gradients=True,
        loss=None,
    ):
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
                    "context."
                )

            strategy = distribute_ctx.get_strategy()
            if (
                not experimental_aggregate_gradients
                and strategy
                and isinstance(
                    strategy.extended,
                    parameter_server_strategy.ParameterServerStrategyExtended,
                )
            ):
                raise NotImplementedError(
                    "`experimental_aggregate_gradients=False is not supported for "
                    "ParameterServerStrategy and CentralStorageStrategy"
                )

            apply_state = self._prepare(var_list)
            if experimental_aggregate_gradients:
                grads_and_vars = self._transform_unaggregated_gradients(grads_and_vars)
                grads_and_vars = self._aggregate_gradients(grads_and_vars)
            grads_and_vars = self._transform_gradients(grads_and_vars)

            return distribute_ctx.get_replica_context().merge_call(
                functools.partial(
                    self._distributed_apply, apply_state=apply_state, loss=loss
                ),
                args=(grads_and_vars,),
                kwargs={
                    "name": name,
                },
            )

    def _distributed_apply(
        self, distribution, grads_and_vars, name, apply_state, loss=None
    ):
        """`apply_gradients` using a `DistributionStrategy`."""

        def apply_grad_to_update_var(var, grad):
            """Apply gradient to variable."""
            if isinstance(var, ops.Tensor):
                raise NotImplementedError("Trying to update a Tensor ", var)

            apply_kwargs = {}
            if isinstance(grad, ops.IndexedSlices):
                if var.constraint is not None:
                    raise RuntimeError(
                        "Cannot use a constraint function on a sparse variable."
                    )
                if "apply_state" in self._sparse_apply_args:
                    apply_kwargs["apply_state"] = apply_state
                return self._resource_apply_sparse_duplicate_indices(
                    grad.values, var, grad.indices, **apply_kwargs
                )

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
                    with ops.name_scope(
                        "update"
                        if eagerly_outside_functions
                        else "update_" + var.op.name,
                        skip_on_eager=True,
                    ):
                        update_ops.extend(
                            distribution.extended.update(
                                var, apply_grad_to_update_var, args=(grad,), group=False
                            )
                        )

            any_symbolic = any(
                isinstance(i, ops.Operation) or tf_utils.is_symbolic_tensor(i)
                for i in update_ops
            )
            if not context.executing_eagerly() or any_symbolic:
                # If the current context is graph mode or any of the update ops are
                # symbolic then the step update should be carried out under a graph
                # context. (eager updates execute immediately)
                with ops._get_graph_from_inputs(
                    update_ops
                ).as_default():  # pylint: disable=protected-access
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


@keras_export("keras.optimizers.AdaBelief")
class AdaBeliefOptimizer(tf.keras.optimizers.Optimizer):
    """
    It implements the AdaBeliefOptimizer proposed by
    Juntang Zhuang et al. in [AdaBelief Optimizer: Adapting stepsizes by the belief
    in observed gradients](https://arxiv.org/abs/2010.07468).
    Contributor(s):
        Jerry Yu [cryu854] <cryu854@gmail.com>
    Example of usage:
    ```python
    from adabelief_tf import AdaBeliefOptimizer
    opt = AdaBeliefOptimizer(lr=1e-3)
    ```
    Note: `amsgrad` is not described in the original paper. Use it with
          caution.
    AdaBeliefOptimizer is not a placement of the heuristic warmup, the settings should be
    kept if warmup has already been employed and tuned in the baseline method.
    You can enable warmup by setting `total_steps` and `warmup_proportion`:
    ```python
    opt = AdaBeliefOptimizer(
        lr=1e-3,
        total_steps=10000,
        warmup_proportion=0.1,
        min_lr=1e-5,
    )
    ```
    In the above example, the learning rate will increase linearly
    from 0 to `lr` in 1000 steps, then decrease linearly from `lr` to `min_lr`
    in 9000 steps.
    Lookahead, proposed by Michael R. Zhang et.al in the paper
    [Lookahead Optimizer: k steps forward, 1 step back]
    (https://arxiv.org/abs/1907.08610v1), can be integrated with AdaBeliefOptimizer,
    which is announced by Less Wright and the new combined optimizer can also
    be called "Ranger". The mechanism can be enabled by using the lookahead
    wrapper. For example:
    ```python
    adabelief = AdaBeliefOptimizer()
    ranger = tfa.optimizers.Lookahead(adabelief, sync_period=6, slow_step_size=0.5)
    ```
    Example of serialization:
    ```python
    optimizer = AdaBeliefOptimizer(learning_rate=lr_scheduler, weight_decay=wd_scheduler)
    config = tf.keras.optimizers.serialize(optimizer)
    new_optimizer = tf.keras.optimizers.deserialize(config,
    custom_objects={"AdaBeliefOptimizer": AdaBeliefOptimizer})
    ```
            Args:
            learning_rate: A `Tensor` or a floating point value, or a schedule
                that is a `tf.keras.optimizers.schedules.LearningRateSchedule`.
                The learning rate.
            beta_1: A float value or a constant float tensor.
                The exponential decay rate for the 1st moment estimates.
            beta_2: A float value or a constant float tensor.
                The exponential decay rate for the 2nd moment estimates.
            epsilon: A small constant for numerical stability.
            weight_decay: A `Tensor` or a floating point value, or a schedule
                that is a `tf.keras.optimizers.schedules.LearningRateSchedule`.
                Weight decay for each parameter.
            rectify: boolean. Whether to enable rectification as in RectifiedAdam
            amsgrad: boolean. Whether to apply AMSGrad variant of this
                algorithm from the paper "On the Convergence of Adam and
                beyond".
            sma_threshold. A float value.
                The threshold for simple mean average.
            total_steps: An integer. Total number of training steps.
                Enable warmup by setting a positive value.
            warmup_proportion: A floating point value.
                The proportion of increasing steps.
            min_lr: A floating point value. Minimum learning rate after warmup.
            name: Optional name for the operations created when applying
                gradients. Defaults to "AdaBeliefOptimizer".
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients
                by norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse
                decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
    """

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-14,
        weight_decay=0.0,
        rectify=True,
        amsgrad=False,
        sma_threshold=5.0,
        total_steps=0,
        warmup_proportion=0.1,
        min_lr=0.0,
        name="AdaBeliefOptimizer",
        print_change_log=True,
        **kwargs
    ):
        super().__init__(name, **kwargs)

        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("weight_decay", weight_decay)
        self._set_hyper("sma_threshold", sma_threshold)
        self._set_hyper("total_steps", int(total_steps))
        self._set_hyper("warmup_proportion", warmup_proportion)
        self._set_hyper("min_lr", min_lr)
        self.epsilon = epsilon or tf.keras.backend.epsilon()
        self.amsgrad = amsgrad
        self.rectify = rectify
        self._has_weight_decay = weight_decay != 0.0
        self._initial_total_steps = total_steps

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")
        for var in var_list:
            self.add_slot(var, "v")
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, "vhat")

    def set_weights(self, weights):
        params = self.weights
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[: len(params)]
        super().set_weights(weights)

    def _decayed_wd(self, var_dtype):
        wd_t = self._get_hyper("weight_decay", var_dtype)
        if isinstance(wd_t, tf.keras.optimizers.schedules.LearningRateSchedule):
            wd_t = tf.cast(wd_t(self.iterations), var_dtype)
        return wd_t

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        wd_t = self._decayed_wd(var_dtype)
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        beta_1_t = self._get_hyper("beta_1", var_dtype)
        beta_2_t = self._get_hyper("beta_2", var_dtype)
        epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_power = tf.math.pow(beta_1_t, local_step)
        beta_2_power = tf.math.pow(beta_2_t, local_step)

        if self._initial_total_steps > 0:
            total_steps = self._get_hyper("total_steps", var_dtype)
            warmup_steps = total_steps * self._get_hyper("warmup_proportion", var_dtype)
            min_lr = self._get_hyper("min_lr", var_dtype)
            decay_steps = tf.maximum(total_steps - warmup_steps, 1)
            decay_rate = (min_lr - lr_t) / decay_steps
            lr_t = tf.where(
                local_step <= warmup_steps,
                lr_t * (local_step / warmup_steps),
                lr_t + decay_rate * tf.minimum(local_step - warmup_steps, decay_steps),
            )

        sma_inf = 2.0 / (1.0 - beta_2_t) - 1.0
        sma_t = sma_inf - 2.0 * local_step * beta_2_power / (1.0 - beta_2_power)

        m_t = m.assign(
            beta_1_t * m + (1.0 - beta_1_t) * grad, use_locking=self._use_locking
        )
        m_corr_t = m_t / (1.0 - beta_1_power)

        v_t = v.assign(
            beta_2_t * v + (1.0 - beta_2_t) * tf.math.square(grad - m_t) + epsilon_t,
            use_locking=self._use_locking,
        )

        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat_t = vhat.assign(tf.maximum(vhat, v_t), use_locking=self._use_locking)
            v_corr_t = tf.math.sqrt(vhat_t / (1.0 - beta_2_power))
        else:
            vhat_t = None
            v_corr_t = tf.math.sqrt(v_t / (1.0 - beta_2_power))

        r_t = tf.math.sqrt(
            (sma_t - 4.0)
            / (sma_inf - 4.0)
            * (sma_t - 2.0)
            / (sma_inf - 2.0)
            * sma_inf
            / sma_t
        )

        if self.rectify:
            sma_threshold = self._get_hyper("sma_threshold", var_dtype)
            var_t = tf.where(
                sma_t >= sma_threshold,
                r_t * m_corr_t / (v_corr_t + epsilon_t),
                m_corr_t,
            )
        else:
            var_t = m_corr_t / (v_corr_t + epsilon_t)

        if self._has_weight_decay:
            var_t += wd_t * var

        var_update = var.assign_sub(lr_t * var_t, use_locking=self._use_locking)

        updates = [var_update, m_t, v_t]
        if self.amsgrad:
            updates.append(vhat_t)
        return tf.group(*updates)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        wd_t = self._decayed_wd(var_dtype)
        beta_1_t = self._get_hyper("beta_1", var_dtype)
        beta_2_t = self._get_hyper("beta_2", var_dtype)
        epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_power = tf.math.pow(beta_1_t, local_step)
        beta_2_power = tf.math.pow(beta_2_t, local_step)

        if self._initial_total_steps > 0:
            total_steps = self._get_hyper("total_steps", var_dtype)
            warmup_steps = total_steps * self._get_hyper("warmup_proportion", var_dtype)
            min_lr = self._get_hyper("min_lr", var_dtype)
            decay_steps = tf.maximum(total_steps - warmup_steps, 1)
            decay_rate = (min_lr - lr_t) / decay_steps
            lr_t = tf.where(
                local_step <= warmup_steps,
                lr_t * (local_step / warmup_steps),
                lr_t + decay_rate * tf.minimum(local_step - warmup_steps, decay_steps),
            )

        sma_inf = 2.0 / (1.0 - beta_2_t) - 1.0
        sma_t = sma_inf - 2.0 * local_step * beta_2_power / (1.0 - beta_2_power)

        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta_1_t)
        m_t = m.assign(m * beta_1_t, use_locking=self._use_locking)
        m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)
        m_corr_t = m_t / (1.0 - beta_1_power)

        v = self.get_slot(var, "v")
        m_t_indices = tf.gather(m_t, indices)
        v_scaled_g_values = tf.math.square(grad - m_t_indices) * (1 - beta_2_t)
        v_t = v.assign(v * beta_2_t + epsilon_t, use_locking=self._use_locking)
        v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat_t = vhat.assign(tf.maximum(vhat, v_t), use_locking=self._use_locking)
            v_corr_t = tf.math.sqrt(vhat_t / (1.0 - beta_2_power))
        else:
            vhat_t = None
            v_corr_t = tf.math.sqrt(v_t / (1.0 - beta_2_power))

        r_t = tf.math.sqrt(
            (sma_t - 4.0)
            / (sma_inf - 4.0)
            * (sma_t - 2.0)
            / (sma_inf - 2.0)
            * sma_inf
            / sma_t
        )

        if self.rectify:
            sma_threshold = self._get_hyper("sma_threshold", var_dtype)
            var_t = tf.where(
                sma_t >= sma_threshold,
                r_t * m_corr_t / (v_corr_t + epsilon_t),
                m_corr_t,
            )
        else:
            var_t = m_corr_t / (v_corr_t + epsilon_t)

        if self._has_weight_decay:
            var_t += wd_t * var

        var_update = self._resource_scatter_add(
            var, indices, tf.gather(-lr_t * var_t, indices)
        )

        updates = [var_update, m_t, v_t]
        if self.amsgrad:
            updates.append(vhat_t)
        return tf.group(*updates)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "beta_1": self._serialize_hyperparameter("beta_1"),
                "beta_2": self._serialize_hyperparameter("beta_2"),
                "decay": self._serialize_hyperparameter("decay"),
                "weight_decay": self._serialize_hyperparameter("weight_decay"),
                "sma_threshold": self._serialize_hyperparameter("sma_threshold"),
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
                "rectify": self.rectify,
                "total_steps": self._serialize_hyperparameter("total_steps"),
                "warmup_proportion": self._serialize_hyperparameter(
                    "warmup_proportion"
                ),
                "min_lr": self._serialize_hyperparameter("min_lr"),
            }
        )
        return config


class Lion(tf.keras.optimizers.legacy.Optimizer):
    r"""Implements the Lion optimization algorithm.

    The Lion optimizer is a custom optimization algorithm based on first-order
    stochastic gradient descent methods. It incorporates a weighted decay term
    and momentum-based updates.

    Attributes:
        learning_rate (float): The learning rate. Defaults to 1e-4.
        beta_1 (float): The exponential decay rate for the first moment estimates. Defaults to 0.9.
        beta_2 (float): The exponential decay rate for the second moment estimates. Defaults to 0.99.
        wd (float): The weight decay factor. Defaults to 0.
        name (str): The name of the optimizer. Defaults to "lion".
    """

    def __init__(
        self, learning_rate=1e-4, beta_1=0.9, beta_2=0.99, wd=0, name="lion", **kwargs
    ):
        """Construct a new Lion optimizer."""

        super(Lion, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self._set_hyper("wd", wd)

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self.add_slot(var, "m")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(Lion, self)._prepare_local(var_device, var_dtype, apply_state)

        beta_1_t = tf.identity(self._get_hyper("beta_1", var_dtype))
        beta_2_t = tf.identity(self._get_hyper("beta_2", var_dtype))
        wd_t = tf.identity(self._get_hyper("wd", var_dtype))
        lr = apply_state[(var_device, var_dtype)]["lr_t"]
        apply_state[(var_device, var_dtype)].update(
            dict(
                lr=lr,
                beta_1_t=beta_1_t,
                one_minus_beta_1_t=1 - beta_1_t,
                beta_2_t=beta_2_t,
                one_minus_beta_2_t=1 - beta_2_t,
                wd_t=wd_t,
            )
        )

    @tf.function(jit_compile=True)
    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        m = self.get_slot(var, "m")
        var_t = var.assign_sub(
            coefficients["lr_t"]
            * (
                tf.math.sign(
                    m * coefficients["beta_1_t"]
                    + grad * coefficients["one_minus_beta_1_t"]
                )
                + var * coefficients["wd_t"]
            )
        )
        with tf.control_dependencies([var_t]):
            m.assign(
                m * coefficients["beta_2_t"] + grad * coefficients["one_minus_beta_2_t"]
            )

    @tf.function(jit_compile=True)
    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        m = self.get_slot(var, "m")
        m_t = m.assign(m * coefficients["beta_1_t"])
        m_scaled_g_values = grad * coefficients["one_minus_beta_1_t"]
        m_t = m_t.scatter_add(tf.IndexedSlices(m_scaled_g_values, indices))
        var_t = var.assign_sub(
            coefficients["lr"] * (tf.math.sign(m_t) + var * coefficients["wd_t"])
        )

        with tf.control_dependencies([var_t]):
            m_t = m_t.scatter_add(tf.IndexedSlices(-m_scaled_g_values, indices))
            m_t = m_t.assign(m_t * coefficients["beta_2_t"] / coefficients["beta_1_t"])
            m_scaled_g_values = grad * coefficients["one_minus_beta_2_t"]
            m_t.scatter_add(tf.IndexedSlices(m_scaled_g_values, indices))

    def get_config(self):
        config = super(Lion, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "beta_1": self._serialize_hyperparameter("beta_1"),
                "beta_2": self._serialize_hyperparameter("beta_2"),
                "wd": self._serialize_hyperparameter("wd"),
            }
        )
        return config
