# code comes from: https://github.com/GPflow/GPflow/issues/1604
from typing import List
from typing import Sequence
from typing import Union

import tensorflow as tf
import tensorflow_probability as tfp


def pack_tensors(tensors: Sequence[Union[tf.Tensor, tf.Variable]]) -> tf.Tensor:
    flats = [tf.reshape(tensor, (-1,)) for tensor in tensors]
    tensors_vector = tf.concat(flats, axis=0)
    return tensors_vector


def unpack_tensors(
    to_tensors: Sequence[Union[tf.Tensor, tf.Variable]], from_vector: tf.Tensor
) -> List[tf.Tensor]:
    s = 0
    values = []
    for target_tensor in to_tensors:
        shape = tf.shape(target_tensor)
        dtype = target_tensor.dtype
        tensor_size = tf.reduce_prod(shape)
        tensor_vector = from_vector[s : s + tensor_size]
        tensor = tf.reshape(tf.cast(tensor_vector, dtype), shape)
        values.append(tensor)
        s += tensor_size
    return values


def assign_tensors(
    to_tensors: Sequence[tf.Variable], values: Sequence[tf.Tensor]
) -> None:
    if len(to_tensors) != len(values):
        raise ValueError("to_tensors and values should have same length")
    for target, value in zip(to_tensors, values):
        target.assign(value)


def create_value_and_gradient_function(loss_closure, trainable_variables, verbose=1):
    """A factory to create a function required by tfp.optimizer.lbfgs_minimize.
    Args:
        loss_closure:
        trainable_variables:
    Returns:
        A function that has a signature of:
            loss_value, gradients = f(model_parameters).
    """
    i = tf.Variable(0)

    def assign_param_values_to_variables(x):
        values = unpack_tensors(trainable_variables, x)
        assign_tensors(trainable_variables, values)

    @tf.function
    def f_value_and_gradients(x):
        """A function that can be used by tfp.optimizer.lbfgs_minimize"""
        # update params
        assign_param_values_to_variables(x)
        # compute loss value and gradients w.r.t. trainable variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(trainable_variables)
            loss_value = loss_closure()
        grads = tape.gradient(loss_value, trainable_variables)

        i.assign_add(1)
        if verbose > 0:
            tf.print("Iter:", i, "loss:", loss_value)

        # return loss and flattened gradients
        return loss_value, pack_tensors(grads)

    return f_value_and_gradients, assign_param_values_to_variables


class LBFGSOptimizer(object):
    def __init__(self, loss_closure, trainable_variables, steps=1):
        tf.keras.backend.set_floatx("float64")
        self.initial_position = pack_tensors(trainable_variables)
        self.results = None
        func, assign = create_value_and_gradient_function(
            loss_closure=loss_closure, trainable_variables=trainable_variables
        )
        self.func = func
        self.assign = assign
        self.steps = steps

    @property
    def epoch(self):
        if self.results is None:
            return 0
        return int(self.results.num_iterations.numpy())

    @property
    def loss(self):
        if self.results is None:
            return None
        return float(self.results.objective_value.numpy())

    def minimize(self):
        if self.results is None:
            initial_poisition = self.initial_position
        else:
            initial_poisition = None
        self.results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=self.func,
            initial_position=initial_poisition,
            previous_optimizer_results=self.results,
            max_iterations=tf.cast(self.epoch + self.steps, dtype=tf.int32),
        )
        self.assign(self.results.position)
