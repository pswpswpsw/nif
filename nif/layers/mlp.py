"""Native dense residual, einsum and bias layers."""
import torch
from torch import nn

from ._common import activation as _activation
from ._common import DenseLayer
from ._common import NativeLayer

# Preserve the public resolver used by the dynamic lookup below.
activation = _activation


class MLP_ResNet(NativeLayer):
    """Two-layer residual block: activation(x + L2(activation(L1(x))))."""

    def __init__(
        self,
        width,
        activation,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        mixed_policy="float32",
        **kwargs
    ):
        super().__init__()
        self.width = width
        self.activation = activation
        self._configure_precision(kwargs.pop("dtype", mixed_policy))
        self.act = globals()["activation"](activation)
        args = dict(
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            mixed_policy=mixed_policy,
        )
        self.L1 = DenseLayer(width, width, activation_name=activation, **args)
        self.L2 = DenseLayer(width, width, **args)

    def forward(self, x, **kwargs):
        dtype = self._compute_dtype(self.L1.w)
        h = self.L2(self.L1(x))
        return self.act(x.to(dtype=dtype) + h.to(dtype=dtype)).to(dtype=self.L1.w.dtype)

    def regularization_loss(self, inputs=None):
        return self.L1.regularization_loss() + self.L2.regularization_loss()

    def get_prunable_weights(self):
        return self.L1.get_prunable_weights() + self.L2.get_prunable_weights()

    def get_config(self):
        return dict(width=self.width, activation=self.activation)


class MLP_SimpleShortCut(NativeLayer):
    """One activated dense layer with a direct input shortcut: x + L1(x)."""

    def __init__(
        self,
        width,
        activation,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        mixed_policy="float32",
        **kwargs
    ):
        super().__init__()
        self.width = width
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.mixed_policy = mixed_policy
        self._configure_precision(kwargs.pop("dtype", mixed_policy))
        self.L1 = DenseLayer(
            width,
            width,
            activation,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            mixed_policy,
        )

    def forward(self, x, **kwargs):
        return x.to(dtype=self._compute_dtype(self.L1.w)) + self.L1(x)

    def regularization_loss(self, inputs=None):
        return self.L1.regularization_loss()

    def get_prunable_weights(self):
        return self.L1.get_prunable_weights()

    def get_config(self):
        return dict(
            width=self.width,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            mixed_policy=self.mixed_policy,
        )


class EinsumLayer(NativeLayer):
    """Wrap ``torch.einsum``; call with a tuple of operand tensors."""

    def __init__(self, equation, **kwargs):
        super().__init__()
        self.equation = equation

    def forward(self, inputs, *args, **kwargs):
        return torch.einsum(self.equation, *inputs)

    def get_config(self):
        return {"equation": self.equation}


class BiasAddLayer(NativeLayer):
    """Add a trainable bias initialized from N(0,.1), truncated at ±.2."""

    def __init__(self, output_dim, mixed_policy="float32", **kwargs):
        super().__init__()
        self.output_dim = output_dim
        self.mixed_policy = mixed_policy
        self._configure_precision(kwargs.pop("dtype", mixed_policy))
        self.last_layer_bias = nn.Parameter(
            torch.empty(output_dim, dtype=self.variable_Dtype)
        )
        nn.init.trunc_normal_(self.last_layer_bias, std=0.1, a=-0.2, b=0.2)

    def forward(self, inputs):
        dtype = self._compute_dtype(self.last_layer_bias)
        return inputs.to(dtype=dtype) + self.last_layer_bias.to(dtype=dtype)

    def get_config(self):
        return dict(output_dim=self.output_dim, mixed_policy=self.mixed_policy)
