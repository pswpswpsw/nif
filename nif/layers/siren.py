"""SIREN and SIREN hypernetwork layers with the original NIF parameter layout.

Kernels are stored as [input, output], preserving generated weight ordering.
The sinusoidal argument is omega_0 * (x @ w) + b (bias is not scaled).
"""
import math

import torch
from torch import nn

from ._common import NativeLayer
from ._common import penalty
from ._common import resolve_dtype


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
    variable_dtype,
):
    """Generate hypernetwork kernels and shape-aware initial output biases.

    Generated entries have first-layer, hidden-layer, final-layer kernels,
    followed by all biases. Last-layer kernels use Glorot scaling, since they
    have no sine activation; all generated biases use 1 / width scaling.
    """
    if min(num_inputs, num_outputs, input_dim, width) <= 0 or omega_0 <= 0:
        raise ValueError("Dimensions and omega_0 must be positive")
    if weight_factor < 0:
        raise ValueError("weight_factor must be nonnegative")
    dtype = resolve_dtype(variable_dtype)
    bound = math.sqrt(6.0 / num_inputs) * weight_factor
    w = torch.empty(num_inputs, num_outputs, dtype=dtype).uniform_(-bound, bound)
    scale = torch.ones(num_outputs, dtype=dtype)
    first = num_weight_first
    hidden = first + num_weight_hidden
    last = hidden + num_weight_last
    if (
        min(num_weight_first, num_weight_hidden, num_weight_last) < 0
        or last > num_outputs
    ):
        raise ValueError("Invalid generated-weight section sizes")
    scale[:first] /= input_dim
    scale[first:hidden] *= math.sqrt(6.0 / width) / omega_0
    scale[hidden:last] *= math.sqrt(6.0 / (width + width))
    scale[last:] /= width
    b = torch.empty(num_outputs, dtype=dtype).uniform_(-1.0, 1.0) * scale
    return w, b


def compute_number_of_weightbias_by_its_position_for_shapenet(cfg_shape_net):
    """Count generated first, hidden and final kernels (excluding biases)."""
    si_dim = cfg_shape_net["input_dim"]
    so_dim = cfg_shape_net["output_dim"]
    width = cfg_shape_net["units"]
    layers = cfg_shape_net["nlayers"]
    connectivity = cfg_shape_net["connectivity"]
    if connectivity == "full":
        first = si_dim * width
        hidden = (
            (2 if cfg_shape_net.get("use_resblock", False) else 1) * layers * width**2
        )
    elif connectivity in ("last", "last_layer"):
        first = hidden = 0
    else:
        raise ValueError("cfg_shape_net['connectivity'] must be 'last' or 'full'")
    return first, hidden, so_dim * width


class SIREN(NativeLayer):
    """A sinusoidal layer, or a linear bottleneck/hypernetwork output layer.

    ``mixed_policy`` accepts a torch dtype, a dtype name, ``mixed_float16``,
    ``mixed_bfloat16``, or an object exposing compute_dtype/variable_dtype.
    Explicit ``regularization_loss`` returns kernel and bias penalties.
    """

    def __init__(
        self,
        num_inputs,
        num_outputs,
        layer_position,
        omega_0,
        cfg_shape_net=None,
        kernel_regularizer=None,
        bias_regularizer=None,
        mixed_policy="float32",
        **kwargs,
    ):
        super().__init__()
        self._configure_precision(kwargs.pop("dtype", mixed_policy))
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.layer_position = layer_position
        self.omega_0 = float(omega_0)
        self.cfg_shape_net = cfg_shape_net
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        if num_inputs <= 0 or num_outputs <= 0 or self.omega_0 <= 0:
            raise ValueError("Dimensions and omega_0 must be positive")
        if layer_position in ("first", "hidden", "bottleneck"):
            bound = (
                1.0 / num_inputs
                if layer_position == "first"
                else math.sqrt(6.0 / num_inputs) / self.omega_0
            )
            w = torch.empty(
                num_inputs, num_outputs, dtype=self.variable_Dtype
            ).uniform_(-bound, bound)
            b = torch.empty(num_outputs, dtype=self.variable_Dtype).uniform_(
                -1.0 / math.sqrt(num_inputs), 1.0 / math.sqrt(num_inputs)
            )
        elif layer_position == "last":
            # The legacy isinstance test was inverted and rejected valid configs.
            if not isinstance(cfg_shape_net, dict):
                raise ValueError("cfg_shape_net is required for a last SIREN layer")
            counts = compute_number_of_weightbias_by_its_position_for_shapenet(
                cfg_shape_net
            )
            w, b = gen_hypernetwork_weights_bias_for_siren_shapenet(
                num_inputs,
                num_outputs,
                cfg_shape_net["weight_init_factor"],
                *counts,
                cfg_shape_net["input_dim"],
                cfg_shape_net["units"],
                self.omega_0,
                self.variable_Dtype,
            )
        else:
            raise ValueError(f"Unknown SIREN layer position {layer_position!r}")
        self.w = nn.Parameter(w)
        self.w._nif_gc_axes = (0,)
        self.b = nn.Parameter(b)
        self.w_init = w.detach().clone()
        self.b_init = b.detach().clone()

    def forward(self, x, **kwargs):
        dtype = self._compute_dtype(self.w)
        linear = x.to(dtype=dtype) @ self.w.to(dtype=dtype)
        if self.layer_position in ("last", "bottleneck"):
            return linear + self.b.to(dtype=dtype)
        return torch.sin(self.omega_0 * linear + self.b.to(dtype=dtype))

    def regularization_loss(self, inputs=None):
        return penalty(self.kernel_regularizer, self.w) + penalty(
            self.bias_regularizer, self.b
        )

    def get_config(self):
        return dict(
            num_inputs=self.num_inputs,
            num_outputs=self.num_outputs,
            layer_position=self.layer_position,
            omega_0=self.omega_0,
            cfg_shape_net=self.cfg_shape_net,
        )

    def get_prunable_weights(self):
        return [self.w]


class SIREN_ResNet(SIREN):
    """Two sine layers with the original 0.5 * (input + hidden) residual.

    Both sets start with identical samples, as in the scientific baseline,
    but are distinct trainable parameters.
    """

    def __init__(
        self,
        num_inputs,
        num_outputs,
        omega_0,
        kernel_regularizer=None,
        bias_regularizer=None,
        mixed_policy="float32",
        **kwargs,
    ):
        if num_inputs != num_outputs:
            raise ValueError(
                "SIREN residual blocks require equal input and output widths"
            )
        super().__init__(
            num_inputs,
            num_outputs,
            "hidden",
            omega_0,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            mixed_policy=mixed_policy,
            **kwargs,
        )
        self.w2 = nn.Parameter(self.w.detach().clone())
        self.w2._nif_gc_axes = (0,)
        self.b2 = nn.Parameter(self.b.detach().clone())

    def forward(self, x, training=None, mask=None):
        dtype = self._compute_dtype(self.w)
        x = x.to(dtype=dtype)
        h = torch.sin(
            self.omega_0 * (x @ self.w.to(dtype=dtype)) + self.b.to(dtype=dtype)
        )
        return 0.5 * (
            x
            + torch.sin(
                self.omega_0 * (h @ self.w2.to(dtype=dtype)) + self.b2.to(dtype=dtype)
            )
        )

    def regularization_loss(self, inputs=None):
        return (
            super().regularization_loss()
            + penalty(self.kernel_regularizer, self.w2)
            + penalty(self.bias_regularizer, self.b2)
        )

    def get_prunable_weights(self):
        return [self.w, self.w2]


class HyperLinearForSIREN(NativeLayer):
    """Linear hypernetwork head with shape-aware SIREN initialization."""

    def __init__(
        self,
        num_inputs,
        num_outputs,
        cfg_shape_net,
        mixed_policy="float32",
        connectivity="full",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        **kwargs,
    ):
        super().__init__()
        self._configure_precision(kwargs.pop("dtype", mixed_policy))
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.cfg_shape_net = dict(cfg_shape_net)
        self.connectivity = connectivity
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        if connectivity == "full":
            counts = compute_number_of_weightbias_by_its_position_for_shapenet(
                cfg_shape_net
            )
        elif connectivity in ("last_layer", "last"):
            counts = (0, 0, num_outputs)
        else:
            raise ValueError("connectivity must be 'full' or 'last_layer'")
        w, b = gen_hypernetwork_weights_bias_for_siren_shapenet(
            num_inputs,
            num_outputs,
            cfg_shape_net["weight_init_factor"],
            *counts,
            cfg_shape_net["input_dim"],
            cfg_shape_net["units"],
            cfg_shape_net["omega_0"],
            self.variable_Dtype,
        )
        self.w = nn.Parameter(w)
        self.w._nif_gc_axes = (0,)
        self.b = nn.Parameter(b)

    def forward(self, x, **kwargs):
        dtype = self._compute_dtype(self.w)
        return x.to(dtype=dtype) @ self.w.to(dtype=dtype) + self.b.to(dtype=dtype)

    def regularization_loss(self, inputs=None):
        loss = penalty(self.kernel_regularizer, self.w) + penalty(
            self.bias_regularizer, self.b
        )
        if self.activity_regularizer is not None:
            if inputs is None:
                raise ValueError("inputs are required for activity regularization")
            if inputs.shape[0] == 0:
                raise ValueError("Activity regularization requires a nonempty batch")
            # Keras activity regularizers divide their scalar penalty by batch size.
            loss = (
                loss
                + penalty(self.activity_regularizer, self(inputs)) / inputs.shape[0]
            )
        return loss

    def get_prunable_weights(self):
        return [self.w]

    def get_config(self):
        return dict(
            num_inputs=self.num_inputs,
            num_outputs=self.num_outputs,
            cfg_shape_net=self.cfg_shape_net,
            connectivity=self.connectivity,
        )
