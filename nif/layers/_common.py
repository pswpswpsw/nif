"""Small native layer utilities; no framework compatibility runtime is needed."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def resolve_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None:
        return torch.float32
    name = str(dtype).replace('torch.', '')
    names = {'float16': torch.float16, 'half': torch.float16,
             'float32': torch.float32, 'float': torch.float32,
             'float64': torch.float64, 'double': torch.float64,
             'bfloat16': torch.bfloat16}
    if name not in names:
        raise ValueError(f'Unsupported floating point dtype: {dtype}')
    return names[name]


def policy_dtypes(policy='float32'):
    """Return variable, compute dtypes for native strings or policy-like objects."""
    if hasattr(policy, 'variable_dtype'):
        return resolve_dtype(policy.variable_dtype), resolve_dtype(policy.compute_dtype)
    if policy == 'mixed_float16':
        return torch.float32, torch.float16
    if policy == 'mixed_bfloat16':
        return torch.float32, torch.bfloat16
    dtype = resolve_dtype(policy)
    return dtype, dtype


def activation(value):
    if callable(value):
        return value
    functions = {None: lambda x: x, 'linear': lambda x: x, 'identity': lambda x: x,
                 'relu': F.relu, 'tanh': torch.tanh, 'sigmoid': torch.sigmoid,
                 'silu': F.silu, 'swish': F.silu, 'gelu': F.gelu, 'elu': F.elu,
                 'selu': F.selu, 'softplus': F.softplus, 'softsign': F.softsign,
                 'sin': torch.sin, 'sine': torch.sin, 'exponential': torch.exp,
                 'hard_sigmoid': lambda x: (x * .2 + .5).clamp(0, 1),
                 'softmax': lambda x: torch.softmax(x, dim=-1)}
    try:
        return functions[value]
    except KeyError as exc:
        raise ValueError(f'Unknown activation {value!r}') from exc


def _initializer_fans(shape):
    """Keras fans for [input, output] kernels, including scalar/1D biases."""
    if not shape:
        return 1, 1
    if len(shape) == 1:
        return shape[0], shape[0]
    receptive_field = math.prod(shape[:-2])
    return shape[-2] * receptive_field, shape[-1] * receptive_field


def initialize(tensor, initializer, default='glorot_uniform'):
    """Initialize kernels and biases with standard Keras string defaults.

    Kernels have [input, output] layout. A rank-one bias of length n has
    fan_in = fan_out = n. Constant defaults to zero; VarianceScaling defaults
    to scale=1, fan_in, truncated normal. Orthogonal and Identity have gain=1
    and require rank >=2 and rank ==2 respectively, as in the public baseline:
    https://github.com/keras-team/keras/blob/v2.11.0/keras/initializers/initializers_v2.py

    Existing 2D draw operations are retained to preserve seeded trajectories.
    """
    initializer = default if initializer is None else initializer
    with torch.no_grad():
        if callable(initializer):
            result = initializer(tensor)
            if result is not None and result is not tensor:
                tensor.copy_(torch.as_tensor(result, device=tensor.device, dtype=tensor.dtype))
            return
        fan_in, fan_out = _initializer_fans(tensor.shape)
        # VarianceScaling guards empty dimensions with max(1, fan).
        fan_avg = max(1, (fan_in + fan_out) / 2)
        fan_in = max(1, fan_in)
        if initializer in ('zeros', 'constant'): tensor.zero_()
        elif initializer == 'ones': tensor.fill_(1)
        elif initializer in ('glorot_uniform', 'xavier_uniform'):
            if tensor.ndim == 2:
                nn.init.xavier_uniform_(tensor)
            else:
                bound = math.sqrt(3 / fan_avg)
                nn.init.uniform_(tensor, -bound, bound)
        elif initializer == 'glorot_normal':
            # Keras GlorotNormal is variance-scaled *truncated* normal. Native
            # xavier_normal_ is untruncated and would change its distribution.
            std = math.sqrt(1 / fan_avg)/.8796256610342398
            nn.init.trunc_normal_(tensor,std=std,a=-2*std,b=2*std)
        elif initializer == 'xavier_normal':
            if tensor.ndim >= 2:
                nn.init.xavier_normal_(tensor)
            else:
                nn.init.normal_(tensor, std=math.sqrt(1 / fan_avg))
        elif initializer == 'random_normal': nn.init.normal_(tensor, std=.05)
        elif initializer == 'random_uniform': nn.init.uniform_(tensor, -.05, .05)
        elif initializer == 'truncated_normal': nn.init.trunc_normal_(tensor, std=.05, a=-.1, b=.1)
        elif initializer == 'he_uniform': nn.init.uniform_(tensor, -math.sqrt(6/fan_in), math.sqrt(6/fan_in))
        elif initializer == 'he_normal': nn.init.trunc_normal_(tensor, std=math.sqrt(2/fan_in)/.8796256610342398, a=-2*math.sqrt(2/fan_in)/.8796256610342398, b=2*math.sqrt(2/fan_in)/.8796256610342398)
        elif initializer == 'lecun_uniform': nn.init.uniform_(tensor, -math.sqrt(3/fan_in), math.sqrt(3/fan_in))
        elif initializer in ('lecun_normal', 'variance_scaling'):
            std = math.sqrt(1/fan_in)/.8796256610342398
            nn.init.trunc_normal_(tensor,std=std,a=-2*std,b=2*std)
        elif initializer == 'orthogonal':
            if tensor.ndim < 2:
                raise ValueError('orthogonal initializer requires at least rank 2')
            # Keras flattens all axes except the last into rows before QR.
            # Native QR has no half/bfloat16 kernel; preserve requested storage
            # precision by computing the factorization in float32 then casting.
            qr_dtype = torch.float32 if tensor.dtype in (torch.float16, torch.bfloat16) else tensor.dtype
            matrix = tensor.new_empty((math.prod(tensor.shape[:-1]), tensor.shape[-1]), dtype=qr_dtype)
            nn.init.orthogonal_(matrix)
            tensor.copy_(matrix.reshape(tensor.shape))
        elif initializer == 'identity':
            if tensor.ndim != 2:
                raise ValueError('identity initializer requires rank 2')
            nn.init.eye_(tensor)
        else: raise ValueError(f'Unknown initializer {initializer!r}')


def penalty(regularizer, value):
    if regularizer is None:
        return value.sum() * 0
    if callable(regularizer):
        return regularizer(value)
    if isinstance(regularizer, (int, float)):
        return regularizer * value.square().sum()
    if isinstance(regularizer, dict):
        return regularizer.get('l1', 0.) * value.abs().sum() + regularizer.get('l2', 0.) * value.square().sum()
    if regularizer == 'l1': return .01 * value.abs().sum()
    if regularizer == 'l2': return .01 * value.square().sum()
    if regularizer == 'l1_l2': return .01 * (value.abs().sum()+value.square().sum())
    raise ValueError(f'Unknown regularizer {regularizer!r}')


class NativeLayer(nn.Module):
    def call(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_config(self):
        return {}

    def _configure_precision(self, policy):
        self.variable_Dtype, self.compute_Dtype = policy_dtypes(policy)
        self._mixed_compute_dtype = (self.compute_Dtype if self.variable_Dtype != self.compute_Dtype else None)

    def _compute_dtype(self, parameter):
        # Standard .to/.double/.float must continue to control the computation.
        return self._mixed_compute_dtype or parameter.dtype


class DenseLayer(NativeLayer):
    """Dense layer with historical kernel layout, used inside residual blocks."""
    def __init__(self, input_dim, output_dim, activation_name=None,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, mixed_policy='float32'):
        super().__init__()
        self._configure_precision(mixed_policy)
        self.w=nn.Parameter(torch.empty(input_dim,output_dim,dtype=self.variable_Dtype))
        self.w._nif_gc_axes=(0,)
        self.b=nn.Parameter(torch.empty(output_dim,dtype=self.variable_Dtype))
        initialize(self.w,kernel_initializer)
        initialize(self.b,bias_initializer,'zeros')
        self.act=activation(activation_name)
        self.kernel_regularizer=kernel_regularizer
        self.bias_regularizer=bias_regularizer

    @property
    def kernel(self): return self.w

    @property
    def bias(self): return self.b

    @property
    def weight(self): return self.w

    def forward(self,x):
        dtype=self._compute_dtype(self.w)
        return self.act(x.to(dtype=dtype) @ self.w.to(dtype=dtype)+self.b.to(dtype=dtype))

    def regularization_loss(self,inputs=None):
        return penalty(self.kernel_regularizer,self.w)+penalty(self.bias_regularizer,self.b)

    def get_prunable_weights(self): return [self.w,self.b]
