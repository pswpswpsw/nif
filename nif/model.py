"""Native Neural Implicit Flow models.

A parameter network maps parameters (time, operating conditions) to the weights
of a coordinate-dependent shape network. Generated kernels retain the original
row-major [input, output] ordering; they are tensors, not copied Parameters, so
spatial derivatives and hypernetwork training remain connected by autograd.
"""
import copy
import json
import numbers
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from .layers import SIREN, SIREN_ResNet, HyperLinearForSIREN
from .layers import MLP_ResNet, MLP_SimpleShortCut, BiasAddLayer

__all__ = ['NIF', 'NIFMultiScale', 'NIFMultiScaleLastLayerParameterized']


from .layers._common import activation as _activation


def _policy(policy):
    if isinstance(policy, torch.dtype):
        return str(policy).removeprefix('torch.'), policy, policy
    name = getattr(policy, 'name', policy)
    dtypes = {'float16': torch.float16, 'float32': torch.float32,
              'float64': torch.float64, 'bfloat16': torch.bfloat16}
    if name in ('mixed_float16', 'mixed_bfloat16'):
        return name, torch.float32, dtypes[name.removeprefix('mixed_')]
    if name not in dtypes:
        raise ValueError(f'Unsupported precision policy: {policy}')
    return name, dtypes[name], dtypes[name]


def _truncated(tensor):
    return nn.init.trunc_normal_(tensor, std=.1, a=-.2, b=.2)


class _Dense(nn.Module):
    """The baseline dense map, with its truncated normal initializer."""
    def __init__(self, inputs, outputs, activation=None, dtype=torch.float32):
        super().__init__()
        self.linear = nn.Linear(inputs, outputs, dtype=dtype)
        self.register_buffer("_dtype_anchor", torch.empty(0, dtype=dtype), persistent=False)
        _truncated(self.linear.weight)
        _truncated(self.linear.bias)
        self.activation = _activation(activation)

    def forward(self, x):
        return self.activation(self.linear(x.to(dtype=self._dtype_anchor.dtype)))


class _ParameterView(nn.Sequential):
    """Register the same layers, never their parent model (no module cycles)."""
    def __init__(self, *layers, mixed_policy='float32'):
        super().__init__(*layers)
        self.mixed_policy = mixed_policy

    def forward(self, x):
        ref = next(self.parameters(), None)
        if ref is not None:
            x = torch.as_tensor(x, device=ref.device, dtype=ref.dtype)
        context = (torch.autocast(device_type=x.device.type, dtype=_policy(self.mixed_policy)[2])
                   if self.mixed_policy.startswith('mixed_') else nullcontext())
        with context:
            return super().forward(x)


class _ShapeView(nn.Module):
    def __init__(self, model, phi_only=False):
        super().__init__()
        self.cfg = copy.deepcopy(model.cfg_shape_net)
        self.kind = ('last' if isinstance(model, NIFMultiScaleLastLayerParameterized)
                     else 'sine' if isinstance(model, NIFMultiScale) else 'mlp')
        self.pi_hidden = model.pi_hidden
        self.mixed_policy = model.mixed_policy
        self.register_buffer('_precision_anchor', model._precision_anchor.clone(), persistent=False)
        self.phi_only = phi_only
        if self.kind == 'last':
            self.snet_list = model.snet_list
            if not phi_only:
                self.last_bias_layer = model.last_bias_layer

    def forward(self, x, w=None):
        context = (torch.autocast(device_type=self._precision_anchor.device.type, dtype=_policy(self.mixed_policy)[2])
                   if self.mixed_policy.startswith('mixed_') else nullcontext())
        with context:
            out = self._evaluate(x, w)
        return out.to(self._precision_anchor.dtype) if self.mixed_policy.startswith('mixed_') else out

    def _evaluate(self, x, w=None):
        if not self.phi_only and w is None:
            x, w = x
        if self.kind == 'last':
            ref = next(self.snet_list.parameters())
            x = torch.as_tensor(x, device=ref.device, dtype=ref.dtype)
            phi = x
            for layer in self.snet_list:
                phi = layer(phi)
            phi = phi.reshape(-1, self.cfg['output_dim'], self.pi_hidden)
            if self.phi_only:
                return phi
            w = torch.as_tensor(w, device=phi.device, dtype=phi.dtype)
            return self.last_bias_layer(torch.einsum('bol,bl->bo', phi, w))
        if not isinstance(w, torch.Tensor):
            w = torch.as_tensor(w)
        x = torch.as_tensor(x, device=w.device, dtype=w.dtype)
        s = self.cfg
        args = (x, w, s['input_dim'], s['output_dim'], s['units'], s['nlayers'])
        if self.kind == 'sine':
            return NIFMultiScale._call_shape_net_mres(
                x, w, s['use_resblock'], s['omega_0'], *args[2:], w.dtype)
        return NIF._call_shape_net(*args, s['activation'], w.dtype)


def _unpack(w, si_dim, so_dim, width, hidden):
    """All kernels first, then all biases, in the original TF flat order."""
    sizes = [(si_dim, width)] + [(width, width)] * hidden + [(width, so_dim)]
    expected = sum(a*b+b for a,b in sizes)
    if w.ndim != 2 or w.shape[-1] != expected:
        raise ValueError(f'Expected generated weights of shape [batch, {expected}]')
    pos = 0
    kernels = []
    for a,b in sizes:
        kernels.append(w[:, pos:pos+a*b].reshape(-1, a, b))
        pos += a*b
    biases = []
    for _,b in sizes:
        biases.append(w[:, pos:pos+b])
        pos += b
    return kernels, biases


def _penalty(parameters, cfg, zero):
    # Preserve the original precedence: specifying L2 selects L2, otherwise L1.
    if isinstance(cfg.get('l2_reg'), numbers.Real):
        return zero + cfg['l2_reg'] * sum(p.square().sum() for p in parameters)
    if isinstance(cfg.get('l1_reg'), numbers.Real):
        return zero + cfg['l1_reg'] * sum(p.abs().sum() for p in parameters)
    return zero


class NIF(nn.Module):
    """Parameter-conditioned implicit field with shortcut MLP shape network.

    Input columns are [parameters, coordinates]. ``build()`` and ``model()``
    return this module. Call ``regularization_loss(batch)`` explicitly in the
    training objective; there are no cached batch losses or implicit fit loops.
    """
    def __init__(self, cfg_shape_net, cfg_parameter_net, mixed_policy='float32'):
        super().__init__()
        if not isinstance(cfg_shape_net, dict) or not isinstance(cfg_parameter_net, dict):
            raise TypeError('Network configurations must be dictionaries')
        self.cfg_shape_net = copy.deepcopy(cfg_shape_net)
        self.cfg_parameter_net = copy.deepcopy(cfg_parameter_net)
        self.mixed_policy, variable_dtype, self._compute_dtype = _policy(mixed_policy)
        self.register_buffer('_precision_anchor', torch.empty(0, dtype=variable_dtype), persistent=False)
        self.si_dim = cfg_shape_net['input_dim']
        self.so_dim = cfg_shape_net['output_dim']
        self.n_sx = cfg_shape_net['units']
        self.l_sx = cfg_shape_net['nlayers']
        self.pi_dim = cfg_parameter_net['input_dim']
        self.pi_hidden = cfg_parameter_net['latent_dim']
        self.n_st = cfg_parameter_net['units']
        self.l_st = cfg_parameter_net['nlayers']
        for name in ('si_dim', 'so_dim', 'n_sx', 'pi_dim', 'pi_hidden', 'n_st'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if any(not isinstance(v, int) or v < 0 for v in (self.l_sx, self.l_st)):
            raise ValueError('nlayers must be a nonnegative integer')
        for name in ('jac_reg', 'l1_reg', 'l2_reg', 'act_l1_reg', 'act_l2_reg'):
            setattr(self, 'p_'+name, cfg_parameter_net.get(name))
        self.pnet_list = nn.ModuleList(self._initialize_pnet(cfg_parameter_net, cfg_shape_net))

    @property
    def variable_Dtype(self):
        return self._precision_anchor.dtype

    @property
    def compute_Dtype(self):
        return self._compute_dtype if self.mixed_policy.startswith('mixed_') else self.variable_Dtype

    def _initialize_pnet(self, p, s):
        self.po_dim = self.l_sx*self.n_sx**2 + (self.si_dim+self.so_dim+1+self.l_sx)*self.n_sx+self.so_dim
        layers = [_Dense(self.pi_dim, self.n_st, p['activation'], self.variable_Dtype)]
        for _ in range(self.l_st):
            layers.append(MLP_SimpleShortCut(self.n_st, p['activation'], _truncated,
                          _truncated, None, None, self.mixed_policy))
        layers.extend([_Dense(self.n_st, self.pi_hidden, dtype=self.variable_Dtype),
                       _Dense(self.pi_hidden, self.po_dim, dtype=self.variable_Dtype)])
        return layers

    def _inputs(self, inputs):
        x = torch.as_tensor(inputs, dtype=self.variable_Dtype, device=self._precision_anchor.device)
        if x.ndim != 2 or x.shape[-1] != self.pi_dim+self.si_dim:
            raise ValueError(f'inputs must have shape [batch, {self.pi_dim+self.si_dim}] in [parameter, coordinate] order')
        return x

    def _autocast(self):
        return (torch.autocast(device_type=self._precision_anchor.device.type, dtype=self.compute_Dtype)
                if self.mixed_policy.startswith('mixed_') else nullcontext())

    def forward(self, inputs, training=None, mask=None):
        x = self._inputs(inputs)
        with self._autocast():
            w, _ = self._call_parameter_net(x[:, :self.pi_dim], self.pnet_list)
            out = self._shape(x[:, self.pi_dim:].to(w.dtype), w)
        return out.to(self.variable_Dtype)

    def call(self, inputs, training=None, mask=None):
        return self.forward(inputs, training=training, mask=mask)

    def _shape(self, x, w):
        return self._call_shape_net(x, w, self.si_dim, self.so_dim, self.n_sx,
                                   self.l_sx, self.cfg_shape_net['activation'], self.variable_Dtype)

    @staticmethod
    def _call_shape_net(input_s, pnet_output, si_dim, so_dim, n_sx, l_sx, activation, variable_dtype):
        ws, bs = _unpack(pnet_output, si_dim, so_dim, n_sx, l_sx)
        act = _activation(activation)
        u = act(torch.einsum('bi,bij->bj', input_s, ws[0]) + bs[0])
        for i in range(l_sx):
            u = u + act(torch.einsum('bi,bij->bj', u, ws[i+1]) + bs[i+1])
        return (torch.einsum('bi,bij->bj', u, ws[-1])+bs[-1]).to(_policy(variable_dtype)[1])

    @staticmethod
    def _call_parameter_net(input_p, pnet_list):
        latent = input_p
        for layer in pnet_list[:-1]:
            latent = layer(latent)
        return pnet_list[-1](latent), latent

    def build(self):
        return self

    def model(self):
        return self

    def model_p_to_w(self):
        return _ParameterView(*self.pnet_list, mixed_policy=self.mixed_policy)

    def model_p_to_lr(self):
        return _ParameterView(*self.pnet_list[:-1], mixed_policy=self.mixed_policy)

    def model_lr_to_w(self):
        return _ParameterView(self.pnet_list[-1], mixed_policy=self.mixed_policy)

    def model_x_to_u_given_w(self):
        return _ShapeView(self)

    def regularization_loss(self, inputs=None):
        """Differentiable scalar: parameter sums, activity sum/batch, Jacobian mean.

        Input-dependent regularization always uses the supplied batch. Calling
        this without inputs when such a penalty is configured raises ValueError.
        Shape-net penalties in the last-layer model use the shape coefficients.
        """
        zero = next(self.parameters()).sum()*0
        loss = _penalty(self.pnet_list.parameters(), self.cfg_parameter_net, zero)
        if hasattr(self, 'snet_list'):
            loss = loss + _penalty(self.snet_list.parameters(), self.cfg_shape_net, zero)
        dependent = any(isinstance(v, numbers.Real) for v in
                        (self.p_act_l1_reg, self.p_act_l2_reg, self.p_jac_reg))
        if dependent:
            if inputs is None:
                raise ValueError('inputs are required for activity and latent Jacobian regularization')
            with torch.enable_grad(), self._autocast():
                x = self._inputs(inputs)
                if x.shape[0] == 0:
                    raise ValueError('regularization inputs must contain at least one row')
                if not x.requires_grad:
                    x = x.detach().requires_grad_(True)
                w, latent = self._call_parameter_net(x[:, :self.pi_dim], self.pnet_list)
                if isinstance(self.p_act_l2_reg, numbers.Real):
                    loss = loss + self.p_act_l2_reg*w.square().sum()/x.shape[0]
                elif isinstance(self.p_act_l1_reg, numbers.Real):
                    loss = loss + self.p_act_l1_reg*w.abs().sum()/x.shape[0]
                if isinstance(self.p_jac_reg, numbers.Real):
                    jac = []
                    for i in range(latent.shape[-1]):
                        grad = torch.autograd.grad(latent[:, i].sum(), x,
                                                   create_graph=True, retain_graph=True)[0]
                        jac.append(grad[:, :self.pi_dim])
                    loss = loss + self.p_jac_reg*torch.stack(jac, 1).square().mean()
        return loss

    def predict(self, inputs, batch_size=None):
        """Convenience inference returning a tensor on this module's device."""
        with torch.no_grad():
            x = self._inputs(inputs)
            if batch_size is None:
                return self(x)
            if batch_size <= 0:
                raise ValueError('batch_size must be positive')
            return torch.cat([self(part) for part in x.split(batch_size)])

    def save_config(self, filename='config.json'):
        with open(filename, 'w') as handle:
            json.dump(dict(cfg_shape_net=self.cfg_shape_net,
                           cfg_parameter_net=self.cfg_parameter_net,
                           mixed_policy=(self.mixed_policy if self.mixed_policy.startswith('mixed_')
                                         else str(self.variable_Dtype).removeprefix('torch.'))), handle, indent=4)


class NIFMultiScale(NIF):
    """SIREN shape network with optional two-sine residual blocks."""
    def _initialize_pnet(self, p, s):
        if not isinstance(s.get('use_resblock'), bool):
            raise ValueError('cfg_shape_net.use_resblock must be bool')
        connectivity = s.get('connectivity')
        if connectivity == 'last_layer':
            self.po_dim = self.pi_hidden
        elif connectivity == 'full':
            hidden = self.l_sx*(2 if s['use_resblock'] else 1)
            self.po_dim = hidden*self.n_sx**2 + (self.si_dim+self.so_dim+1+hidden)*self.n_sx+self.so_dim
        else:
            raise ValueError('connectivity must be full or last_layer')
        layers = []
        if p['activation'] == 'sine':
            omega = p['omega_0']
            layers.append(SIREN(self.pi_dim, self.n_st, 'first', omega, s, mixed_policy=self.mixed_policy))
            for _ in range(self.l_st):
                if p['use_resblock']:
                    layers.append(SIREN_ResNet(self.n_st, self.n_st, omega, mixed_policy=self.mixed_policy))
                else:
                    layers.append(SIREN(self.n_st, self.n_st, 'hidden', omega, s, mixed_policy=self.mixed_policy))
            layers.append(SIREN(self.n_st, self.pi_hidden, 'bottleneck', omega, s, mixed_policy=self.mixed_policy))
        else:
            layers.append(_Dense(self.pi_dim, self.n_st, p['activation'], self.variable_Dtype))
            for _ in range(self.l_st):
                cls = MLP_ResNet if p['use_resblock'] else MLP_SimpleShortCut
                layers.append(cls(self.n_st, p['activation'], _truncated, _truncated,
                                  None, None, self.mixed_policy))
            layers.append(_Dense(self.n_st, self.pi_hidden, dtype=self.variable_Dtype))
        layers.append(HyperLinearForSIREN(self.pi_hidden, self.po_dim, s, self.mixed_policy,
                                         connectivity=connectivity))
        return layers

    def _shape(self, x, w):
        return self._call_shape_net_mres(x, w, self.cfg_shape_net['use_resblock'],
            self.cfg_shape_net['omega_0'], self.si_dim, self.so_dim, self.n_sx, self.l_sx, self.variable_Dtype)

    @staticmethod
    def _call_shape_net_mres(input_s, pnet_output, flag_resblock, omega_0,
                             si_dim, so_dim, n_sx, l_sx, variable_dtype):
        ws, bs = _unpack(pnet_output, si_dim, so_dim, n_sx, l_sx*(2 if flag_resblock else 1))
        def sine(x, i):
            return torch.sin(omega_0*torch.einsum('bi,bij->bj', x, ws[i])+bs[i])
        u = sine(input_s, 0)
        for i in range(l_sx):
            if flag_resblock:
                u = .5*(u+sine(sine(u, 2*i+1), 2*i+2))
            else:
                u = sine(u, i+1)
        return (torch.einsum('bi,bij->bj', u, ws[-1])+bs[-1]).to(_policy(variable_dtype)[1])


class NIFMultiScaleLastLayerParameterized(NIFMultiScale):
    """Linear representation u(p,x) = phi(x) a(p) + b, with learned bases."""
    def __init__(self, cfg_shape_net, cfg_parameter_net, mixed_policy='float32'):
        if cfg_shape_net.get('connectivity') != 'last_layer':
            raise ValueError('last-layer model requires connectivity=last_layer')
        super().__init__(cfg_shape_net, cfg_parameter_net, mixed_policy)
        self.s_l1_reg = cfg_shape_net.get('l1_reg')
        self.s_l2_reg = cfg_shape_net.get('l2_reg')
        self.snet_list = nn.ModuleList(self._initialize_snet(cfg_shape_net))
        self.last_bias_layer = BiasAddLayer(self.so_dim, mixed_policy=self.mixed_policy)

    def _initialize_snet(self, s):
        layers = [SIREN(self.si_dim, self.n_sx, 'first', s['omega_0'], s, mixed_policy=self.mixed_policy)]
        for _ in range(self.l_sx):
            if s['use_resblock']:
                layers.append(SIREN_ResNet(self.n_sx, self.n_sx, s['omega_0'], mixed_policy=self.mixed_policy))
            else:
                layers.append(SIREN(self.n_sx, self.n_sx, 'hidden', s['omega_0'], s, mixed_policy=self.mixed_policy))
        layers.append(SIREN(self.n_sx, self.po_dim*self.so_dim, 'bottleneck', s['omega_0'], s,
                            mixed_policy=self.mixed_policy))
        return layers

    def _call_shape_net_get_phi_x(self, input_s, snet_layers_list, so_dim, pi_hidden):
        phi = input_s
        for layer in snet_layers_list:
            phi = layer(phi)
        return phi.reshape(-1, so_dim, pi_hidden)

    def _call_shape_net_mres_only_para_last_layer(self, input_s, snet_layers_list,
                                                 pnet_output, so_dim, pi_hidden, variable_dtype):
        phi = self._call_shape_net_get_phi_x(input_s, snet_layers_list, so_dim, pi_hidden)
        return self.last_bias_layer(torch.einsum('bol,bl->bo', phi, pnet_output)).to(_policy(variable_dtype)[1])

    def _shape(self, x, w):
        return self._call_shape_net_mres_only_para_last_layer(x, self.snet_list, w,
                                                            self.so_dim, self.pi_hidden, self.variable_Dtype)

    def model_p_to_lr(self):
        return self.model_p_to_w()

    def model_lr_to_w(self):
        # In this architecture coefficients ARE the generated final-layer weights.
        return nn.Identity()

    def model_x_to_phi(self):
        return _ShapeView(self, phi_only=True)
