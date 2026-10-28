# Copyright 2018 The TensorFlow Probability Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See TFP_LICENSE.txt and TFP_NOTICE.txt in this directory.
# Modified 2026: unbatched PyTorch adaptation, optimizer/checkpoint interface.
"""Single-device real PyTorch L-BFGS with the original TFP stopping contract.

The scalar core follows TFP 0.25 lbfgs/bfgs_utils and Hager--Zhang (2005/2006).
No TensorFlow dependency, batched optimization, or optimizer differentiation is
provided. Math stays in the parameter dtype. Backend reductions/rounding may
differ from TensorFlow; this does not establish full scientific tutorial parity.
"""
import copy
import math
from types import SimpleNamespace

import torch
from torch.optim import Optimizer

from nif.optimizers._hager_zhang import Point, divide_no_nan, hager_zhang


_SCHEMA = 'nif.reference_lbfgs.v1'


class ReferenceLBFGS(Optimizer):
    """Unbatched reference L-BFGS with independent gradient/x/f stopping tests.

    ``max_iter`` limits ADDITIONAL iterations per ``step``; result counters and
    correction history are cumulative. ``step`` returns the entry loss, with a
    detached runtime ``result`` exposing the final state. The generic helper
    translates its cumulative ``max_iterations`` into the remaining budget.
    Only one parameter group, a single device/dtype (real f32/f64), dense
    contiguous parameters, and dense gradients are supported. Saved state is
    tensors/primitives only and can be loaded with ``weights_only=True``.
    """
    def __init__(self, params, max_iter=50, num_correction_pairs=10, tolerance=1e-8,
                 x_tolerance=0., f_relative_tolerance=0., f_absolute_tolerance=0.,
                 max_line_search_iterations=50):
        options = dict(max_iter=max_iter, num_correction_pairs=num_correction_pairs,
                       tolerance=tolerance, x_tolerance=x_tolerance,
                       f_relative_tolerance=f_relative_tolerance,
                       f_absolute_tolerance=f_absolute_tolerance,
                       max_line_search_iterations=max_line_search_iterations)
        self._validate_options(options)
        super().__init__(params, options)
        if len(self.param_groups) != 1:
            raise ValueError('ReferenceLBFGS supports exactly one parameter group')
        self._parameters = list(self.param_groups[0]['params'])
        if not self._parameters or any(p.numel() == 0 for p in self._parameters):
            raise ValueError('ReferenceLBFGS requires nonempty parameters')
        first = self._parameters[0]
        if first.dtype not in (torch.float32, torch.float64):
            raise TypeError('ReferenceLBFGS requires real float32 or float64 parameters')
        if any(p.dtype != first.dtype or p.device != first.device or not p.is_contiguous()
               or p.layout != torch.strided for p in self._parameters):
            raise ValueError('Parameters must be contiguous and share one dtype/device')
        self._shapes = [list(p.shape) for p in self._parameters]
        self._size = sum(p.numel() for p in self._parameters)
        self.result = None

    def add_param_group(self, param_group):
        if self.param_groups:
            raise ValueError('ReferenceLBFGS supports exactly one parameter group')
        return super().add_param_group(param_group)

    @staticmethod
    def _validate_options(options):
        for name in ('max_iter', 'num_correction_pairs', 'max_line_search_iterations'):
            value = options[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == 'max_iter' else 1):
                raise ValueError(name + ' must be an integer in its valid range')
        for name in ('tolerance', 'x_tolerance', 'f_relative_tolerance', 'f_absolute_tolerance'):
            value = float(options[name])
            if not math.isfinite(value) or value < 0:
                raise ValueError(name + ' must be finite and nonnegative')

    def _flat(self):
        return torch.cat([p.detach().reshape(-1) for p in self._parameters])

    def _assign(self, position):
        with torch.no_grad():
            start = 0
            for p in self._parameters:
                p.copy_(position[start:start + p.numel()].view_as(p))
                start += p.numel()

    def _evaluate(self, closure, position):
        self._assign(position)
        with torch.enable_grad():
            loss = closure()
        first = self._parameters[0]
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise ValueError('Closure must return a scalar tensor')
        if loss.dtype != first.dtype or loss.device != first.device:
            raise ValueError('Closure loss must match parameter dtype/device')
        gradients = []
        for p in self._parameters:
            grad = p.grad
            if grad is None:
                gradients.append(torch.zeros_like(p).reshape(-1))
            else:
                if grad.layout != torch.strided:
                    raise ValueError('Sparse gradients are not supported')
                if grad.dtype != first.dtype or grad.device != first.device or grad.shape != p.shape:
                    raise ValueError('Gradient must match parameter dtype/device/shape')
                gradients.append(grad.detach().reshape(-1))
        return loss.detach().clone(), torch.cat(gradients).clone()

    def _publish(self, state):
        self.result = SimpleNamespace(**{key: value.detach().clone() if isinstance(value, torch.Tensor)
                                        else value for key, value in state.items()})

    @staticmethod
    def _direction(state):
        gradient = state['objective_gradient']
        positions, gradients = state['position_deltas'], state['gradient_deltas']
        count = min(state['num_iterations'], len(positions))
        if count == 0:
            return -gradient
        rho = (gradients * positions).sum(dim=-1)
        q = gradient.clone()
        alphas = []
        # Include the zero queue prefix just as TFP's scan does. Its selected
        # q is unaffected by those zeros, which use divide_no_nan.
        for s, y, sy in zip(positions.flip(0), gradients.flip(0), rho.flip(0)):
            alpha = divide_no_nan((s * q).sum(), sy)
            alphas.append(alpha)
            q = q - alpha * y
        gamma = rho[-1] / (gradients[-1] * gradients[-1]).sum()
        r = gamma * q
        for alpha, s, y, sy in zip(reversed(alphas), positions, gradients, rho):
            beta = divide_no_nan((y * r).sum(), sy)
            r = r + (alpha - beta) * s
        return -r

    @staticmethod
    def _nonfinite(state, value, gradient):
        state['converged'] = state['converged'] or bool(value == -torch.inf)
        invalid = not bool(torch.isfinite(value) & torch.isfinite(gradient).all())
        state['failed'] = state['failed'] or (not state['converged'] and invalid)

    @torch.no_grad()
    def step(self, closure):
        if closure is None:
            raise ValueError('ReferenceLBFGS requires a closure')
        options = self.param_groups[0]
        self._validate_options(options)
        state = self.state[self._parameters[0]]
        current = self._flat()
        accepted = current.clone()
        saved_state = copy.deepcopy(dict(state))
        try:
            if state:
                if not torch.equal(current, state['position']):
                    raise ValueError('Parameters changed outside optimizer; construct a fresh optimizer')
                if len(state['position_deltas']) != options['num_correction_pairs']:
                    raise ValueError('Correction history size cannot change during resume')
                entry_loss = state['objective_value'].clone()
            else:
                value, gradient = self._evaluate(closure, current)
                entry_loss = value.clone()
                queue = current.new_zeros((options['num_correction_pairs'], self._size))
                state.update(position=current.clone(), objective_value=value, objective_gradient=gradient,
                    num_iterations=0, num_objective_evaluations=1,
                    converged=bool(torch.isfinite(value) & (gradient.abs().max() < options['tolerance'])),
                    failed=False, position_deltas=queue, gradient_deltas=queue.clone())
            for _ in range(options['max_iter']):
                if state['converged'] or state['failed']:
                    break
                self._nonfinite(state, state['objective_value'], state['objective_gradient'])
                inactive = state['converged'] or state['failed']
                direction = self._direction(state)
                previous = state['position']
                old_value = state['objective_value']
                old_gradient = state['objective_gradient']
                zero = Point(previous.new_zeros(()), old_value,
                             (old_gradient * direction).sum(), old_gradient)
                def evaluate(step_size):
                    value, gradient = self._evaluate(closure, previous + step_size * direction)
                    return Point(step_size, value, (gradient * direction).sum(), gradient)
                search = hager_zhang(evaluate, zero, options['max_line_search_iterations'], inactive=inactive)
                state['num_iterations'] += 1
                state['num_objective_evaluations'] += search.evaluations
                state['failed'] = state['failed'] or (not state['converged'] and not search.converged)
                if not inactive:
                    point = search.left
                    self._nonfinite(state, point.f, point.gradient)
                    position = previous + point.x * direction
                    change = (point.f - old_value).abs()
                    converged = not state['failed'] and bool(
                        (point.gradient.abs().max() <= options['tolerance'])
                        | ((position - previous).abs().max() <= options['x_tolerance'])
                        | (change <= options['f_relative_tolerance'] * old_value)
                        | (change <= options['f_absolute_tolerance']))
                    state.update(position=position, objective_value=point.f,
                                 objective_gradient=point.gradient,
                                 converged=state['converged'] or converged)
                    if not state['converged'] and not state['failed']:
                        for key, delta in (('position_deltas', position - previous),
                                           ('gradient_deltas', point.gradient - old_gradient)):
                            state[key] = torch.cat((state[key][1:], delta.unsqueeze(0)), dim=0)
                accepted = state['position'].clone()
            self._assign(state['position'])
            self._publish(state)
            return entry_loss
        except Exception:
            # Keep completed accepted iterations, but never leave a failed
            # closure's speculative parameters installed in the model.
            if state and 'position' in state:
                accepted = state['position'].clone()
            self._assign(accepted)
            if not saved_state and 'position' not in state:
                state.clear()
            if state and 'position' in state:
                self._publish(state)
            raise

    def state_dict(self):
        result = super().state_dict()
        result['reference_lbfgs'] = dict(schema=_SCHEMA, dtype=str(self._parameters[0].dtype),
            device=str(self._parameters[0].device), shapes=self._shapes)
        return result

    def load_state_dict(self, state_dict):
        marker = state_dict.get('reference_lbfgs')
        expected = dict(schema=_SCHEMA, dtype=str(self._parameters[0].dtype),
                        device=str(self._parameters[0].device), shapes=self._shapes)
        if marker != expected:
            raise ValueError('Not a compatible ReferenceLBFGS schema/dtype/device/shape checkpoint')
        groups = state_dict.get('param_groups', [])
        if len(groups) != 1 or len(groups[0].get('params', [])) != len(self._parameters):
            raise ValueError('Checkpoint parameter group mismatch')
        self._validate_options(groups[0])
        raw_states = state_dict.get('state', {})
        identifiers = groups[0]['params']
        if set(raw_states) - {identifiers[0]}:
            raise ValueError('Unexpected checkpoint state entries')
        raw = raw_states.get(identifiers[0], {})
        if raw:
            shape = (self._size,)
            tensor_shapes = {'position': shape, 'objective_gradient': shape, 'objective_value': (),
                'position_deltas': (groups[0]['num_correction_pairs'], self._size),
                'gradient_deltas': (groups[0]['num_correction_pairs'], self._size)}
            required = set(tensor_shapes) | {'num_iterations', 'num_objective_evaluations', 'converged', 'failed'}
            if set(raw) != required:
                raise ValueError('Malformed reference checkpoint fields')
            for key, shape in tensor_shapes.items():
                value = raw[key]
                if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                        or value.dtype != self._parameters[0].dtype
                        or value.device != self._parameters[0].device or not value.is_contiguous()):
                    raise ValueError('Checkpoint tensor mismatch: ' + key)
            for key in ('num_iterations', 'num_objective_evaluations'):
                if type(raw[key]) is not int or raw[key] < 0:
                    raise ValueError('Malformed checkpoint counter')
            if any(type(raw[key]) is not bool for key in ('converged', 'failed')):
                raise ValueError('Malformed checkpoint flags')
        clean = copy.deepcopy(state_dict)
        clean.pop('reference_lbfgs')
        super().load_state_dict(clean)
        state = self.state[self._parameters[0]]
        if state:
            self._publish(state)
        else:
            self.result = None
