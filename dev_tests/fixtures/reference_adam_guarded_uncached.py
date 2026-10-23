"""Native Adam with the reference notebooks' epsilon-hat convention."""
import math

import torch
from torch.optim import Optimizer


class ReferenceAdam(Optimizer):
    r"""Adam with epsilon added to the square root of *raw* second moments.

    For optimizer iteration t (starting at one), the dense update is::

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        alpha = lr * sqrt(1 - beta2**t) / (1 - beta1**t)
        parameter -= alpha * m / (sqrt(v) + eps)

    This is the TensorFlow/Keras reference's epsilon-hat convention, including
    its default 1e-7. It is not torch.optim.Adam's bias-corrected-denominator
    epsilon. No TensorFlow runtime is used. Scalars are evaluated in each
    parameter's device/dtype; framework scalar-promotion and fused-operation
    rounding can still differ. Explicit float32-promoted betas may be supplied
    for a float64 diagnostic of that backend's declared hyperparameters.

    Supports dense real floating parameters, parameter groups, closures,
    AMSGrad, and native state_dict restoration. The shared optimizer iteration
    advances once per step, including parameters with a missing gradient;
    missing gradients leave their parameters and slots unchanged. No weight
    decay, clipping, sparse gradients, capturable or differentiable mode is
    implied. Tutorial regularization remains part of the objective.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-7,
                 amsgrad=False):
        self._iterations = 0
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, amsgrad=amsgrad))

    def add_param_group(self, param_group):
        options = {**self.defaults, **param_group}
        if not math.isfinite(options['lr']) or options['lr'] < 0:
            raise ValueError('lr must be finite and nonnegative')
        if not math.isfinite(options['eps']) or options['eps'] <= 0:
            raise ValueError('eps (epsilon hat) must be finite and positive')
        if len(options['betas']) != 2 or any(not 0 <= b < 1 for b in options['betas']):
            raise ValueError('betas must be in [0, 1)')
        super().add_param_group(param_group)

    def state_dict(self):
        result = super().state_dict()
        result['reference_iterations'] = self._iterations
        return result

    def load_state_dict(self, state_dict):
        iterations = int(state_dict['reference_iterations'])
        if iterations < 0:
            raise ValueError('reference_iterations must be nonnegative')
        super().load_state_dict(state_dict)
        self._iterations = iterations

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Validate before changing any parameter or the shared iteration.
        has_gradient = False
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    has_gradient = True
                    if p.grad.is_sparse:
                        raise RuntimeError('ReferenceAdam requires dense gradients')
                    if not p.is_floating_point():
                        raise RuntimeError('ReferenceAdam requires real floating parameters')
        if not has_gradient:
            raise ValueError('No gradients provided for any parameter')
        self._iterations += 1
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                m, v = state['exp_avg'], state['exp_avg_sq']
                b1, b2 = (p.new_tensor(b) for b in group['betas'])
                # Match new Keras Adam's dense assign_add ordering. Its Python
                # (1-beta) is evaluated before conversion to the slot dtype.
                m.add_((p.grad - m) * (1 - group['betas'][0]))
                v.add_((p.grad.square() - v) * (1 - group['betas'][1]))
                variance = v
                if group['amsgrad']:
                    if 'max_exp_avg_sq' not in state:
                        state['max_exp_avg_sq'] = torch.zeros_like(p)
                    variance = state['max_exp_avg_sq']
                    torch.maximum(variance, v, out=variance)
                t = p.new_tensor(self._iterations)
                alpha = p.new_tensor(group['lr']) * (1 - b2.pow(t)).sqrt() / (1 - b1.pow(t))
                p.sub_(alpha * m / (variance.sqrt() + group['eps']))
        return loss
