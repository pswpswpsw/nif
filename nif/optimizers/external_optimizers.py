"""Native AdaBelief, Lion and the completed published L4-Adam algorithm.

The public Keras-era option names remain accepted, but parameters are supplied
as the first argument and updates use ``zero_grad/backward/step``. Optimizer
state is saved with the ordinary PyTorch ``state_dict`` interface.

Ordinary optimizer clipping follows the baseline OptimizerV2 order: clipvalue
first, then clipnorm independently for each tensor, or global_clipnorm across
all parameters. Group-specific clipvalue/clipnorm settings are supported;
global_clipnorm must agree across groups. This differs intentionally from the
historical ``gtcf.get_centralized_gradients`` helper, whose explicit code uses
a common norm before clipvalue.

Reference implementation of those base-optimizer conventions:
https://github.com/tensorflow/tensorflow/blob/v2.8.0/tensorflow/python/keras/optimizer_v2/optimizer_v2.py

"""
import math

import torch
from torch.optim import Optimizer


# Distinguish an omitted native alias from an explicit None fallback request.
_EPS_UNSET = object()


def _number(value, step):
    return float(value(step) if callable(value) else value)


def _validate(lr, betas, epsilon=0, weight_decay=0):
    if not callable(lr) and lr < 0:
        raise ValueError('learning rate must be nonnegative')
    if not all(0 <= beta < 1 for beta in betas):
        raise ValueError('betas must lie in [0, 1)')
    if epsilon < 0 or (not callable(weight_decay) and weight_decay < 0):
        raise ValueError('epsilon and weight decay must be nonnegative')


def centralize_gradient(gradient, parameter=None, axes=None):
    """Centralize over input axes, respecting native Linear's transposed kernel.

    Native weights have ``[out, in, ...]`` layout, so the default reduction is
    over axes 1 onward. A custom ``[in, out]`` parameter may declare
    ``parameter._nif_gc_axes = (0,)`` (or pass ``axes=(0,)``).
    Bias vectors are unchanged.
    """
    if gradient.ndim <= 1:
        return gradient
    if axes is None:
        axes = getattr(parameter, '_nif_gc_axes', tuple(range(1, gradient.ndim)))
    return gradient - gradient.mean(dim=tuple(axes), keepdim=True)


def _validate_clipping(options):
    for key in ('clipnorm', 'global_clipnorm', 'clipvalue'):
        value = options.get(key)
        if value is not None and (not math.isfinite(float(value)) or value < 0):
            raise ValueError(f'{key} must be finite and nonnegative')
    if options.get('clipnorm') is not None and options.get('global_clipnorm') is not None:
        raise ValueError('clipnorm and global_clipnorm are mutually exclusive')


def _norm_scale(norm, limit):
    # Match clip_by_norm without changing tiny gradients by an added epsilon.
    # A zero limit is valid and zeros every finite gradient, including norm 0.
    if limit == 0:
        return torch.zeros_like(norm)
    return limit / norm.clamp_min(limit)


def _gradients(optimizer):
    result = []
    global_clipnorm = optimizer.param_groups[0].get('global_clipnorm')
    for group in optimizer.param_groups:
        _validate_clipping(group)
        if group.get('global_clipnorm') != global_clipnorm:
            raise ValueError('global_clipnorm must agree across all parameter groups')
        entries = []
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad.detach()
            # Sparse AdaBelief updates use the original sparse row mask below.
            sparse = grad.coalesce() if grad.is_sparse else None
            if sparse is not None:
                grad = sparse.to_dense()
            if group.get('gradient_centralization', False):
                grad = centralize_gradient(grad, p)
                sparse = None
            entries.append((p, grad, sparse))
        # OptimizerV2 clips component values before either norm transform.
        clipvalue = group.get('clipvalue')
        if clipvalue is not None:
            entries = [(p, g.clamp(-clipvalue, clipvalue), s) for p, g, s in entries]
        clipnorm = group.get('clipnorm')
        if clipnorm is not None:
            entries = [(p, g * _norm_scale(g.norm(), clipnorm), s) for p, g, s in entries]
        result.append((group, entries))
    if global_clipnorm is not None:
        gradients = [g for _, entries in result for _, g, _ in entries]
        if gradients:
            reference = gradients[0]
            norm = torch.sqrt(sum(g.square().sum().to(reference) for g in gradients))
            scale = _norm_scale(norm, global_clipnorm)
            result = [(group, [(p, g * scale.to(g), sparse) for p, g, sparse in entries])
                      for group, entries in result]
    return result


def _options(kwargs):
    options = {key: kwargs.pop(key, default) for key, default in
               [('gradient_centralization', False), ('clipnorm', None),
                ('global_clipnorm', None), ('clipvalue', None)]}
    if 'centralize_gradients' in kwargs:
        options['gradient_centralization'] = kwargs.pop('centralize_gradients')
    if 'use_gc' in kwargs:
        options['gradient_centralization'] = kwargs.pop('use_gc')
    if kwargs:
        raise TypeError('Unknown optimizer options: ' + ', '.join(kwargs))
    _validate_clipping(options)
    return options


class AdaBeliefOptimizer(Optimizer):
    """AdaBelief with the original NIF variance and rectification conventions.

    ``m = beta1*m + (1-beta1)*grad``;
    ``v = beta2*v + (1-beta2)*(grad-m)**2 + epsilon``.
    Epsilon is added both to the uncorrected variance and the final square-root
    denominator, exactly as in the baseline. Weight decay is decoupled from
    moments. Rectification, AMSGrad, warmup/linear decay, learning-rate schedules,
    clipping, and gradient centralization are supported. Bias-correction powers,
    step scalars, SMA and rectification use each parameter's dtype/device, as in
    the reference; identical scalars are cached only within a step and group.
    Moment coefficients subtract from one after casting beta to parameter dtype;
    each product is evaluated before addition, using the updated m in variance.
    Native float64 parameters retain caller beta precision without a float32 cast.
    After the closure, all-None gradients raise before optimizer/schedule mutation.
    Zero tensors count as updates. On a real update all existing groups advance
    their per-group clocks; a late-added group starts with its own step one.
    Explicit ``epsilon=None`` or zero uses the baseline backend default 1e-7.
    The native ``eps`` alias takes precedence when supplied and uses the same
    fallback; omitting both retains this optimizer's default epsilon of 1e-14.
    """
    def __init__(self, params, learning_rate=.001, beta_1=.9, beta_2=.999,
                 epsilon=1e-14, weight_decay=0., rectify=True, amsgrad=False,
                 sma_threshold=5., total_steps=0, warmup_proportion=.1,
                 min_lr=0., name='AdaBeliefOptimizer', print_change_log=True,
                 lr=None, betas=None, eps=_EPS_UNSET, decay=0., **kwargs):
        lr = learning_rate if lr is None else lr
        betas = (beta_1, beta_2) if betas is None else betas
        epsilon = (epsilon if eps is _EPS_UNSET else eps) or 1e-7
        _validate(lr, betas, epsilon, weight_decay)
        if total_steps < 0 or not 0 <= warmup_proportion <= 1 or min_lr < 0 or decay < 0:
            raise ValueError('Invalid warmup or decay options')
        defaults = dict(lr=lr, betas=betas, eps=epsilon, weight_decay=weight_decay,
                        rectify=rectify, amsgrad=amsgrad, sma_threshold=sma_threshold,
                        total_steps=int(total_steps), warmup_proportion=warmup_proportion,
                        min_lr=min_lr, decay=decay, **_options(kwargs))
        self.name = name
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if not any(p.grad is not None for group in self.param_groups for p in group['params']):
            raise ValueError('No gradients provided for any parameter')
        for group, entries in _gradients(self):
            # The baseline optimizer has one time counter, including parameters
            # without a gradient. Keep it per group for native parameter groups.
            step = group.get('step', 0) + 1
            group['step'] = step
            b1, b2 = group['betas']; eps = group['eps']
            lr = _number(group['lr'], step - 1) / (1 + group['decay'] * (step - 1))
            total = group['total_steps']
            if total:
                warmup = total * group['warmup_proportion']
                if warmup > 0 and step <= warmup:
                    lr *= step / warmup
                else:
                    decay_steps = max(total - warmup, 1)
                    lr += (group['min_lr'] - lr) * min(step - warmup, decay_steps) / decay_steps
            wd = _number(group['weight_decay'], step - 1)
            # Reference scalar arithmetic uses the parameter dtype, including
            # powers and cancellation-sensitive SMA. Cache only within this step
            # and group; tensor selection avoids a host sync for each parameter.
            scalar_cache = {}
            for p, grad, sparse in entries:
                key = (p.device, p.dtype)
                if key not in scalar_cache:
                    beta1, beta2 = p.new_tensor(b1), p.new_tensor(b2)
                    local_step = p.new_tensor(step)
                    power1, power2 = beta1.pow(local_step), beta2.pow(local_step)
                    correction1, correction2 = 1 - power1, 1 - power2
                    rectify, r = None, None
                    if group['rectify']:
                        rho_inf = 2 / (1 - beta2) - 1
                        rho = rho_inf - 2 * local_step * power2 / correction2
                        rectify = (rho >= group['sma_threshold']) & (rho > 4)
                        r = ((rho-4)/(rho_inf-4)*(rho-2)/(rho_inf-2)*rho_inf/rho).sqrt()
                    scalar_cache[key] = beta1, beta2, 1 - beta1, 1 - beta2, correction1, correction2, rectify, r
                beta1, beta2, complement1, complement2, correction1, correction2, rectify, r = scalar_cache[key]
                state = self.state[p]
                if not state:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_var'] = torch.zeros_like(p)
                    if group['amsgrad']:
                        state['max_exp_avg_var'] = torch.zeros_like(p)
                state['step'] = step
                m, v = state['exp_avg'], state['exp_avg_var']
                # The reference forms both products before their sum/assignment.
                m.mul_(beta1).add_(grad * complement1)
                residual = (grad-m).square()
                mask = None
                if sparse is not None:
                    # Only observed sparse entries contribute to variance, as
                    # in the original scatter-update AdaBelief implementation.
                    mask = torch.zeros_like(p, dtype=torch.bool)
                    mask[tuple(sparse.indices())] = True
                    residual = residual * mask
                v.mul_(beta2).add_(residual * complement2).add_(eps)
                variance = v
                if group['amsgrad']:
                    variance = state['max_exp_avg_var']
                    torch.maximum(variance, v, out=variance)
                direction = m / correction1
                denominator = (variance / correction2).sqrt() + eps
                if group['rectify']:
                    direction = torch.where(rectify, r * direction / denominator, direction)
                else:
                    direction = direction / denominator
                if wd:
                    direction = direction.add(p, alpha=wd)
                if mask is not None:
                    direction = direction * mask
                p.sub_(direction * lr)
        return loss

    def get_config(self):
        return dict(self.defaults, name=self.name)


class Lion(Optimizer):
    """Lion: sign of the beta1 blend, followed by beta2 momentum update.

    Weight decay is decoupled. ``wd`` and ``weight_decay`` are aliases.
    """
    def __init__(self, params, learning_rate=1e-4, beta_1=.9, beta_2=.99,
                 wd=0., name='lion', lr=None, betas=None, weight_decay=None, decay=0.,
                 **kwargs):
        lr = learning_rate if lr is None else lr
        betas = (beta_1, beta_2) if betas is None else betas
        wd = wd if weight_decay is None else weight_decay
        _validate(lr, betas, weight_decay=wd)
        if decay < 0:
            raise ValueError('decay must be nonnegative')
        self.name = name
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=wd, decay=decay, **_options(kwargs)))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group, entries in _gradients(self):
            step = group.get('step', 0); group['step'] = step+1
            lr = _number(group['lr'], step) / (1 + group['decay']*step)
            wd = _number(group['weight_decay'], step)
            b1, b2 = group['betas']
            for p, grad, _ in entries:
                state = self.state[p]
                if not state:
                    state['exp_avg'] = torch.zeros_like(p)
                m = state['exp_avg']
                direction = (m*b1 + grad*(1-b1)).sign()
                p.mul_(1-lr*wd).add_(direction, alpha=-lr)
                m.mul_(b2).add_(grad, alpha=1-b2)
                state['step'] = step+1
        return loss

    def get_config(self):
        return dict(self.defaults, name=self.name)


class L4Adam(Optimizer):
    """Published L4-Adam, completing the unfinished baseline implementation.

    Reference: Rolinek and Martius, NeurIPS 2018, *L4: Practical loss-based
    stepsize adaptation for deep learning*.

    Raw exponential moving averages have betas ``1-1/tau_m`` and
    ``1-1/tau_s``. Bias-correct them to obtain ``g`` and ``s``, and set
    ``v = g / (sqrt(s) + epsilon)``. Initialize ``L_min = gamma_0 * L``,
    then take ``min(L_min, L)`` before each update. The SINGLE global step is
    ``alpha*(L-gamma*L_min)/(sum_over_all_parameters(g*v)+epsilon)``.
    Apply ``parameter -= step*v`` and then ``L_min *= 1+1/tau``.
    The epsilon in the direction is an explicit numerical safeguard, separate
    from the additive denominator epsilon. ``lr``/``learning_rate`` denote
    alpha (default .15), not an extra Adam step. All groups must share the
    global step options; moment timescales can differ between groups.

    Use ``step(loss=current_loss)`` after backward, or supply a closure that
    returns the current loss. A closure may perform backward itself; otherwise
    it is performed here. The moving minimum and global counter are included
    in ``state_dict`` and resume exactly.
    Explicit ``epsilon=None`` or zero uses the baseline backend default 1e-7.
    The native ``eps`` alias takes precedence when supplied and uses the same
    fallback. Omitting both retains the default epsilon of 1e-7.
    """
    def __init__(self, params, learning_rate=.15, tau_m=10., tau_s=1000.,
                 tau=1000., gamma_0=.75, gamma=.9, epsilon=1e-7,
                 name='L4Adam', lr=None, eps=_EPS_UNSET, **kwargs):
        lr = learning_rate if lr is None else lr
        epsilon = (epsilon if eps is _EPS_UNSET else eps) or 1e-7
        if tau_m < 1 or tau_s < 1 or tau <= 0 or epsilon <= 0:
            raise ValueError('tau_m/tau_s must be >=1, tau and epsilon must be positive')
        if not 0 <= gamma_0 <= 1 or not 0 <= gamma <= 1:
            raise ValueError('gamma_0 and gamma must lie in [0,1]')
        _validate(lr, (1-1/tau_m, 1-1/tau_s), epsilon)
        self.name = name
        super().__init__(params, dict(lr=lr, tau_m=tau_m, tau_s=tau_s, tau=tau,
                                     gamma_0=gamma_0, gamma=gamma, eps=epsilon,
                                     **_options(kwargs)))

    @property
    def l_min(self):
        return self.state.get('global', {}).get('l_min')

    @torch.no_grad()
    def step(self, closure=None, loss=None):
        if closure is not None and not callable(closure):
            if loss is not None:
                raise ValueError('Specify the loss only once')
            loss, closure = closure, None
        if closure is not None:
            self.zero_grad()
            with torch.enable_grad():
                loss = closure()
                if not any(p.grad is not None for group in self.param_groups for p in group['params']):
                    loss.backward()
        if loss is None:
            raise ValueError('L4Adam requires a current scalar loss or a closure')
        value = float(loss.detach()) if isinstance(loss, torch.Tensor) else float(loss)
        if not math.isfinite(value):
            raise ValueError('L4Adam loss must be finite')
        base = self.param_groups[0]
        for group in self.param_groups[1:]:
            if any(group[key] != base[key] for key in ('lr', 'tau', 'gamma', 'gamma_0', 'eps')):
                raise ValueError('L4Adam parameter groups must share global step options')
        prepared = _gradients(self)
        if not any(entries for _, entries in prepared):
            return loss
        state = self.state.setdefault('global', {'step': 0, 'l_min': base['gamma_0']*value})
        state['step'] += 1
        minimum = min(state['l_min'], value)
        directions = []; denominator = None
        for group, entries in prepared:
            b1, b2 = 1-1/group['tau_m'], 1-1/group['tau_s']
            for p, grad, sparse in entries:
                if sparse is not None:
                    raise RuntimeError('L4Adam does not support sparse gradients')
                local = self.state[p]
                if not local:
                    local.update(exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p), step=0)
                local['step'] += 1; t = local['step']
                local['exp_avg'].mul_(b1).add_(grad, alpha=1-b1)
                local['exp_avg_sq'].mul_(b2).addcmul_(grad, grad, value=1-b2)
                g = local['exp_avg'] / (1-b1**t)
                s = local['exp_avg_sq'] / (1-b2**t)
                v = g / (s.sqrt()+group['eps'])
                dot = (g*v).sum()
                denominator = dot if denominator is None else denominator + dot.to(denominator)
                directions.append((p, v))
        multiplier = _number(base['lr'], state['step']-1) * (value-base['gamma']*minimum) / (denominator+base['eps'])
        for p, v in directions:
            p.sub_(v * multiplier.to(p))
        state['l_min'] = minimum * (1+1/base['tau'])
        return loss

    def get_config(self):
        return dict(self.defaults, name=self.name)
