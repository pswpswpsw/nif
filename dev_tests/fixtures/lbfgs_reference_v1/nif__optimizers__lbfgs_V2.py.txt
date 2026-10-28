"""Native PyTorch L-BFGS and portable tensor packing helpers.

Names from the original TensorFlow Probability wrapper are retained for source
migration. No TensorFlow graph or process-wide default dtype is used.
"""
from types import SimpleNamespace

import torch


def pack_tensors(tensors):
    tensors = list(tensors)
    if not tensors:
        return torch.empty(0)
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def unpack_tensors(to_tensors, from_vector):
    to_tensors = list(to_tensors)
    if from_vector.ndim != 1 or from_vector.numel() != sum(t.numel() for t in to_tensors):
        raise ValueError('Packed vector size does not match the target tensors')
    values = []; start = 0
    for target in to_tensors:
        values.append(from_vector[start:start+target.numel()].to(target).reshape(target.shape))
        start += target.numel()
    return values


def assign_tensors(to_tensors, values):
    to_tensors, values = list(to_tensors), list(values)
    if len(to_tensors) != len(values):
        raise ValueError('to_tensors and values should have same length')
    if any(target.shape != value.shape for target, value in zip(to_tensors, values)):
        raise ValueError('Assigned tensors must have matching shapes')
    with torch.no_grad():
        for target, value in zip(to_tensors, values):
            target.copy_(value)


def create_value_and_gradient_function(loss_closure, trainable_variables, verbose=1):
    variables = list(trainable_variables)

    def assign(x):
        assign_tensors(variables, unpack_tensors(variables, x))

    def value_and_gradients(x):
        assign(x)
        with torch.enable_grad():
            value = loss_closure()
            grads = torch.autograd.grad(value, variables, allow_unused=True)
        grads = [torch.zeros_like(p) if g is None else g for p, g in zip(variables, grads)]
        value_and_gradients.iter += 1
        if verbose:
            print(f'Iter: {value_and_gradients.iter} loss: {float(value.detach()):.8g}')
        return value.detach(), pack_tensors(grads).detach()

    value_and_gradients.iter = 0
    return value_and_gradients, assign


def lbfgs_minimize(value_and_gradients_function, initial_position=None,
                   previous_optimizer_results=None, max_iterations=50,
                   num_correction_pairs=10, tolerance=1e-8, x_tolerance=1e-12,
                   f_relative_tolerance=0., max_line_search_iterations=50,
                   parallel_iterations=1, **kwargs):
    """Run native torch.optim.LBFGS with a packed value-and-gradient callable.

    ``max_iterations`` is cumulative when resuming previous results, matching
    the original wrapper. PyTorch's strong-Wolfe line search uses its own
    internal trial limit; ``max_line_search_iterations`` bounds the total
    objective-evaluation budget per iteration. ``parallel_iterations`` is
    retained as an inert compatibility option (the solver is sequential).
    """
    if kwargs:
        raise TypeError('Unknown L-BFGS options: ' + ', '.join(kwargs))
    previous = previous_optimizer_results
    if initial_position is None:
        if previous is None:
            raise ValueError('Provide initial_position or previous_optimizer_results')
        initial_position = previous.position
    position = torch.nn.Parameter(initial_position.detach().clone())
    prior_iterations = 0 if previous is None else int(previous.num_iterations)
    iterations = max(0, int(max_iterations)-prior_iterations)
    optimizer = torch.optim.LBFGS([position], max_iter=max(iterations, 1),
                                  max_eval=max(iterations, 1)*int(max_line_search_iterations),
                                  history_size=int(num_correction_pairs),
                                  tolerance_grad=float(tolerance),
                                  tolerance_change=max(float(x_tolerance), float(f_relative_tolerance)),
                                  line_search_fn='strong_wolfe')
    if previous is not None and hasattr(previous, '_native_state'):
        optimizer.load_state_dict(previous._native_state)
        optimizer.param_groups[0].update(max_iter=max(iterations, 1), max_eval=max(iterations, 1)*int(max_line_search_iterations))
    evaluations = 0
    def closure():
        nonlocal evaluations
        optimizer.zero_grad()
        with torch.enable_grad():
            value, grad = value_and_gradients_function(position.detach())
        if grad.shape != position.shape:
            raise ValueError('Gradient shape does not match the packed parameters')
        position.grad = grad.detach().to(position).clone()
        evaluations += 1
        return value.detach()
    if iterations:
        optimizer.step(closure)
    value = closure()
    state = optimizer.state[position]
    actual_iterations = int(state.get('n_iter', prior_iterations))
    finite = bool(torch.isfinite(value) and torch.isfinite(position).all() and torch.isfinite(position.grad).all())
    return SimpleNamespace(
        position=position.detach().clone(), objective_value=value.detach().clone(),
        objective_gradient=position.grad.detach().clone(),
        num_iterations=actual_iterations,
        num_objective_evaluations=evaluations + (0 if previous is None else previous.num_objective_evaluations),
        converged=finite and bool(position.grad.abs().max() <= tolerance),
        failed=not finite, _native_state=optimizer.state_dict())


class LBFGSOptimizer:
    def __init__(self, loss_closure, trainable_variables, steps=1, verbose=1):
        variables = list(trainable_variables)
        self.initial_position = pack_tensors(variables).detach().clone()
        self.results = None
        self.func, self.assign = create_value_and_gradient_function(loss_closure, variables, verbose)
        self.steps = int(steps)
        if self.steps <= 0:
            raise ValueError('steps must be positive')

    @property
    def epoch(self):
        return 0 if self.results is None else int(self.results.num_iterations)

    @property
    def loss(self):
        return None if self.results is None else float(self.results.objective_value)

    def minimize(self):
        self.results = lbfgs_minimize(
            self.func, initial_position=self.initial_position if self.results is None else None,
            previous_optimizer_results=self.results, max_iterations=self.epoch+self.steps)
        self.assign(self.results.position)
        return self.results
