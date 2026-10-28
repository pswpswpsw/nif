"""Frozen phase-0 recipe. Activate these SAME bytes only with implementation approval.

Public analytic CPU contracts; no NIF data/model/training or cross-backend bit gate.
ReferenceLBFGS uses max_iter ADDITIONAL iterations per step and cumulative result
counters; generic lbfgs_minimize uses a cumulative max_iterations limit.
"""
import io

import pytest
import torch

from nif.optimizers import ReferenceLBFGS
from nif.optimizers.lbfgs_V2 import lbfgs_minimize

OPTIONS = dict(num_correction_pairs=20, tolerance=1e-15, x_tolerance=1e-15,
               f_relative_tolerance=1e-15, f_absolute_tolerance=0.,
               max_line_search_iterations=100)


def run_parameter(dtype, objective, *, max_iter=1000, **overrides):
    p = torch.nn.Parameter(torch.ones(1, dtype=dtype, device='cpu'))
    optimizer = ReferenceLBFGS([p], max_iter=max_iter, **dict(OPTIONS, **overrides))
    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = objective(p)
        loss.backward()
        return loss
    optimizer.step(closure)
    return p, optimizer


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('scale', [1., 1e-8])
def test_public_scalar_quadratics_progress_and_gradient_stop(dtype, scale):
    p, optimizer = run_parameter(dtype, lambda x: .5 * scale * x.square().sum())
    result = optimizer.result
    assert .5 * scale * p.detach().square().sum() < .5 * scale
    assert result.converged and not result.failed
    assert result.objective_gradient.abs().max() <= OPTIONS['tolerance']
    assert torch.equal(result.position, p.detach().reshape(-1))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_flat_offset_quartic_stops_with_nonzero_gradient(dtype):
    p, optimizer = run_parameter(dtype, lambda x: 2. ** 20 + .125 * (x ** 4).sum())
    result = optimizer.result
    assert result.converged and not result.failed
    assert result.objective_gradient.abs().max() > OPTIONS['tolerance']
    assert 0 < result.num_iterations < 1000
    assert 0 < p.detach().item() < 1
    assert result.objective_value.item() == 2. ** 20


def diagonal_objective(x):
    diagonal = x.new_tensor([1., 10.])
    return .5 * (diagonal * x.square()).sum(), diagonal * x


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_generic_resume_is_cumulative_and_finite_limit_is_not_failure(dtype):
    initial = torch.ones(2, dtype=dtype)
    first = lbfgs_minimize(diagonal_objective, initial_position=initial,
                           max_iterations=1, **OPTIONS)
    resumed = lbfgs_minimize(diagonal_objective, previous_optimizer_results=first,
                             max_iterations=2, **OPTIONS)
    full = lbfgs_minimize(diagonal_objective, initial_position=initial,
                          max_iterations=2, **OPTIONS)
    assert first.num_iterations == 1 and resumed.num_iterations == full.num_iterations == 2
    assert not first.converged and not first.failed
    assert not resumed.converged and not resumed.failed
    assert resumed.num_objective_evaluations == full.num_objective_evaluations
    for name in ('position', 'objective_value', 'objective_gradient'):
        assert torch.equal(getattr(resumed, name), getattr(full, name))
    # Exhausted cumulative budget performs no extra evaluation or iteration.
    unchanged = lbfgs_minimize(diagonal_objective, previous_optimizer_results=resumed,
                               max_iterations=2, **OPTIONS)
    assert unchanged.num_iterations == resumed.num_iterations
    assert unchanged.num_objective_evaluations == resumed.num_objective_evaluations
    assert torch.equal(unchanged.position, resumed.position)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_signed_old_objective_in_relative_stopping(dtype):
    results = []
    for offset in (100., -100.):
        def objective(x):
            value, gradient = diagonal_objective(x)
            return value + offset, gradient
        results.append(lbfgs_minimize(objective, initial_position=torch.ones(2, dtype=dtype),
            max_iterations=1, **dict(OPTIONS, tolerance=0., x_tolerance=0., f_relative_tolerance=.1)))
    assert results[0].converged and not results[0].failed
    assert not results[1].converged and not results[1].failed


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_optimizer_checkpoint_replays_same_native_trajectory(dtype):
    p = torch.nn.Parameter(torch.ones(2, dtype=dtype))
    first = ReferenceLBFGS([p], max_iter=1, **OPTIONS)
    def closure(optimizer, parameter):
        def evaluate():
            optimizer.zero_grad(set_to_none=True)
            value, _ = diagonal_objective(parameter)
            value.backward()
            return value
        return evaluate
    first.step(closure(first, p))
    buffer = io.BytesIO()
    torch.save({'parameter': p.detach().clone(), 'optimizer': first.state_dict()}, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    q = torch.nn.Parameter(saved['parameter'].clone())
    restored = ReferenceLBFGS([q], max_iter=1, **OPTIONS)
    restored.load_state_dict(saved['optimizer'])
    restored.step(closure(restored, q))
    direct_p = torch.nn.Parameter(torch.ones(2, dtype=dtype))
    direct = ReferenceLBFGS([direct_p], max_iter=2, **OPTIONS)
    direct.step(closure(direct, direct_p))
    assert torch.equal(q, direct_p)
    assert restored.result.num_iterations == direct.result.num_iterations == 2
    assert restored.result.num_objective_evaluations == direct.result.num_objective_evaluations


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('value,gradient,converged,failed', [
    (float('inf'), 1., False, True),
    (-float('inf'), 1., True, False),
    (1., float('nan'), False, True),
])
def test_reference_nonfinite_flags(dtype, value, gradient, converged, failed):
    def objective(x):
        return x.new_tensor(value), torch.full_like(x, gradient)
    result = lbfgs_minimize(objective, initial_position=torch.zeros(1, dtype=dtype),
                             max_iterations=1000, **OPTIONS)
    assert bool(result.converged) == converged
    assert bool(result.failed) == failed


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_unbounded_linear_failure_and_negative_infinity_policy(dtype):
    def objective(x):
        return x.sum(), torch.ones_like(x)
    result = lbfgs_minimize(objective, initial_position=torch.zeros(1, dtype=dtype),
                             max_iterations=1000, **OPTIONS)
    # These flags are measured from actual TFP. Do not assert its invalid trial
    # position bits or exact backend overflow/line-search evaluation count.
    assert bool(result.converged) == (dtype == torch.float32)
    assert bool(result.failed) == (dtype == torch.float64)


def test_reject_torch_state_and_mismatched_checkpoint_before_casting():
    p = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    reference = ReferenceLBFGS([p], max_iter=1, **OPTIONS)
    legacy = torch.optim.LBFGS([p]).state_dict()
    with pytest.raises((ValueError, TypeError)):
        reference.load_state_dict(legacy)
    reference.step(lambda: _backward_diagonal(reference, p))
    state = reference.state_dict()
    for q in (torch.nn.Parameter(torch.ones(2, dtype=torch.float32)),
              torch.nn.Parameter(torch.ones(1, 2, dtype=torch.float64))):
        target = ReferenceLBFGS([q], max_iter=1, **OPTIONS)
        with pytest.raises((ValueError, TypeError)):
            target.load_state_dict(state)


def _backward_diagonal(optimizer, parameter):
    optimizer.zero_grad(set_to_none=True)
    value, _ = diagonal_objective(parameter)
    value.backward()
    return value


def test_closure_exception_restores_last_accepted_parameters():
    p = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
    optimizer = ReferenceLBFGS([p], max_iter=1000, **OPTIONS)
    evaluations = 0
    def closure():
        nonlocal evaluations
        evaluations += 1
        optimizer.zero_grad(set_to_none=True)
        if evaluations > 1:
            raise RuntimeError('public trial exception')
        value = .5 * p.square().sum()
        value.backward()
        return value
    with pytest.raises(RuntimeError, match='public trial exception'):
        optimizer.step(closure)
    assert evaluations == 2
    assert torch.equal(p.detach(), torch.ones_like(p))


def test_reject_complex_mixed_dtype_noncontiguous_and_sparse_gradient():
    for parameters in (
        [torch.nn.Parameter(torch.ones(2, dtype=torch.complex64))],
        [torch.nn.Parameter(torch.ones(1, dtype=torch.float32)),
         torch.nn.Parameter(torch.ones(1, dtype=torch.float64))],
        [torch.nn.Parameter(torch.ones(2, 2).transpose(0, 1))],
    ):
        with pytest.raises((ValueError, TypeError)):
            ReferenceLBFGS(parameters, max_iter=1, **OPTIONS)
    p = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    optimizer = ReferenceLBFGS([p], max_iter=1, **OPTIONS)
    def closure():
        p.grad = torch.sparse_coo_tensor([[0]], [1.], size=(2,), dtype=p.dtype)
        return p.detach().sum()
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        optimizer.step(closure)
