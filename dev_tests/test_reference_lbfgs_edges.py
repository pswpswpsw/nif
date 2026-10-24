"""Public API/analytic edge cases for the scalar reference optimizer."""
import copy

import pytest
import torch

from nif.optimizers import ReferenceLBFGS
from nif.optimizers.lbfgs_V2 import lbfgs_minimize
from nif.optimizers._hager_zhang import Point, hager_zhang


OPTIONS = dict(num_correction_pairs=20, tolerance=1e-15, x_tolerance=1e-15,
               f_relative_tolerance=1e-15, f_absolute_tolerance=0.,
               max_line_search_iterations=100)


def test_adding_parameter_group_is_rejected_before_mutation():
    first = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
    second = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
    optimizer = ReferenceLBFGS([first], max_iter=1, **OPTIONS)
    with pytest.raises(ValueError, match='one parameter group'):
        optimizer.add_param_group({'params': [second]})
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]['params'] == [first]


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_exhausted_unbracketed_interval_bisects_before_classifying(dtype):
    coefficient = torch.tensor(1e-8, dtype=dtype)
    def line(step):
        position = torch.ones(1, dtype=dtype) - step * coefficient
        value, gradient = .5 * coefficient * position.square().sum(), coefficient * position
        return Point(step, value, (gradient * -coefficient).sum(), gradient)
    result = hager_zhang(line, line(torch.zeros((), dtype=dtype)), max_iterations=1)
    # Actual TFP always runs U3 after an exhausted, still-unbracketed expansion.
    # Floating-point stagnation fails U3 while adjacent endpoints also mark
    # convergence. These two search flags can both be true; the optimizer uses
    # `not search.converged`, not search.failed, for its own failure condition.
    assert result.converged and result.failed
    assert result.iterations == 1 and result.evaluations > 3
    assert result.left.x < result.right.x
    assert torch.nextafter(result.left.x, result.right.x) >= result.right.x


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_initial_gradient_boundary_is_strict_and_finite_zero_budget_is_not_failure(dtype):
    def objective(x):
        return .5 * x.square().sum(), x.clone()
    result = lbfgs_minimize(objective, initial_position=torch.ones(1, dtype=dtype),
                           max_iterations=0, **dict(OPTIONS, tolerance=1.))
    assert not result.converged and not result.failed
    assert result.num_iterations == 0 and result.num_objective_evaluations == 1
    inside = lbfgs_minimize(objective, initial_position=torch.full((1,), .5, dtype=dtype),
                           max_iterations=0, **dict(OPTIONS, tolerance=1.))
    assert inside.converged and not inside.failed


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_public_rosenbrock_curvature_and_line_search_control(dtype):
    def objective(x):
        a, b = x[0], x[1]
        residual = b-a*a
        return (1-a)**2 + 100*residual**2, torch.stack([-2*(1-a)-400*a*residual, 200*residual])
    result = lbfgs_minimize(objective, initial_position=torch.tensor([-1.2, 1.], dtype=dtype),
                           max_iterations=1000, **OPTIONS)
    assert result.converged and not result.failed
    # Analytic optimum is zero. This uses dtype precision, not a recorded
    # backend trajectory/evaluation count or a scientific training threshold.
    assert result.objective_value.abs() <= 128 * torch.finfo(dtype).eps
    assert torch.isfinite(result.position).all()


def test_incoming_tensor_dtype_and_version_are_validated_before_base_loader_cast():
    point = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
    optimizer = ReferenceLBFGS([point], max_iter=0, **OPTIONS)
    def closure():
        optimizer.zero_grad(set_to_none=True)
        value = .5 * point.square().sum()
        value.backward()
        return value
    optimizer.step(closure)
    state = optimizer.state_dict()
    bad = copy.deepcopy(state)
    identifier = bad['param_groups'][0]['params'][0]
    bad['state'][identifier]['position'] = bad['state'][identifier]['position'].float()
    with pytest.raises(ValueError, match='tensor mismatch'):
        optimizer.load_state_dict(bad)
    bad = copy.deepcopy(state)
    bad['reference_lbfgs']['schema'] = 'unrecognized-version'
    with pytest.raises(ValueError, match='schema'):
        optimizer.load_state_dict(bad)
