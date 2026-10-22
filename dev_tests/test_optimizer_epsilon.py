"""Public baseline epsilon fallback and native alias constructor semantics."""
import copy

import pytest
import torch

from nif.optimizers import AdaBeliefOptimizer, L4Adam


OPTIMIZERS = (AdaBeliefOptimizer, L4Adam)


def make_optimizer(cls, options):
    p = torch.nn.Parameter(torch.tensor([1., -2., 3.], dtype=torch.float64))
    return p, cls([p], **options)


def advance(p, optimizer, grad, loss=2.):
    p.grad = torch.tensor(grad, dtype=p.dtype)
    if isinstance(optimizer, L4Adam):
        optimizer.step(loss=loss)
    else:
        optimizer.step()


def assert_state_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_state_equal(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize('cls', OPTIMIZERS)
@pytest.mark.parametrize('keyword', ['epsilon', 'eps'])
@pytest.mark.parametrize('value', [None, 0.])
def test_none_and_zero_epsilon_keep_zero_gradient_parameters_finite(cls, keyword, value):
    p, optimizer = make_optimizer(cls, {keyword: value})
    initial = p.detach().clone()
    # AdaBelief's default rectification reaches its variance denominator after
    # the first few steps, exposing zero/zero rather than just constructor state.
    for _ in range(10):
        advance(p, optimizer, [0., 0., 0.])
        assert torch.isfinite(p).all()
        torch.testing.assert_close(p, initial, rtol=0, atol=0)
    assert optimizer.defaults['eps'] == 1e-7
    assert optimizer.param_groups[0]['eps'] == 1e-7


@pytest.mark.parametrize('cls', OPTIMIZERS)
@pytest.mark.parametrize('keyword', ['epsilon', 'eps'])
@pytest.mark.parametrize('value', [None, 0.])
def test_epsilon_fallback_matches_explicit_backend_default_and_resume(cls, keyword, value):
    p, optimizer = make_optimizer(cls, {keyword: value})
    q, reference = make_optimizer(cls, {'epsilon': 1e-7})
    for step in range(12):
        grad = [.1 * (step + 1), -.3, 0.]
        advance(p, optimizer, grad, loss=3. - step * .1)
        advance(q, reference, grad, loss=3. - step * .1)
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        assert_state_equal(optimizer.state_dict(), reference.state_dict())
        if step == 5:
            # Check that the resolved epsilon survives serialization, including
            # resuming into a constructor whose default may be different.
            state = copy.deepcopy(optimizer.state_dict())
            optimizer = cls([p])
            optimizer.load_state_dict(state)


@pytest.mark.parametrize('cls', OPTIMIZERS)
@pytest.mark.parametrize('value,expected', [(None, 1e-7), (0., 1e-7), (2e-5, 2e-5)])
def test_explicit_native_eps_alias_overrides_canonical_epsilon(cls, value, expected):
    _, optimizer = make_optimizer(cls, {'epsilon': 3e-6, 'eps': value})
    assert optimizer.defaults['eps'] == expected


@pytest.mark.parametrize('cls,default', [(AdaBeliefOptimizer, 1e-14), (L4Adam, 1e-7)])
def test_omitted_epsilon_retains_original_positive_default(cls, default):
    p, optimizer = make_optimizer(cls, {})
    q, explicit = make_optimizer(cls, {'epsilon': default})
    assert optimizer.defaults['eps'] == default
    for step in range(12):
        advance(p, optimizer, [.2, -.4, 0.], loss=2. + step * .1)
        advance(q, explicit, [.2, -.4, 0.], loss=2. + step * .1)
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        assert_state_equal(optimizer.state_dict(), explicit.state_dict())


@pytest.mark.parametrize('cls', OPTIMIZERS)
@pytest.mark.parametrize('keyword', ['epsilon', 'eps'])
def test_negative_epsilon_still_rejected(cls, keyword):
    with pytest.raises(ValueError, match='epsilon'):
        make_optimizer(cls, {keyword: -1e-7})
