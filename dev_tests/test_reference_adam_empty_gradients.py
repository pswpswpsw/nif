"""Public all-None boundary regressions; no independent evaluator fixtures."""
import copy

import pytest
import torch

from nif.optimizers import ReferenceAdam


def assert_identical(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert actual.dtype == expected.dtype and actual.device == expected.device
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_identical(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_identical(a, b)
    else:
        assert actual == expected


def make_pair():
    p = torch.nn.Parameter(torch.tensor([0.25, -0.5], dtype=torch.float64))
    q = torch.nn.Parameter(torch.tensor([0.125], dtype=torch.float32))
    optimizer = ReferenceAdam(
        [{"params": [p]}, {"params": [q], "lr": 0.002}], amsgrad=True
    )
    return p, q, optimizer


@pytest.mark.parametrize("warmup", [False, True])
@pytest.mark.parametrize("use_closure", [False, True])
def test_all_none_raises_without_mutating_parameters_slots_or_iteration(
    warmup, use_closure
):
    p, q, optimizer = make_pair()
    if warmup:
        p.grad = torch.full_like(p, 1e-10)
        q.grad = torch.ones_like(q)
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    parameters = [p.detach().clone(), q.detach().clone()]
    state = copy.deepcopy(optimizer.state_dict())
    calls = []

    def closure():
        # Guard must inspect gradients AFTER the closure clears them.
        assert torch.is_grad_enabled()
        optimizer.zero_grad(set_to_none=True)
        calls.append(True)
        return p.square().sum()

    if use_closure:
        p.grad = torch.ones_like(p)
    with pytest.raises(ValueError, match="[Nn]o gradients"):
        optimizer.step(closure if use_closure else None)
    assert calls == ([True] if use_closure else [])
    assert_identical([p, q], parameters)
    assert_identical(optimizer.state_dict(), state)


def test_rejected_empty_call_cannot_shift_next_bias_correction():
    p, q, optimizer = make_pair()
    a, b, control = make_pair()
    for param in (p, q, a, b):
        param.grad = torch.full_like(param, 1e-12)
    optimizer.step()
    control.step()
    optimizer.zero_grad(set_to_none=True)
    try:
        optimizer.step()
    except ValueError:
        pass
    for param in (p, q, a, b):
        param.grad = torch.full_like(param, -2e-12)
    optimizer.step()
    control.step()
    assert_identical([p, q], [a, b])
    assert_identical(optimizer.state_dict(), control.state_dict())


def test_partial_none_still_advances_shared_iteration_across_groups():
    p, q, optimizer = make_pair()
    p.grad = torch.ones_like(p)
    optimizer.step()
    p_state = copy.deepcopy(optimizer.state[p])
    p_value = p.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    q.grad = torch.ones_like(q)
    optimizer.step()
    assert optimizer.state_dict()["reference_iterations"] == 2
    assert_identical(p, p_value)
    assert_identical(optimizer.state[p], p_state)
    # q first receives a gradient on GLOBAL iteration 2, not local iteration 1.
    beta1, beta2 = (q.new_tensor(x) for x in (0.9, 0.999))
    alpha = q.new_tensor(0.002) * (1 - beta2**2).sqrt() / (1 - beta1**2)
    expected = q.new_tensor([0.125]) - alpha * q.new_tensor(0.1) / (
        q.new_tensor(0.001).sqrt() + 1e-7
    )
    assert_identical(q, expected)


def test_real_zero_gradient_is_an_update_including_existing_momentum():
    p, q, optimizer = make_pair()
    p.grad = torch.zeros_like(p)
    optimizer.step()
    assert optimizer.state_dict()["reference_iterations"] == 1
    assert p in optimizer.state and q not in optimizer.state
    p.grad.fill_(1.0)
    optimizer.step()
    before = p.detach().clone()
    p.grad.zero_()
    optimizer.step()
    assert optimizer.state_dict()["reference_iterations"] == 3
    assert not torch.equal(p, before)  # zero gradient still applies momentum


def test_closure_can_supply_the_first_real_gradient():
    p, q, optimizer = make_pair()

    def closure():
        loss = p.square().sum()
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    assert loss.requires_grad
    assert optimizer.state_dict()["reference_iterations"] == 1
    assert p in optimizer.state and q not in optimizer.state
