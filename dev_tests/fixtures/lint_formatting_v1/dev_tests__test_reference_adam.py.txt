"""Public epsilon-hat regressions: direct declared-precision scalar recurrence.

The fallback deliberately exercises the existing notebook optimizer on RED.
No TensorFlow runtime or evaluator fixtures are used.
"""
import copy
import numpy as np
import pytest
import torch
import nif.optimizers as optimizers

Adam = getattr(optimizers, 'ReferenceAdam', torch.optim.Adam)


@pytest.mark.parametrize('dtype,scalar', [(torch.float64, np.float64), (torch.float32, np.float32)])
@pytest.mark.parametrize('betas', [(0.9, 0.999), (0.5, 0.75)])
@pytest.mark.parametrize('scale', [0., 1e-12, 0.2])
def test_epsilon_hat_recurrence(dtype, scalar, betas, scale):
    # Binary-exact betas isolate algebra from hyperparameter representation.
    # Decimal betas are explicitly represented at the parameter's precision;
    # backend float32-to-float64 promotion is a separate rounding diagnostic.
    p = torch.nn.Parameter(torch.zeros(3, dtype=dtype))
    opt = Adam([p], lr=1e-3, betas=betas, eps=1e-7)
    expected = np.zeros(3, dtype=scalar)
    m, v = expected.copy(), expected.copy()
    b1, b2 = map(scalar, betas)
    for step in range(1, 11):
        g = np.array([scale, -scale * step, scale * (step % 3)], dtype=scalar)
        p.grad = torch.tensor(g, dtype=dtype)
        opt.step()
        m = b1*m + (scalar(1)-b1)*g
        v = b2*v + (scalar(1)-b2)*g*g
        alpha = scalar(1e-3)*np.sqrt(scalar(1)-b2**scalar(step))/(scalar(1)-b1**scalar(step))
        expected -= alpha*m/(np.sqrt(v)+scalar(1e-7))
        tol = 3e-5 if dtype == torch.float32 else 2e-13
        torch.testing.assert_close(p, torch.tensor(expected), rtol=tol, atol=1e-25)
        torch.testing.assert_close(opt.state[p]['exp_avg'], torch.tensor(m), rtol=tol, atol=1e-25)
        torch.testing.assert_close(opt.state[p]['exp_avg_sq'], torch.tensor(v), rtol=tol, atol=1e-30)


def test_float32_promoted_betas_are_separate_from_epsilon_placement():
    # Match actual float32-declared beta values exactly in a float64 diagnostic.
    b1, b2 = (float(np.float32(b)) for b in (.9, .999))
    p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    opt = Adam([p], betas=(b1, b2), lr=1e-3, eps=1e-7)
    p.grad = torch.tensor([1e-12], dtype=p.dtype)
    opt.step()
    m, v = (1-b1)*1e-12, (1-b2)*1e-24
    expected = -1e-3*np.sqrt(1-b2)/(1-b1)*m/(np.sqrt(v)+1e-7)
    assert p.item() == pytest.approx(expected, rel=1e-13, abs=0.)


def test_groups_closure_and_resume():
    p = torch.nn.Parameter(torch.tensor([.3], dtype=torch.float64))
    q = torch.nn.Parameter(torch.tensor([-.2], dtype=torch.float32))
    opt = Adam([{'params':[p], 'lr':.002}, {'params':[q], 'lr':.001, 'eps':2e-7}])
    def closure():
        opt.zero_grad()
        loss = p.square().sum() + q.square().sum()
        loss.backward()
        return loss
    loss = opt.step(closure)
    assert loss.requires_grad
    state = copy.deepcopy(opt.state_dict())
    ps, qs = torch.nn.Parameter(p.detach().clone()), torch.nn.Parameter(q.detach().clone())
    resumed = Adam([{'params':[ps]}, {'params':[qs]}])
    resumed.load_state_dict(state)
    for a, b in ((p, ps), (q, qs)):
        a.grad = torch.full_like(a, 1e-10)
        b.grad = a.grad.clone()
    opt.step(); resumed.step()
    torch.testing.assert_close(ps, p, rtol=0, atol=0)
    torch.testing.assert_close(qs, q, rtol=0, atol=0)
