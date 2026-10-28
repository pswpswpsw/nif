"""Source-derived AdaBelief scalar-precision controls, with no TF dependency.

Reference: ad9113df14590189b12068b6b3fa7c8c8d9187d4, dense/sparse
_resource_apply methods in nif/optimizers/external_optimizers.py. Numpy scalar
operations explicitly retain the parameter dtype and original expression order.
Actual float32-declared betas are promoted explicitly for float64 controls; this
separates scalar calculation precision from hyperparameter representation.
"""
import io

import numpy as np
import pytest
import torch

from nif.optimizers import AdaBeliefOptimizer

BETAS = tuple(float(np.float32(b)) for b in (0.9, 0.999))


def coefficients(dtype, step, betas=BETAS):
    scalar = np.float32 if dtype == torch.float32 else np.float64
    one, t = scalar(1), scalar(step)
    b1, b2 = map(scalar, betas)
    power1, power2 = np.power(b1, t), np.power(b2, t)
    maximum = scalar(2) / (one - b2) - one
    rho = maximum - scalar(2) * t * power2 / (one - power2)
    with np.errstate(invalid="ignore"):
        r = np.sqrt(
            (rho - scalar(4))
            / (maximum - scalar(4))
            * (rho - scalar(2))
            / (maximum - scalar(2))
            * maximum
            / rho
        )
    return one - power1, one - power2, rho, r


def direction(m, v, dtype, step, rectify, betas=BETAS, epsilon=1e-14):
    c1, c2, rho, r = coefficients(dtype, step, betas)
    corrected = m / float(c1)
    if not rectify:
        return corrected / ((v / float(c2)).sqrt() + epsilon)
    if rho >= 5:
        return float(r) * corrected / ((v / float(c2)).sqrt() + epsilon)
    return corrected


def test_source_float32_transition_scalar_values():
    _, _, rho, r = coefficients(torch.float32, 6)
    assert float(rho) == 6.00048828125
    assert float(r) == 0.02586868219077587
    _, _, rho64, r64 = coefficients(torch.float64, 6)
    assert float(rho64) == pytest.approx(5.994163826821023, abs=1e-10)
    assert float(r64) == pytest.approx(0.025820946567591475, abs=1e-12)
    assert coefficients(torch.float32, 5)[2] < 5
    assert coefficients(torch.float32, 6)[2] >= 5


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("rectify", [False, True])
@pytest.mark.parametrize("amsgrad", [False, True])
@pytest.mark.parametrize("step", [1, 5, 6, 7, 20, 50])
def test_single_update_uses_parameter_dtype_scalars(dtype, rectify, amsgrad, step):
    # Zero parameter values expose the update itself without cancellation against
    # an arbitrary large initial value; include zero/tiny/ordinary gradients.
    p = torch.nn.Parameter(torch.zeros(4, dtype=dtype))
    optimizer = AdaBeliefOptimizer(
        [p], lr=0.001, betas=BETAS, rectify=rectify, amsgrad=amsgrad
    )
    optimizer.param_groups[0]["step"] = step - 1
    m = (
        torch.tensor([0.0, 2e-13, 0.15, -0.3], dtype=dtype)
        if step > 1
        else torch.zeros_like(p)
    )
    v = (
        torch.tensor([0.0, 1e-25, 0.03, 0.005], dtype=dtype)
        if step > 1
        else torch.zeros_like(p)
    )
    maximum = v * 2
    optimizer.state[p].update(exp_avg=m.clone(), exp_avg_var=v.clone(), step=step - 1)
    if amsgrad:
        optimizer.state[p]["max_exp_avg_var"] = maximum.clone()
    g = torch.tensor([0.0, 1e-12, 0.2, -0.4], dtype=dtype)
    p.grad = g
    b1, b2 = BETAS
    expected_m = b1 * m + (1 - b1) * g
    expected_v = b2 * v + (1 - b2) * (g - expected_m).square() + 1e-14
    if amsgrad:
        maximum = torch.maximum(maximum, expected_v)
    expected = -0.001 * direction(
        expected_m, maximum if amsgrad else expected_v, dtype, step, rectify
    )
    optimizer.step()
    tolerance = 4e-7 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(
        optimizer.state[p]["exp_avg"], expected_m, rtol=tolerance, atol=0
    )
    torch.testing.assert_close(
        optimizer.state[p]["exp_avg_var"], expected_v, rtol=tolerance, atol=0
    )
    if amsgrad:
        torch.testing.assert_close(
            optimizer.state[p]["max_exp_avg_var"], maximum, rtol=tolerance, atol=0
        )
    torch.testing.assert_close(p, expected, rtol=tolerance, atol=0)
    assert p[0] == 0 and torch.isfinite(p).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("rectify", [False, True])
def test_default_beta_bias_correction_from_measured_moments(dtype, rectify):
    # Keep the already measured native moment recurrence unchanged. This control
    # isolates only scalar bias/rectification calculations at declared defaults.
    p = torch.nn.Parameter(torch.zeros(3, dtype=dtype))
    opt = AdaBeliefOptimizer([p], lr=0.001, rectify=rectify)
    for step in range(1, 13):
        p.grad = torch.tensor([0.1, -0.3, 0.0], dtype=dtype)
        with torch.no_grad():
            p.zero_()  # expose each individual update
        opt.step()
        state = opt.state[p]
        expected = -0.001 * direction(
            state["exp_avg"],
            state["exp_avg_var"],
            dtype,
            step,
            rectify,
            betas=(0.9, 0.999),
        )
        torch.testing.assert_close(
            p, expected, rtol=4e-7 if dtype == torch.float32 else 2e-12, atol=0
        )


def test_mixed_dtype_groups_schedules_and_checkpoint_continuation():
    parameters = [
        torch.nn.Parameter(torch.zeros(3, dtype=d))
        for d in (torch.float32, torch.float64) * 2
    ]
    opt = AdaBeliefOptimizer(
        [{"params": parameters[:2]}, {"params": parameters[2:], "amsgrad": True}],
        betas=BETAS,
    )
    control = None
    resumed = None
    for step in range(1, 18):
        for i, group in enumerate(opt.param_groups):
            group["lr"] = 0.001 * (i + 1) / (step + 1)
        if control is not None:
            for a, b in zip(control.param_groups, opt.param_groups):
                a["lr"] = b["lr"]
        before = [p.detach().clone() for p in parameters]
        for i, p in enumerate(parameters):
            p.grad = torch.full_like(p, (i + 1) * 0.01)
        if control is not None:
            for p, q in zip(parameters, resumed):
                q.grad = p.grad.clone()
        opt.step()
        for i, p in enumerate(parameters):
            state = opt.state[p]
            v = state.get("max_exp_avg_var", state["exp_avg_var"])
            expected = before[i] - 0.001 * (i // 2 + 1) / (step + 1) * direction(
                state["exp_avg"], v, p.dtype, step, True
            )
            torch.testing.assert_close(
                p, expected, rtol=4e-7 if p.dtype == torch.float32 else 2e-12, atol=0
            )
        if control is not None:
            control.step()
            for p, q in zip(parameters, resumed):
                torch.testing.assert_close(p, q, rtol=0, atol=0)
                for key, value in opt.state[p].items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(
                            value, control.state[q][key], rtol=0, atol=0
                        )
                    else:
                        assert value == control.state[q][key]
        if step in (5, 8):
            buffer = io.BytesIO()
            torch.save(opt.state_dict(), buffer)
            buffer.seek(0)
            resumed = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
            control = AdaBeliefOptimizer(
                [{"params": resumed[:2]}, {"params": resumed[2:]}]
            )
            control.load_state_dict(torch.load(buffer, weights_only=True))


def test_sparse_observed_rows_share_corrected_scalars_and_keep_other_rows():
    p = torch.nn.Parameter(torch.zeros(3, 2, dtype=torch.float32))
    opt = AdaBeliefOptimizer([p], betas=BETAS, amsgrad=True)
    for step in range(1, 11):
        before = p.detach().clone()
        row = step % 2
        p.grad = torch.sparse_coo_tensor(
            torch.tensor([[row]]), torch.tensor([[0.1, -0.2]]), p.shape
        )
        opt.step()
        state = opt.state[p]
        expected = before - 0.001 * direction(
            state["exp_avg"], state["max_exp_avg_var"], p.dtype, step, True
        )
        torch.testing.assert_close(p[row], expected[row], rtol=4e-7, atol=0)
        torch.testing.assert_close(p[2], before[2], rtol=0, atol=0)
        torch.testing.assert_close(p[1 - row], before[1 - row], rtol=0, atol=0)
