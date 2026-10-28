import math

import pytest
import torch

from nif.layers import BiasAddLayer
from nif.layers import EinsumLayer
from nif.layers import HessianLayer
from nif.layers import HyperLinearForSIREN
from nif.layers import JacobianLayer
from nif.layers import JacRegLatentLayer
from nif.layers import MLP_ResNet
from nif.layers import MLP_SimpleShortCut
from nif.layers import ParameterOutputL1ActReg
from nif.layers import SIREN
from nif.layers import SIREN_ResNet
from nif.layers.siren import gen_hypernetwork_weights_bias_for_siren_shapenet


def test_siren_formula_initialization_precision_and_residual():
    layer = SIREN(3, 3, "first", 7.0, mixed_policy="float64")
    assert layer.w.dtype == torch.float64
    assert layer.w.abs().max() <= 1 / 3
    assert layer.b.abs().max() <= 1 / math.sqrt(3)
    x = torch.randn(4, 3, dtype=torch.float64)
    assert torch.allclose(layer(x), torch.sin(7 * x @ layer.w + layer.b))
    res = SIREN_ResNet(3, 3, 7.0, mixed_policy="float64")
    assert torch.equal(res.w, res.w2)
    assert res.w is not res.w2
    h = torch.sin(7 * x @ res.w + res.b)
    assert torch.allclose(res(x), 0.5 * (x + torch.sin(7 * h @ res.w2 + res.b2)))
    assert layer.float()(x.float()).dtype == torch.float32


def test_hyper_initialization_sections():
    w, b = gen_hypernetwork_weights_bias_for_siren_shapenet(
        20, 46, 0.02, 6, 18, 6, 2, 3, 10.0, torch.float64
    )
    assert w.dtype == b.dtype == torch.float64
    assert w.abs().max() <= math.sqrt(6 / 20) * 0.02
    for section, bound in [
        (b[:6], 0.5),
        (b[6:24], math.sqrt(2) / 10),
        (b[24:30], 1.0),
        (b[30:], 1 / 3),
    ]:
        assert section.abs().max() <= bound


class Polynomial(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))

    def forward(self, x):
        return torch.stack((self.a * x[:, 0] ** 3 + x[:, 1], x[:, 0] * x[:, 1]), -1)


def test_hessian_full_output_selection_and_parameter_gradient():
    model = Polynomial()
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64, requires_grad=True)
    y, j, h = HessianLayer(model, [0], [0, 1])(x)
    assert y.shape == (2, 2) and j.shape == (2, 1, 2) and h.shape == (2, 1, 2, 2)
    assert torch.allclose(j[:, 0, 0], 6 * x[:, 0] ** 2)
    assert torch.allclose(h[:, 0, 0, 0], 12 * x[:, 0])
    assert torch.allclose(h[:, 0, 1, :], torch.zeros_like(h[:, 0, 1, :]))
    (j.square().mean() + h.square().mean()).backward()
    assert model.a.grad is not None and model.a.grad > 0
    assert x.grad is not None
    _, ji = JacobianLayer(model, 1, 0)(x.detach())
    assert ji.shape == (2, 1, 1)
    assert torch.equal(ji[:, 0, 0], x[:, 1])


@pytest.mark.parametrize("constant", [False, True])
def test_zero_higher_derivatives_have_valid_backward(constant):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, x):
            return self.a.expand(x.shape[0], 1) if constant else self.a * x[:, :1]

    m = Model()
    x = torch.randn(3, 2)
    _, j, h = HessianLayer(m, [0], [0, 1])(x)
    assert torch.equal(h, torch.zeros_like(h))
    h.sum().backward()
    assert m.a.grad is not None and m.a.grad == 0
    if constant:
        assert torch.equal(j, torch.zeros_like(j))


def test_explicit_regularization_reductions_no_cache():
    class Augment(torch.nn.Module):
        def forward(self, x):
            return x * 2, x**2

    m = Augment()
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    p = ParameterOutputL1ActReg(m, l1=0.3)
    assert torch.equal(p(x), 2 * x)
    assert torch.allclose(p.regularization_loss(x), 0.3 * x.square().abs().sum())
    jac = JacRegLatentLayer(m, [0, 1], [0, 1], l1=0.7)
    assert torch.equal(jac(x), 2 * x)
    expected = 0.7 * (2 * x).square().sum() / (x.shape[0] * 4)
    assert torch.allclose(jac.regularization_loss(x), expected)
    with pytest.raises(ValueError):
        jac.regularization_loss()
    with pytest.raises(ValueError):
        p.regularization_loss()
    jac.regularization_loss(x).backward()
    assert x.grad is not None


def test_mlp_formulas_and_utility_layers():
    x = torch.randn(4, 3, dtype=torch.float64)
    kw = dict(
        width=3,
        activation="tanh",
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        mixed_policy="float64",
    )
    r = MLP_ResNet(**kw)
    assert torch.allclose(r(x), torch.tanh(x + r.L2(r.L1(x))))
    s = MLP_SimpleShortCut(**kw)
    assert torch.allclose(s(x), x + s.L1(x))
    b = BiasAddLayer(3, "float64")
    assert torch.equal(b(x), x + b.last_layer_bias)
    e = EinsumLayer("bi,bi->b")
    assert torch.allclose(e((x, x)), x.square().sum(-1))


def test_initialization_distribution_and_hyper_regularization():
    torch.manual_seed(9127)
    layer = SIREN(128, 1024, "hidden", 11.0)
    expected = (math.sqrt(6 / 128) / 11) ** 2 / 3
    assert abs(layer.w.var(unbiased=False).item() / expected - 1) < 0.025
    cfg = dict(
        input_dim=2,
        output_dim=1,
        units=3,
        nlayers=1,
        use_resblock=False,
        connectivity="full",
        weight_init_factor=0.02,
        omega_0=10.0,
    )
    # 6 first + 9 hidden + 3 last weights, plus 3 + 3 + 1 biases.
    h = HyperLinearForSIREN(
        4,
        25,
        cfg,
        "float64",
        kernel_regularizer=lambda z: 0.2 * z.square().sum(),
        bias_regularizer=lambda z: 0.3 * z.abs().sum(),
        activity_regularizer=lambda z: 0.4 * z.abs().sum(),
    )
    x = torch.randn(5, 4, dtype=torch.float64)
    expected = (
        0.2 * h.w.square().sum() + 0.3 * h.b.abs().sum() + 0.4 * h(x).abs().sum() / 5
    )
    assert torch.allclose(h.regularization_loss(x), expected)
    with pytest.raises(ValueError):
        h.regularization_loss()
    before = h.regularization_loss(x).detach().clone()
    h(torch.randn(17, 4, dtype=torch.float64))
    assert torch.equal(h.regularization_loss(x).detach(), before)
    h.regularization_loss(x).backward()
    assert torch.isfinite(h.w.grad).all()
    assert h.w._nif_gc_axes == (0,)


def test_siren_parameter_and_input_hessian_gradcheck():
    model = SIREN(2, 2, "first", 3.0, mixed_policy="float64")
    x = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
    wrapper = HessianLayer(model, [0, 1], [1, 0])
    assert torch.autograd.gradcheck(
        lambda inp: wrapper(inp)[2], (x,), eps=1e-6, atol=1e-5
    )
    _, j, h = wrapper(x)
    loss = j.square().mean() + h.square().mean()
    grads = torch.autograd.grad(loss, (model.w, model.b))
    # Independent finite difference for a selected parameter, not an autograd replay.
    step = 1e-6
    with torch.no_grad():
        model.w[0, 0] += step
    _, jp, hp = wrapper(x)
    plus = jp.square().mean() + hp.square().mean()
    with torch.no_grad():
        model.w[0, 0] -= 2 * step
    _, jm, hm = wrapper(x)
    minus = jm.square().mean() + hm.square().mean()
    with torch.no_grad():
        model.w[0, 0] += step
    assert torch.allclose(
        grads[0][0, 0], (plus - minus) / (2 * step), rtol=1e-5, atol=1e-5
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable locally")
def test_cuda_layer_and_derivatives():
    layer = SIREN(2, 2, "first", 4.0, mixed_policy="float64").cuda()
    x = torch.randn(3, 2, device="cuda", dtype=torch.float64)
    y, j, h = HessianLayer(layer, [0], [0, 1])(x)
    assert y.is_cuda and j.is_cuda and h.is_cuda
    (j.square().mean() + h.square().mean()).backward()
    assert layer.w.grad.is_cuda


def test_glorot_normal_retains_truncated_variance_scaling():
    from nif.layers._common import DenseLayer

    torch.manual_seed(833)
    layer = DenseLayer(512, 256, kernel_initializer="glorot_normal")
    std = math.sqrt(2 / (512 + 256)) / 0.8796256610342398
    assert layer.w.abs().max() <= 2 * std
    assert abs(layer.w.var(unbiased=False).item() / (2 / (512 + 256)) - 1) < 0.025
