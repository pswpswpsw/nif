"""Self-owned tests of the public migration mathematics (not acceptance tests)."""
import json

import pytest
import torch

from nif import NIF
from nif import NIFMultiScale
from nif import NIFMultiScaleLastLayerParameterized


def configs(res=False, activation="sine", connectivity="full"):
    s = dict(
        input_dim=2,
        output_dim=2,
        units=4,
        nlayers=1,
        activation="tanh",
        use_resblock=res,
        omega_0=3.0,
        connectivity=connectivity,
        weight_init_factor=0.1,
    )
    p = dict(
        input_dim=1,
        latent_dim=3,
        units=5,
        nlayers=1,
        activation=activation,
        use_resblock=res,
        omega_0=2.0,
    )
    return s, p


@pytest.mark.parametrize(
    "cls", [NIF, NIFMultiScale, NIFMultiScaleLastLayerParameterized]
)
@pytest.mark.parametrize("res", [False, True])
@pytest.mark.parametrize("activation", ["sine", "tanh"])
def test_model_views_gradients_checkpoint(cls, res, activation, tmp_path):
    s, p = configs(
        res,
        activation,
        "last_layer" if cls is NIFMultiScaleLastLayerParameterized else "full",
    )
    if cls is NIF:
        p["activation"] = "tanh"
    m = cls(s, p).double()
    assert isinstance(m, torch.nn.Module) and m.build() is m and m.model() is m
    x = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
    y = m(x)
    assert y.shape == (6, 2) and y.dtype == torch.float64
    w = m.model_p_to_w()(x[:, :1])
    lr = m.model_p_to_lr()(x[:, :1])
    torch.testing.assert_close(m.model_lr_to_w()(lr), w)
    view = m.model_x_to_u_given_w()
    torch.testing.assert_close(view([x[:, 1:], w]), y)
    assert set(map(id, m.model_p_to_w().parameters())) <= set(map(id, m.parameters()))
    if cls is NIFMultiScaleLastLayerParameterized:
        phi = m.model_x_to_phi()(x[:, 1:])
        torch.testing.assert_close(
            y, torch.einsum("bol,bl->bo", phi, w) + m.last_bias_layer.last_layer_bias
        )
    torch.autograd.grad(y.sum(), x, create_graph=True)[0].square().mean().backward()
    assert any(
        t.grad is not None and torch.isfinite(t.grad).all() for t in m.parameters()
    )
    path = tmp_path / "weights.pt"
    torch.save(m.state_dict(), path)
    restored = cls(s, p).double()
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(restored(x), y)
    m.save_config(tmp_path / "config.json")
    assert (
        json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))[
            "cfg_shape_net"
        ]
        == s
    )


def manual_shape(x, w, s, multiscale):
    d, h, o, layers = s["input_dim"], s["units"], s["output_dim"], s["nlayers"]
    n = layers * (2 if multiscale and s["use_resblock"] else 1)
    sizes = [(d, h)] + [(h, h)] * n + [(h, o)]
    weights = []
    pos = 0
    for a, b in sizes:
        weights.append(w[:, pos : pos + a * b].reshape(-1, a, b))
        pos += a * b
    biases = []
    for a, b in sizes:
        biases.append(w[:, pos : pos + b])
        pos += b

    def affine(z, i):
        return torch.einsum("bi,bij->bj", z, weights[i])

    def act(z, i):
        return (
            torch.sin(s["omega_0"] * affine(z, i) + biases[i])
            if multiscale
            else torch.tanh(affine(z, i) + biases[i])
        )

    u = act(x, 0)
    if multiscale and s["use_resblock"]:
        for i in range(layers):
            u = 0.5 * (u + act(act(u, 1 + 2 * i), 2 + 2 * i))
    else:
        for i in range(layers):
            u = act(u, i + 1) + (u if not multiscale else 0)
    return affine(u, -1) + biases[-1]


@pytest.mark.parametrize("cls", [NIF, NIFMultiScale])
@pytest.mark.parametrize("res", [False, True])
def test_generated_layout(cls, res):
    s, p = configs(res, "tanh")
    m = cls(s, p).double()
    x = torch.randn(3, 2, dtype=torch.float64)
    w = torch.linspace(-0.3, 0.4, 3 * m.po_dim, dtype=torch.float64).reshape(3, -1)
    torch.testing.assert_close(
        m.model_x_to_u_given_w()(x, w), manual_shape(x, w, s, cls is NIFMultiScale)
    )


def test_regularization_reductions_and_no_stale_cache():
    s, p = configs(False, "tanh", "last_layer")
    s["l2_reg"] = 0.7
    p["l2_reg"] = 0.2
    m = NIFMultiScaleLastLayerParameterized(s, p).double()
    expected = 0.2 * sum(
        v.square().sum() for v in m.pnet_list.parameters()
    ) + 0.7 * sum(v.square().sum() for v in m.snet_list.parameters())
    torch.testing.assert_close(m.regularization_loss(), expected)
    s, p = configs(False, "tanh")
    p.update(act_l1_reg=0.3, jac_reg=0.4)
    m = NIFMultiScale(s, p).double()
    x = torch.randn(5, 3, dtype=torch.float64)
    m(x)
    with pytest.raises(ValueError, match="inputs"):
        m.regularization_loss()
    xx = x.clone().requires_grad_()
    latent = m.model_p_to_lr()(xx[:, :1])
    jac = torch.stack(
        [
            torch.autograd.grad(
                latent[:, j].sum(), xx, create_graph=True, retain_graph=True
            )[0][:, :1]
            for j in range(3)
        ],
        dim=1,
    )
    expected = (
        0.3 * m.model_p_to_w()(xx[:, :1]).abs().sum() / len(xx)
        + 0.4 * jac.square().mean()
    )
    torch.testing.assert_close(m.regularization_loss(xx), expected)
    m.regularization_loss(xx).backward()
    assert any(v.grad is not None for v in m.parameters())


@pytest.mark.parametrize("policy", ["float32", "float64", "mixed_bfloat16"])
@pytest.mark.parametrize("activation", ["sine", "tanh"])
def test_precision_and_subnet_equivalence(policy, activation):
    s, p = configs(True, activation)
    m = NIFMultiScale(s, p, mixed_policy=policy)
    x = torch.randn(4, 3, dtype=torch.float64 if policy == "float64" else torch.float32)
    y = m(x)
    assert y.dtype == x.dtype and torch.isfinite(y).all()
    w = m.model_p_to_w()(x[:, :1])
    reconstructed = m.model_x_to_u_given_w()(x[:, 1:], w)
    torch.testing.assert_close(
        reconstructed.to(y.dtype),
        y,
        atol=0.01 if policy.startswith("mixed_") else 1e-7,
        rtol=0.03 if policy.startswith("mixed_") else 1e-6,
    )


def test_finite_training_reduces_loss():
    torch.manual_seed(8)
    s, p = configs(False, "tanh")
    m = NIF(s, p)
    x = torch.randn(40, 3)
    target = torch.cat((x[:, :1] + 0.2 * x[:, 1:2], x[:, 2:3] ** 2), dim=1)
    optimizer = torch.optim.Adam(m.parameters(), lr=0.01)
    initial = (m(x) - target).square().mean().item()
    for _ in range(60):
        optimizer.zero_grad()
        loss = (m(x) - target).square().mean() + m.regularization_loss(x)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    assert (m(x) - target).square().mean() < initial * 0.3
