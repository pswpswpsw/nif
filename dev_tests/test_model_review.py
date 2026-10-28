"""Additional Agent A layer-owner review of model precision and subnet APIs."""
import json

import pytest
import torch

from nif.model import NIF
from nif.model import NIFMultiScale
from nif.model import NIFMultiScaleLastLayerParameterized


def configs(cls, act="tanh"):
    s = dict(
        input_dim=2,
        output_dim=2,
        units=4,
        nlayers=1,
        activation="tanh",
        use_resblock=True,
        omega_0=3.0,
        weight_init_factor=0.1,
        connectivity="last_layer"
        if cls is NIFMultiScaleLastLayerParameterized
        else "full",
    )
    p = dict(
        input_dim=1,
        latent_dim=3,
        units=5,
        nlayers=1,
        activation=act,
        use_resblock=True,
        omega_0=2.0,
    )
    return s, p


@pytest.mark.parametrize(
    "cls", [NIF, NIFMultiScale, NIFMultiScaleLastLayerParameterized]
)
@pytest.mark.parametrize("precision", ["mixed_bfloat16", "mixed_float16"])
def test_mixed_precision_subnet_reconstruction(cls, precision):
    torch.manual_seed(71)
    m = cls(*configs(cls), mixed_policy=precision)
    x = torch.randn(9, 3)
    y = m(x)
    w = m.model_p_to_w()(x[:, :1])
    recomposed = m.model_x_to_u_given_w()(x[:, 1:], w)
    assert recomposed.dtype == y.dtype == torch.float32
    # Extraction must preserve the parent's arithmetic, including its policy.
    torch.testing.assert_close(recomposed, y, atol=0, rtol=0)
    if cls is NIFMultiScaleLastLayerParameterized:
        assert m.model_x_to_phi()(x[:, 1:]).dtype == torch.float32


@pytest.mark.parametrize(
    "cls", [NIF, NIFMultiScale, NIFMultiScaleLastLayerParameterized]
)
def test_config_and_weights_restore_native_dtype(cls, tmp_path):
    m = cls(*configs(cls)).double()
    path = tmp_path / "config.json"
    m.save_config(path)
    restored = cls(**json.loads(path.read_text(encoding="utf-8")))
    restored.load_state_dict(m.state_dict())
    assert next(restored.parameters()).dtype == torch.float64
    x = torch.randn(4, 3, dtype=torch.float64)
    torch.testing.assert_close(restored(x), m(x), atol=0, rtol=0)


@pytest.mark.parametrize(
    "cls", [NIF, NIFMultiScale, NIFMultiScaleLastLayerParameterized]
)
def test_regularization_coefficients_and_batch_reductions(cls):
    s, p = configs(cls)
    p.update(l1_reg=0.15, l2_reg=0.2, act_l1_reg=0.25, act_l2_reg=0.3, jac_reg=0.4)
    if cls is NIFMultiScaleLastLayerParameterized:
        s.update(l1_reg=0.6, l2_reg=0.7)
    m = cls(s, p, mixed_policy="float64")
    x = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    # Both regularizers supplied: original gives L2 precedence over L1.
    expected = 0.2 * sum(v.square().sum() for v in m.pnet_list.parameters())
    if cls is NIFMultiScaleLastLayerParameterized:
        expected = expected + 0.7 * sum(
            v.square().sum() for v in m.snet_list.parameters()
        )
    latent = x[:, :1]
    for layer in m.pnet_list[:-1]:
        latent = layer(latent)
    output = m.pnet_list[-1](latent)
    grad = torch.stack(
        [
            torch.autograd.grad(
                latent[:, j].sum(), x, create_graph=True, retain_graph=True
            )[0][:, :1]
            for j in range(latent.shape[-1])
        ],
        1,
    )
    expected = (
        expected + 0.3 * output.square().sum() / len(x) + 0.4 * grad.square().mean()
    )
    torch.testing.assert_close(m.regularization_loss(x), expected)
    torch.testing.assert_close(m.regularization_loss(torch.cat((x, x))), expected)
    # Final physical output bias has no kernel/bias regularizer in the baseline.
    if cls is NIFMultiScaleLastLayerParameterized:
        with torch.no_grad():
            m.last_bias_layer.last_layer_bias.add_(100)
        torch.testing.assert_close(m.regularization_loss(x), expected)
    gradients = torch.autograd.grad(
        m.regularization_loss(x), list(m.pnet_list.parameters())
    )
    assert all(torch.isfinite(g).all() for g in gradients)
