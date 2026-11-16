"""Contracts retained while sharing precision, field and gradient helpers."""
from types import SimpleNamespace

import pytest
import torch

from nif import NIFMultiScaleLastLayerParameterized
from nif.layers._common import policy_dtypes
from nif.model import _policy
from nif.optimizers.lbfgs import function_factory
from nif.optimizers.lbfgs_V2 import create_value_and_gradient_function
from nif.optimizers.lbfgs_V2 import pack_tensors


@pytest.mark.parametrize(
    "name, variable, compute",
    [
        ("float16", torch.float16, torch.float16),
        ("float32", torch.float32, torch.float32),
        ("float64", torch.float64, torch.float64),
        ("bfloat16", torch.bfloat16, torch.bfloat16),
        ("mixed_float16", torch.float32, torch.float16),
        ("mixed_bfloat16", torch.float32, torch.bfloat16),
    ],
)
def test_model_and_layer_named_precision_contract(name, variable, compute):
    assert _policy(name) == (name, variable, compute)
    assert _policy(SimpleNamespace(name=name)) == (name, variable, compute)
    assert policy_dtypes(name) == (variable, compute)


@pytest.mark.parametrize("value", ["half", "double", "float", "torch.float64", None])
def test_layer_aliases_do_not_expand_model_policy_inputs(value):
    expected = {
        "half": torch.float16,
        "double": torch.float64,
        "float": torch.float32,
        "torch.float64": torch.float64,
        None: torch.float32,
    }[value]
    assert policy_dtypes(value) == (expected, expected)
    with pytest.raises(ValueError, match="Unsupported precision policy"):
        _policy(value)


def test_precision_object_protocols_remain_distinct():
    layer_policy = SimpleNamespace(variable_dtype="double", compute_dtype="half")
    assert policy_dtypes(layer_policy) == (torch.float64, torch.float16)
    with pytest.raises(TypeError):
        _policy(layer_policy)
    with pytest.raises(ValueError, match="Unsupported floating point dtype"):
        policy_dtypes(SimpleNamespace(name="float64"))
    for dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        assert _policy(dtype) == (str(dtype).removeprefix("torch."), dtype, dtype)
        assert policy_dtypes(dtype) == (dtype, dtype)


@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("precision", ["float64", "mixed_bfloat16"])
def test_last_layer_views_preserve_graph_and_rng(residual, precision):
    shape = dict(
        input_dim=2,
        output_dim=3,
        units=5,
        nlayers=1,
        activation="sine",
        use_resblock=residual,
        omega_0=2.5,
        connectivity="last_layer",
        weight_init_factor=0.1,
    )
    parameter = dict(
        input_dim=1,
        latent_dim=2,
        units=4,
        nlayers=1,
        activation="tanh",
        use_resblock=residual,
        omega_0=1.5,
    )
    torch.manual_seed(713)
    model = NIFMultiScaleLastLayerParameterized(shape, parameter, precision)
    initial_rng = torch.get_rng_state().clone()
    field_view = model.model_x_to_u_given_w()
    basis_view = model.model_x_to_phi()
    assert torch.equal(initial_rng, torch.get_rng_state())
    assert set(map(id, field_view.parameters())) <= set(map(id, model.parameters()))
    assert set(map(id, basis_view.parameters())) <= set(map(id, model.parameters()))
    assert len(list(field_view.modules())) < len(list(model.modules()))

    # Asymmetric coordinates, three outputs and a two-dimensional latent avoid
    # accidentally treating spatial, field and basis dimensions interchangeably.
    inputs = torch.tensor(
        [[-0.4, 0.1, 0.8], [0.7, -0.5, 0.2], [0.3, 0.9, -0.6]],
        dtype=model.variable_Dtype,
        requires_grad=True,
    )
    coefficients = model.model_p_to_w()(inputs[:, :1])
    expected = model(inputs)
    actual = field_view(inputs[:, 1:], coefficients)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        field_view([inputs[:, 1:], coefficients]), expected, rtol=0, atol=0
    )
    direct_phi = model._call_shape_net_get_phi_x(inputs[:, 1:], model.snet_list, 3, 2)
    if precision == "float64":
        torch.testing.assert_close(
            basis_view(inputs[:, 1:]), direct_phi, rtol=0, atol=0
        )
    assert basis_view(inputs[:, 1:]).shape == (3, 3, 2)
    variables = (inputs, *model.parameters())
    left = torch.autograd.grad(expected.square().sum(), variables, retain_graph=True)
    right = torch.autograd.grad(actual.square().sum(), variables)
    for original, recomposed in zip(left, right):
        torch.testing.assert_close(recomposed, original, rtol=0, atol=0)


class _QuadraticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor([[0.5, -0.2], [0.3, 0.7]], dtype=torch.float64)
        )
        self.unused = torch.nn.Parameter(torch.tensor([4.0, -9.0], dtype=torch.float64))

    def forward(self, value):
        return value @ self.weight.T


def test_both_gradient_factories_preserve_assignment_unused_grads_and_logging(capsys):
    model = _QuadraticModule()
    x = torch.tensor([[1.0, -2.0], [0.3, 0.8], [-0.7, 0.4]], dtype=torch.float64)
    target = torch.tensor([[0.2, 0.6], [-0.4, 0.1], [0.9, -0.3]], dtype=x.dtype)

    def loss(prediction, truth):
        return (prediction - truth).square().mean()

    factory = function_factory(model, loss, x.numpy(), target.numpy(), display_epoch=2)
    closure, assign = create_value_and_gradient_function(
        lambda: loss(model(x), target), model.parameters(), verbose=1
    )
    initial = pack_tensors(model.parameters()).detach()
    backing = torch.zeros(initial.numel() * 2, dtype=initial.dtype)
    position = backing[::2]
    position.copy_(initial + 0.1)
    assert not position.is_contiguous()
    assign(position)
    difference = x @ model.weight.detach().T - target
    expected_value = difference.square().mean()
    expected_gradient = torch.cat(
        ((2 * difference.T @ x / difference.numel()).reshape(-1), torch.zeros(2))
    )
    with torch.no_grad():
        value, gradient = factory(position)
        other_value, other_gradient = closure(position)
    torch.testing.assert_close(value, expected_value, rtol=0, atol=0)
    torch.testing.assert_close(other_value, expected_value, rtol=0, atol=0)
    torch.testing.assert_close(gradient, expected_gradient, rtol=1e-15, atol=1e-15)
    torch.testing.assert_close(other_gradient, gradient, rtol=0, atol=0)
    assert not value.requires_grad and not gradient.requires_grad
    assert factory.iter == closure.iter == 1
    assert factory.history == [float(expected_value)]
    assert capsys.readouterr().out.startswith("Iter: 1 loss:")
    factory(position)
    assert capsys.readouterr().out.startswith("Epoch: 2 loss:")
    assert len(factory.idx) == 2 and factory.part.tolist() == [0, 0, 0, 0, 1, 1]
    assert factory.shapes == [torch.Size([2, 2]), torch.Size([2])]
    assert factory.assign_new_model_parameters(initial) is None
    torch.testing.assert_close(
        pack_tensors(model.parameters()), initial, rtol=0, atol=0
    )
