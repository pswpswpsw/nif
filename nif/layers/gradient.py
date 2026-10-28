"""Differentiable pointwise Jacobians and Hessians for neural implicit fields.

A batch consists of independent points, as in the original batch-Jacobian
operators. Summing the chosen output across rows computes all point derivatives
in one reverse pass. Models coupling distinct batch rows are not pointwise and
must instead use a full Jacobian (for example torch.func.jacrev).

All reverse passes create graphs, including explicit connected zeros for
constant and linear functions, so derivative losses can train model parameters.
"""
import numbers

import torch

from ._common import NativeLayer


def _indices(index, size, name):
    if isinstance(index, numbers.Integral):
        index = [int(index)]
    elif isinstance(index, torch.Tensor):
        index = index.tolist()
    index = list(index)
    if not index:
        raise ValueError(f"{name} must select at least one component")
    if any(not isinstance(i, numbers.Integral) or not -size <= i < size for i in index):
        raise IndexError(f"{name} index outside dimension {size}")
    return [int(i) % size for i in index]


def _input(x):
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.get_default_dtype())
    if x.ndim != 2 or not x.is_floating_point():
        raise ValueError(
            "Derivative inputs must be floating point [batch, inputs] tensors"
        )
    # Preserve a caller's graph; only manufacture a leaf when no graph exists.
    return x if x.requires_grad else x.detach().requires_grad_(True)


def _zero(x, y):
    # Includes y so zero derivatives retain parameter differentiation as well.
    return x * 0 + y.sum() * 0


def _gradient(y, x, anchor):
    gradient = None
    if y.requires_grad:
        gradient = torch.autograd.grad(
            y,
            x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
    zero = _zero(x, anchor)
    return zero if gradient is None else gradient + zero


def _jacobian(y, x, x_index, y_index):
    if not isinstance(y, torch.Tensor) or y.ndim != 2 or y.shape[0] != x.shape[0]:
        raise ValueError("Model output must be [batch, outputs]")
    xi = _indices(x_index, x.shape[1], "x_index")
    yi = _indices(y_index, y.shape[1], "y_index")
    jac = torch.stack([_gradient(y[:, i], x, y)[:, xi] for i in yi], dim=1)
    return jac, xi


def compute_output_and_grad(model, x, x_index, y_index):
    """Return full model output and selected [batch, output, input] Jacobian."""
    with torch.enable_grad():
        x = _input(x)
        y = model(x)
        jac, _ = _jacobian(y, x, x_index, y_index)
        return y, jac


def compute_output_and_augment_grad(model, x, x_index, y_index):
    """Differentiate the second output of an augmented (output, latent) model."""
    with torch.enable_grad():
        x = _input(x)
        y, latent = model(x)
        jac, _ = _jacobian(latent, x, x_index, y_index)
        return y, jac


def compute_output_and_grad_and_hessian(model, x, x_index, y_index):
    """Return full output, selected Jacobian, and selected-input Hessian.

    Hessian axes are [batch, selected_outputs, selected_inputs, selected_inputs].
    """
    with torch.enable_grad():
        x = _input(x)
        y = model(x)
        jac, xi = _jacobian(y, x, x_index, y_index)
        hessian = torch.stack(
            [
                torch.stack(
                    [_gradient(jac[:, o, i], x, y)[:, xi] for i in range(len(xi))],
                    dim=1,
                )
                for o in range(jac.shape[1])
            ],
            dim=1,
        )
        return y, jac, hessian


class JacobianLayer(NativeLayer):
    """Wrap a native pointwise model and expose its selected input derivatives."""

    def __init__(self, model, y_index, x_index, **kwargs):
        super().__init__()
        self.model = model
        self.y_index = y_index
        self.x_index = x_index

    def forward(self, x, **kwargs):
        return compute_output_and_grad(self.model, x, self.x_index, self.y_index)

    def get_config(self):
        return dict(y_index=self.y_index, x_index=self.x_index)


class HessianLayer(JacobianLayer):
    """Expose full output plus selected first and second input derivatives."""

    def forward(self, x, **kwargs):
        return compute_output_and_grad_and_hessian(
            self.model, x, self.x_index, self.y_index
        )


class JacRegLatentLayer(JacobianLayer):
    """Pass through model output and explicitly penalize its latent Jacobian.

    ``model(x)`` returns ``(output, latent)``. The historical ``l1`` name is
    retained, but the penalty is l1 * mean(Jacobian ** 2), not an absolute norm.
    Call ``regularization_loss(inputs)`` in the training objective. No batch is
    cached, so evaluating unrelated data cannot silently change this penalty.
    """

    def __init__(
        self, model, y_index, x_index, l1=1e-2, mixed_policy="float32", **kwargs
    ):
        super().__init__(model, y_index, x_index, **kwargs)
        self.l1 = float(l1)
        self.mixed_policy = mixed_policy

    def forward(self, x, **kwargs):
        y, _ = self.model(x)
        return y

    def regularization_loss(self, inputs=None):
        if inputs is None:
            raise ValueError("inputs are required for latent Jacobian regularization")
        _, jac = compute_output_and_augment_grad(
            self.model, inputs, self.x_index, self.y_index
        )
        return self.l1 * jac.square().mean()

    def get_config(self):
        return dict(super().get_config(), l1=self.l1)
