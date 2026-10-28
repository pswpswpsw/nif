"""Native implementation of the original NIF L-BFGS training convenience API."""
import numpy as np
import torch

from nif.optimizers.lbfgs_V2 import assign_tensors
from nif.optimizers.lbfgs_V2 import lbfgs_minimize
from nif.optimizers.lbfgs_V2 import pack_tensors
from nif.optimizers.lbfgs_V2 import unpack_tensors


def function_factory(model, loss, train_x, train_y, display_epoch=1):
    """Pack a module into a callable returning scalar loss and flat gradients.

    ``loss(prediction, target)`` retains the baseline argument order. Inputs
    are converted to the module's device/dtype. ``idx``, ``part``, ``shapes``,
    ``iter``, ``history`` and ``assign_new_model_parameters`` remain available.
    """
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("Model has no trainable parameters")
    reference = parameters[0]
    train_x = torch.as_tensor(train_x, device=reference.device, dtype=reference.dtype)
    train_y = torch.as_tensor(train_y, device=reference.device, dtype=reference.dtype)
    idx = []
    parts = []
    count = 0
    for i, p in enumerate(parameters):
        idx.append(
            torch.arange(count, count + p.numel(), device=p.device).reshape(p.shape)
        )
        parts.extend([i] * p.numel())
        count += p.numel()

    def assign(params_1d):
        assign_tensors(parameters, unpack_tensors(parameters, params_1d))

    def f(params_1d):
        assign(params_1d)
        with torch.enable_grad():
            value = loss(model(train_x), train_y)
            gradients = torch.autograd.grad(value, parameters, allow_unused=True)
        gradients = [
            torch.zeros_like(p) if g is None else g
            for p, g in zip(parameters, gradients)
        ]
        f.iter += 1
        f.history.append(float(value.detach()))
        if display_epoch and f.iter % display_epoch == 0:
            print(f"Epoch: {f.iter} loss: {f.history[-1]:.8g}")
        return value.detach(), pack_tensors(gradients).detach()

    f.iter = 0
    f.idx = idx
    f.part = torch.tensor(parts, dtype=torch.int64, device=reference.device)
    f.shapes = [p.shape for p in parameters]
    f.assign_new_model_parameters = assign
    f.history = []
    return f


class TFPLBFGS:
    """Historical name; optimization uses the shared native ReferenceLBFGS core."""

    def __init__(self, model, loss_fun, inps, outs, display_epoch=1):
        self.func = function_factory(model, loss_fun, inps, outs, display_epoch)
        self.model = model
        self.results = None

    def minimize(self, rounds=50, max_iter=50):
        if int(rounds) <= 0 or int(max_iter) <= 0:
            raise ValueError("rounds and max_iter must be positive")
        for _ in range(int(rounds)):
            self.results = lbfgs_minimize(
                self.func,
                initial_position=pack_tensors(
                    p for p in self.model.parameters() if p.requires_grad
                ),
                num_correction_pairs=20,
                tolerance=1e-15,
                x_tolerance=1e-15,
                f_relative_tolerance=1e-15,
                max_iterations=int(max_iter),
                max_line_search_iterations=100,
            )
            self.func.assign_new_model_parameters(self.results.position)
        return self.results

    @property
    def history(self):
        return {
            "iteration": np.arange(1, len(self.func.history) + 1),
            "loss": list(self.func.history),
        }
