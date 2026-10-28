"""Gradient centralization adapted from the original NIF/GCTF implementation.

The baseline reduces all input axes of a TF kernel ``[in, out]``. Native
Linear stores ``[out, in]``, so the equivalent reduction is dimension 1.
Custom TF-layout parameters can declare ``_nif_gc_axes=(0,)``.

The historical gradient-returning helper explicitly applies ``clipnorm`` to
one norm across all its gradients, then applies ``clipvalue``. This reduction
and ordering are preserved. Ordinary optimizer.step instead follows its base
OptimizerV2 behavior: clipvalue first, then a separate clipnorm per tensor (or
an explicitly requested global_clipnorm).
"""
import torch

from nif.optimizers.external_optimizers import _norm_scale
from nif.optimizers.external_optimizers import centralize_gradient
from nif.optimizers.external_optimizers import L4Adam


def get_centralized_gradients(optimizer, loss, params):
    """Compute differentiable centralized gradients for a scalar loss.

    The returned tensors can be assigned to ``parameter.grad`` before step.
    The convenience installer below also centralizes ordinary backward grads.
    """
    params = list(params)
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    if any(grad is None for grad in grads):
        raise ValueError(
            "An operation has None for gradient; all parameters must participate in the loss"
        )
    grads = [centralize_gradient(grad, p) for p, grad in zip(params, grads)]
    clipnorm = getattr(optimizer, "clipnorm", optimizer.defaults.get("clipnorm"))
    clipvalue = getattr(optimizer, "clipvalue", optimizer.defaults.get("clipvalue"))
    if clipnorm is not None and clipnorm > 0:
        norm = torch.sqrt(sum(grad.square().sum() for grad in grads))
        scale = _norm_scale(norm, clipnorm)
        grads = [grad * scale.to(grad) for grad in grads]
    if clipvalue is not None and clipvalue > 0:
        grads = [grad.clamp(-clipvalue, clipvalue) for grad in grads]
    return grads


def centralized_gradients_for_optimizer(optimizer):
    """Enable centralization on an optimizer and return the legacy helper.

    ``centralized_gradients_for_optimizer(opt)`` installs a native step hook;
    ``loss.backward(); opt.step()`` then actually uses centralized gradients.
    For compatibility the return value accepts ``(loss, params)``. Repeated
    installation is harmless. Remove via ``opt._nif_centralization_hook.remove()``.
    For L4Adam loss-only closures, its automatic backward is completed inside
    the wrapper before centralization; closures doing backward still run once.
    """

    def centralize():
        with torch.no_grad():
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        grad = p.grad.to_dense() if p.grad.is_sparse else p.grad
                        p.grad = centralize_gradient(grad, p)

    def before_step(opt, args, kwargs):
        # Closures can create gradients inside step, so centralize after the
        # closure as well as the ordinary eager backward path.
        closure = kwargs.get("closure")
        positional = len(args) > 1 and callable(args[1])
        if positional:
            closure = args[1]
        if closure is not None:

            def wrapped_closure():
                result = closure()
                # L4 accepts a loss-only closure and otherwise performs backward
                # after this wrapper returns. Complete that same automatic step
                # here so GC sees the gradients before moments are updated.
                if isinstance(opt, L4Adam) and not any(
                    p.grad is not None
                    for group in opt.param_groups
                    for p in group["params"]
                ):
                    result.backward()
                centralize()
                return result

            if positional:
                args = (args[0], wrapped_closure, *args[2:])
            else:
                kwargs = dict(kwargs, closure=wrapped_closure)
        else:
            centralize()
        return args, kwargs

    if not hasattr(optimizer, "_nif_centralization_hook"):
        optimizer._nif_centralization_hook = optimizer.register_step_pre_hook(
            before_step
        )

    def gradients(loss, params):
        return get_centralized_gradients(optimizer, loss, params)

    return gradients
