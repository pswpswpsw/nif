"""Assignment-based magnitude pruning for the tutorial 7 reference workflow.

Unlike torch.nn.utils.prune, masks are assigned under no_grad before a forward;
raw kernel gradients and optimizer slots remain active at masked coordinates.
"""
import math

import torch
from torch import nn


class _PruningState(nn.Module):
    def __init__(self, weight, config):
        super().__init__()
        self.config = dict(config)
        self.register_buffer("mask", torch.ones_like(weight))
        self.register_buffer("threshold", weight.new_zeros(()))
        self.register_buffer(
            "step", torch.zeros((), dtype=torch.int64, device=weight.device)
        )
        self.register_buffer("refresh_count", torch.zeros_like(self.step))

    def get_extra_state(self):
        return dict(self.config)

    def set_extra_state(self, state):
        self.config = dict(state)


def _apply_mask(layer):
    state = layer._nif_pruning
    with torch.no_grad():
        getattr(layer, state.config["name"]).mul_(state.mask)


@torch.no_grad()
def _pruning_pre_forward(layer, inputs):
    state = layer._nif_pruning
    cfg = state.config
    weight = getattr(layer, cfg["name"])
    if layer.training:
        state.step.add_(1)
        step = int(state.step)
        if (
            cfg["begin_step"] <= step <= cfg["end_step"]
            and (step - cfg["begin_step"]) % cfg["frequency"] == 0
        ):
            # TFMOT's schedule and survivor-count arithmetic explicitly use FP32,
            # even for a FP64 kernel. torch.round uses the same ties-to-even rule.
            progress = torch.tensor((step - cfg["begin_step"]), dtype=torch.float32) / (
                cfg["end_step"] - cfg["begin_step"]
            )
            progress = progress.clamp(0, 1)
            sparsity = (cfg["initial_sparsity"] - cfg["final_sparsity"]) * (
                1 - progress
            ).pow(cfg["power"]) + cfg["final_sparsity"]
            keep = max(1, int(torch.round(weight.numel() * (1 - sparsity))))
            magnitude = weight.abs()
            # Rank the actual current parameter BEFORE applying the previous mask:
            # an optimizer update may have regrown a previously masked coordinate.
            threshold = magnitude.flatten().kthvalue(weight.numel() - keep + 1).values
            state.threshold.copy_(threshold)
            state.mask.copy_(magnitude >= threshold)
            state.refresh_count.add_(1)
    _apply_mask(layer)


class PolynomialMagnitudePruning:
    """Attach reference-convention pruning to one kernel, leaving its bias alone.

    The training counter starts at zero and increments before each training
    forward. Refresh masks at eligible steps, then assign weight *= mask before
    *every* forward (including eval). Call on_epoch_end() before checkpointing:
    optimizer updates can transiently regrow masked coordinates. Refresh ranks
    current magnitudes, keeps max(round(N*(1-s)), 1) as a threshold index, and
    retains all threshold ties. Actual sparsity can therefore differ from s.

    Model state_dict contains mask, threshold, counter and schedule configuration.
    To resume, attach a controller to the same kernel in the new model before
    loading its state_dict, and separately restore the optimizer state. strip()
    remasks and removes only pruning control state/hooks, retaining the original
    Parameter object and genuine sparse values for normal export. Eager dense
    real floating kernels are supported with one forward followed by its backward
    per optimizer update, as in tutorial 7. Multiple pending forward graphs before
    a combined backward with coordinate gradients are unsupported because masking
    modifies the kernel in place. Tracing/export follows strip().
    """

    def __init__(
        self,
        layer,
        name="w",
        *,
        initial_sparsity=0.1,
        final_sparsity=0.9,
        begin_step=0,
        end_step=200000,
        power=3.0,
        frequency=4000
    ):
        if hasattr(layer, "_nif_pruning"):
            raise ValueError("layer already has a pruning controller")
        weight = getattr(layer, name)
        if (
            not isinstance(weight, nn.Parameter)
            or not weight.is_floating_point()
            or weight.numel() == 0
        ):
            raise ValueError("pruning requires a nonempty real floating Parameter")
        for value in (initial_sparsity, final_sparsity):
            if not 0 <= value <= 1:
                raise ValueError("sparsity must be in [0, 1]")
        if (
            any(not isinstance(x, int) for x in (begin_step, end_step, frequency))
            or begin_step < 0
            or end_step <= begin_step
            or frequency <= 0
        ):
            raise ValueError(
                "require integer 0 <= begin_step < end_step and frequency > 0"
            )
        if not math.isfinite(power) or power <= 0:
            raise ValueError("power must be finite and positive")
        config = dict(
            name=name,
            initial_sparsity=initial_sparsity,
            final_sparsity=final_sparsity,
            begin_step=begin_step,
            end_step=end_step,
            power=power,
            frequency=frequency,
        )
        layer.add_module("_nif_pruning", _PruningState(weight, config))
        self.layer = layer
        self._handle = layer.register_forward_pre_hook(_pruning_pre_forward)

    @property
    def mask(self):
        return self.layer._nif_pruning.mask

    @property
    def step(self):
        return int(self.layer._nif_pruning.step)

    @property
    def refresh_count(self):
        return int(self.layer._nif_pruning.refresh_count)

    def on_epoch_end(self):
        """Clean regrowth without advancing the counter or refreshing the mask."""
        _apply_mask(self.layer)

    def strip(self):
        """Keep sparse weights and Parameter identity; remove the pruning state."""
        self.on_epoch_end()
        self._handle.remove()
        del self.layer._nif_pruning
        return self.layer
