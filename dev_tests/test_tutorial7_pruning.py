"""Source-derived pruning contracts, executing tutorial 7's actual setup cell.

For RED the facade runs the existing epoch callback and reparameterization.
Expected results are hand-constructed timing, gradient and threshold controls.
"""
import json
import time
from pathlib import Path

import nbformat
import numpy as np
import pytest
import torch
from matplotlib import pyplot as plt

from nif.layers import HyperLinearForSIREN

ROOT = Path(__file__).resolve().parents[1]
NB = nbformat.read(next((ROOT / "tutorial").glob("7_*_pytorch.ipynb")), as_version=4)


def setup(weights, *, end=200000, frequency=4000, initial=0.1, final=0.9):
    layer = HyperLinearForSIREN(
        1,
        len(weights),
        dict(weight_init_factor=0.01, input_dim=1, units=2, omega_0=30.0),
        connectivity="last_layer",
    ).double()
    with torch.no_grad():
        layer.w.copy_(torch.tensor([weights], dtype=torch.float64))
        layer.b.zero_()
    model = torch.nn.Sequential(layer)
    scope = dict(
        torch=torch,
        model_for_pruning=model,
        SMOKE=False,
        prune_epoch=1000,
        pruning_epochs=50000,
        epochs=lambda x: x,
        train_dataset=range(4),
    )
    source = next(
        c.source
        for c in NB.cells
        if c.cell_type == "code"
        and "selected_name,hyper_siren_layer=selected[0]" in c.source
    )
    exec(source, scope)
    controller = scope.get("pruning_controller")
    if controller is not None:
        # Reattach with a short public schedule for local controlled cases.
        controller.strip()
        from nif.pruning import PolynomialMagnitudePruning

        controller = PolynomialMagnitudePruning(
            layer,
            "w",
            initial_sparsity=initial,
            final_sparsity=final,
            begin_step=0,
            end_step=end,
            frequency=frequency,
        )
    return layer, model, scope, controller


def begin(scope, controller, epoch):
    if controller is None:
        scope["pruning_schedule"](epoch)


def mask(layer, controller):
    return controller.mask if controller is not None else layer.w_mask


def raw(layer):
    return getattr(layer, "w_orig", layer.w)


def test_first_refresh_is_before_forward_4000_not_epoch_zero():
    layer, model, scope, ctl = setup(range(1, 11))
    begin(scope, ctl, 0)
    model.train()
    for step in range(1, 4000):
        with torch.no_grad():
            model(torch.ones(1, 1, dtype=torch.float64))
        assert torch.count_nonzero(layer.w) == 10, f"premature mask at step {step}"
    with torch.no_grad():
        model(torch.ones(1, 1, dtype=torch.float64))
    # s(4000)=.1470464: round(10*(1-s))=9 survivors.
    assert torch.count_nonzero(layer.w) == 9


def test_raw_gradient_slots_and_transient_regrowth():
    layer, model, scope, ctl = setup(
        range(1, 11), end=1, frequency=1, initial=0.5, final=0.5
    )
    begin(scope, ctl, 0)
    if ctl is None:
        scope["set_sparsity"](0.5)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model(torch.ones(1, 1, dtype=torch.float64)).sum().backward()
    torch.testing.assert_close(raw(layer).grad, torch.ones_like(raw(layer)))
    selected = mask(layer, ctl) == 0
    opt.step()
    assert (opt.state[raw(layer)]["exp_avg"][selected] != 0).all()
    assert (opt.state[raw(layer)]["exp_avg_sq"][selected] != 0).all()
    assert (raw(layer)[selected] != 0).all()
    model.eval()
    with torch.no_grad():
        model(torch.ones(1, 1, dtype=torch.float64))
    assert (raw(layer)[selected] == 0).all()


def test_refresh_can_reselect_regrown_coordinate():
    layer, model, scope, ctl = setup(
        [1.0, 2.0, 3.0, 4.0], end=2, frequency=1, initial=0.5, final=0.5
    )
    begin(scope, ctl, 0)
    if ctl is None:
        scope["set_sparsity"](0.5)
    with torch.no_grad():
        model(torch.ones(1, 1, dtype=torch.float64))
        raw(layer)[0, 0] = 100.0  # a preceding optimizer update can regrow it
        if ctl is None:
            scope["set_sparsity"](0.5)
        model(torch.ones(1, 1, dtype=torch.float64))
    assert mask(layer, ctl).tolist() == [[1.0, 0.0, 0.0, 1.0]]
    assert layer.w[0, 0] == 100.0


@pytest.mark.parametrize(
    "weights,target,expected",
    [
        ([1.0, 2.0, 2.0, 2.0, 3.0], 0.4, [0, 1, 1, 1, 1]),  # threshold ties all survive
        (
            [1.0, 2.0, 3.0, 4.0, 5.0],
            0.5,
            [0, 0, 0, 1, 1],
        ),  # round(2.5) -> 2, not N-round(2.5)
        ([0.0, 0.0, 0.0], 0.9, [1, 1, 1]),  # zero threshold keeps tied zeros
        ([1.0, 2.0, 3.0], 1.0, [0, 0, 1]),  # at least one survivor
    ],
)
def test_threshold_ties_round_to_even_and_minimum_one(weights, target, expected):
    layer, model, scope, ctl = setup(
        weights, end=1, frequency=1, initial=target, final=target
    )
    if ctl is None:
        scope["set_sparsity"](target)
    with torch.no_grad():
        model(torch.ones(1, 1, dtype=torch.float64))
    assert mask(layer, ctl).tolist() == [expected]


def test_epoch_end_cleanup_precedes_checkpoint(tmp_path):
    layer, model, scope, ctl = setup(
        range(1, 11), end=1, frequency=1, initial=0.5, final=0.5
    )
    if ctl is None:
        scope["set_sparsity"](0.5)
    scope.update(
        nn=torch.nn,
        device=torch.device("cpu"),
        OUT=tmp_path,
        json=json,
        time=time,
        plt=plt,
        np=np,
    )
    exec(NB.cells[2].source, scope)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    kwargs = {} if ctl is None else {"after_epoch": lambda epoch: ctl.on_epoch_end()}
    scope["train"](
        model,
        [
            (
                torch.ones(1, 1, dtype=torch.float64),
                torch.zeros(1, 10, dtype=torch.float64),
            )
        ],
        opt,
        1,
        "cleanup",
        **kwargs,
    )
    saved = torch.load(tmp_path / "cleanup_latest.pt", weights_only=True)["model_state"]
    selected = mask(layer, ctl) == 0
    assert (raw(layer)[selected] == 0).all(), "epoch end must clean transient regrowth"
    assert (saved.get("0.w", saved.get("0.w_orig"))[selected] == 0).all()
    if ctl is not None:
        assert ctl.step == 1 and ctl.refresh_count == 1
