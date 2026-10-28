"""Current T1/T3 actual helper training/checkpoint continuation, CPU smoke only."""
import copy
import json
import time
from pathlib import Path

import matplotlib
import nbformat
import numpy as np
import pytest
import torch
from torch import nn

import nif
from nif.demo import CylinderFlow
from nif.demo import TravelingWave
from nif.optimizers import AdaBeliefOptimizer

# Select the noninteractive backend before importing pyplot.
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype
        assert torch.equal(
            a.detach().contiguous().view(torch.uint8),
            b.detach().contiguous().view(torch.uint8),
        )
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("number", [1, 3])
def test_actual_tutorial_helpers_resume_real_gradients_bitwise_on_cpu(tmp_path, number):
    notebook = nbformat.read(
        next((ROOT / "tutorial").glob(f"{number}_*_pytorch.ipynb")), as_version=4
    )
    namespace = dict(
        torch=torch,
        nn=nn,
        np=np,
        json=json,
        time=time,
        copy=copy,
        plt=plt,
        device=torch.device("cpu"),
        OUT=tmp_path,
        SMOKE=True,
    )
    exec(
        compile(
            notebook.cells[2].source,
            "<actual notebook training/checkpoint helpers>",
            "exec",
        ),
        namespace,
    )
    config_cell = next(
        c.source
        for c in notebook.cells
        if c.cell_type == "code" and c.source.startswith("cfg_shape_net =")
    )
    exec(compile(config_cell, "<actual notebook configuration>", "exec"), namespace)
    cls = nif.NIF if number == 1 else nif.NIFMultiScaleLastLayerParameterized

    def make_model():
        return cls(
            copy.deepcopy(namespace["cfg_shape_net"]),
            copy.deepcopy(namespace["cfg_parameter_net"]),
            "float32",
        )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(729)
        model = make_model()
        data = TravelingWave() if number == 1 else CylinderFlow()
        # Controlled short CPU subset only; notebook source/science stays intact.
        rows = np.linspace(0, len(data.data) - 1, 32, dtype=int)
        dim = 2 if number == 1 else 3
        batch = [
            torch.as_tensor(data.data[rows, :dim], dtype=torch.float32),
            torch.as_tensor(data.data[rows, dim:], dtype=torch.float32),
        ]
        if number == 3:
            batch.append(torch.as_tensor(data.sample_weight[rows], dtype=torch.float32))
        opt = AdaBeliefOptimizer(model.parameters(), lr=0.005 if number == 1 else 0.001)
        train = namespace["train"]
        train(model, [batch], opt, 7, "before_resume", centralization=number == 3)
        saved = torch.load(
            tmp_path / "before_resume_latest.pt", weights_only=True, map_location="cpu"
        )
        restored = make_model()
        restored.load_state_dict(saved["model_state"])
        resumed = AdaBeliefOptimizer(restored.parameters())
        resumed.load_state_dict(saved["optimizer_state"])
        same(model.state_dict(), restored.state_dict())
        same(opt.state_dict(), resumed.state_dict())
        train(model, [batch], opt, 3, "continued", centralization=number == 3)
        train(restored, [batch], resumed, 3, "restored", centralization=number == 3)
        same(model.state_dict(), restored.state_dict())
        same(opt.state_dict(), resumed.state_dict())
        same(model(batch[0]), restored(batch[0]))
        assert all(group["step"] == 10 for group in opt.param_groups)
        assert all(state["step"] == 10 for state in opt.state.values())
