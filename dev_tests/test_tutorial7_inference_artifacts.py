"""CPU artifact integration check; no tutorial training/convergence claim."""
import copy
import json
import os
from pathlib import Path
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn

import nif
from nif.layers import HyperLinearForSIREN
from nif.pruning import PolynomialMagnitudePruning

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]


def test_four_benchmark_files_are_real_inference_artifacts(tmp_path):
    notebook = json.loads((ROOT / "tutorial" / "7_model_pruning_and_quantization_pytorch.ipynb").read_text(encoding="utf-8"))
    scope = dict(Path=Path, copy=copy, json=json, time=time, np=np, torch=torch,
                 nn=nn, plt=plt, nif=nif, OUT=tmp_path, SMOKE=True,
                 device=torch.device("cpu"), HyperLinearForSIREN=HyperLinearForSIREN)

    def run(index):
        exec("".join(notebook["cells"][index]["source"]), scope)

    torch.manual_seed(17)
    run(2)
    run(5)
    model = nif.NIFMultiScale(scope["cfg_shape_net"], scope["cfg_parameter_net"], "float32").build()
    optimizer = nif.optimizers.ReferenceAdam(model.parameters(), lr=1e-3, eps=1e-7)
    # One synthetic update creates real optimizer slots for the separate resume file.
    model(torch.tensor([[.1, .2], [.3, .4]])).square().mean().backward()
    optimizer.step()
    scope.update(model=model, optimizer=optimizer)
    training_path = tmp_path / "baseline_latest.pt"
    scope["save_checkpoint"](model, optimizer, training_path, 0)
    run(8)
    run(10)

    pruned_model = copy.deepcopy(scope["model"])
    selected = [layer for layer in pruned_model.modules() if isinstance(layer, HyperLinearForSIREN)]
    assert len(selected) == 1
    controller = PolynomialMagnitudePruning(selected[0], "w", initial_sparsity=.9,
        final_sparsity=.9, begin_step=0, end_step=1, frequency=1)
    pruned_model.train()
    with torch.no_grad():
        pruned_model(torch.tensor([[.1, .2]]))
    controller.on_epoch_end()
    scope.update(model_for_pruning=pruned_model, pruning_controller=controller,
                 hyper_siren_layer=selected[0], run_pruning_epochs=1)
    for index in (24, 26, 28, 29, 31, 33, 34, 37):
        run(index)

    features = torch.tensor(scope["input_feature"], dtype=torch.float32)
    for key, expected in (("original_file", scope["y_pred_unpruned"]),
                          ("pruned_file", scope["y_pred"])):
        payload = torch.load(scope[key], map_location="cpu", weights_only=True)
        assert set(payload) == {"model_state", "framework", "artifact_format", "model_class",
                                "cfg_shape_net", "cfg_parameter_net", "mixed_policy"}
        assert not any("_nif_pruning" in name for name in payload["model_state"])
        restored = nif.NIFMultiScale(payload["cfg_shape_net"], payload["cfg_parameter_net"], "float32").eval()
        restored.load_state_dict(payload["model_state"])
        with torch.no_grad():
            actual = restored(features).numpy().reshape(200, 10)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    for key, model in (("exported_file", scope["model_for_export"]),
                       ("quantized_file", scope["quantized_model"])):
        restored = torch.jit.load(scope[key], map_location="cpu")
        with torch.no_grad():
            torch.testing.assert_close(restored(features), model(features), rtol=0, atol=0)

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    paths = [scope[key] for key in ("original_file", "pruned_file", "exported_file", "quantized_file")]
    assert len({path.resolve() for path in paths}) == 4
    assert set(metrics["artifact_formats"]) == set(metrics["zip_deflated_bytes"]) == {path.name for path in paths}
    assert metrics["compression_method"] == "ZIP_DEFLATED"
    assert all(path.is_file() and metrics["zip_deflated_bytes"][path.name] > 0 for path in paths)
    training_payload = torch.load(training_path, map_location="cpu", weights_only=True)
    assert training_payload["optimizer_state"]["state"]
    assert training_payload["epoch"] == 0
