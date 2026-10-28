"""Public CPU regressions for tutorial 7's original scientific diagnostics.

Execute the notebook's output/export cells with controlled matrices and a tiny
model. These checks do not train the tutorial or establish full convergence.
"""
import contextlib
import copy
import io
import json
import os
import time
import zipfile
from pathlib import Path

# CPU and plotting isolation must be configured before numerical imports.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
STEM = "7_model_pruning_and_quantization"


def notebook(suffix=""):
    return json.loads(
        (ROOT / "tutorial" / (STEM + suffix + ".ipynb")).read_text(encoding="utf-8")
    )


def cell(nb, index):
    return "".join(nb["cells"][index]["source"])


@pytest.fixture
def predictions():
    # Two nonzero singular values distinguish the spectral and Frobenius norms.
    baseline = np.zeros((200, 10))
    baseline[0, 0], baseline[1, 1] = 4.0, 3.0
    pruned = baseline.copy()
    pruned[0, 0] += 2.0
    pruned[1, 1] += 2.0
    quantized = baseline.copy()
    quantized[0, 0] += 1.0
    quantized[1, 1] += 3.0
    return dict(np=np, y_pred_unpruned=baseline, y_pred=pruned, y_pred_quant=quantized)


def collect_metrics(scope):
    captured = {}
    scope = dict(
        scope,
        epochs=lambda n: n,
        nepoch=5000,
        run_pruning_epochs=50000,
        sparsity=0.9,
        zero_count=9,
        weight_count=10,
        quantized_linears=[object()],
        write_metrics=captured.update,
    )
    for key in ("original_file", "pruned_file", "exported_file", "quantized_file"):
        scope.setdefault(key, Path(key + ".pt"))
    scope.setdefault("get_gzipped_model_size", lambda path: 1)
    exec(cell(notebook("_pytorch"), 37), scope)
    return captured


@pytest.mark.parametrize(
    "key,prediction",
    [
        ("relative_mse_prune", "y_pred"),
        ("relative_mse_prune_quant", "y_pred_quant"),
    ],
)
def test_native_legacy_relative_mse_matches_reference_spectral_norm(
    predictions, key, prediction
):
    metrics = collect_metrics(predictions)
    baseline = predictions["y_pred_unpruned"]
    expected = (
        np.linalg.norm(predictions[prediction] - baseline, ord=2) ** 2
        / np.linalg.norm(baseline, ord=2) ** 2
    )
    assert metrics[key] == pytest.approx(expected, abs=1e-12)


def test_colab_quantized_relative_mse_matches_original_cell44(predictions):
    original = eval(cell(notebook(), 44), dict(predictions))
    modern = eval(cell(notebook("_colab"), 44), dict(predictions))
    assert modern == pytest.approx(original, abs=1e-12)


def test_colab_reported_relative_mse_matches_original_cell47(predictions):
    outputs = []
    for suffix in ("", "_colab"):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(cell(notebook(suffix), 47), dict(predictions))
        outputs.append(
            [float(line.split(" is ")[1]) for line in output.getvalue().splitlines()]
        )
    assert outputs[1] == pytest.approx(outputs[0], abs=1e-12)


def render_comparison(source, predictions, tmp_path, monkeypatch):
    records = []
    original_contourf = Axes.contourf

    def record(axis, t, x, values, *args, **kwargs):
        records.append((np.array(t), np.array(x), np.array(values), kwargs.copy()))
        return original_contourf(axis, t, x, values, *args, **kwargs)

    t, x = np.linspace(-1, 1, 10), np.linspace(-1, 1, 200)
    tt, xx = np.meshgrid(t, x)
    scope = dict(predictions, plt=plt, tt=tt, xx=xx, OUT=tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(Axes, "contourf", record)
        patch.setattr(plt, "show", lambda: None)
        exec(source, scope)
    plt.close("all")
    return records


def test_native_six_panel_values_and_grid_match_original(
    predictions, tmp_path, monkeypatch
):
    reference = render_comparison(
        cell(notebook(), 46), predictions, tmp_path, monkeypatch
    )
    native = render_comparison(
        cell(notebook("_pytorch"), 36), predictions, tmp_path, monkeypatch
    )
    assert len(reference) == len(native) == 6
    for index, (expected, actual) in enumerate(zip(reference, native)):
        for name, left, right in zip(("t", "x", "field"), expected[:3], actual[:3]):
            assert right.shape == (200, 10)
            np.testing.assert_array_equal(right, left, err_msg=f"panel {index}, {name}")


def test_native_six_panel_levels_and_colormap_match_original(
    predictions, tmp_path, monkeypatch
):
    reference = render_comparison(
        cell(notebook(), 46), predictions, tmp_path, monkeypatch
    )
    native = render_comparison(
        cell(notebook("_pytorch"), 36), predictions, tmp_path, monkeypatch
    )
    assert len(reference) == len(native) == 6
    for expected, actual in zip(reference, native):
        np.testing.assert_array_equal(actual[3]["levels"], expected[3]["levels"])
        assert actual[3]["cmap"] == expected[3]["cmap"] == "seismic"


class TinyModel(nn.Module):
    """Cheap stand-in with the notebook checkpoint factory's public signature."""

    def __init__(
        self, cfg_shape_net=None, cfg_parameter_net=None, mixed_policy="float32"
    ):
        super().__init__()
        self.cfg_shape_net = cfg_shape_net or {"input_dim": 2}
        self.cfg_parameter_net = cfg_parameter_net or {"output_dim": 1}
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def tiny_export_scope(tmp_path):
    torch.manual_seed(0)
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model(torch.tensor([[0.2, 0.3]])).square().sum().backward()
    optimizer.step()
    assert optimizer.state, "The control must contain actual Adam slots."
    scope = dict(
        torch=torch,
        nn=nn,
        np=np,
        plt=plt,
        Path=Path,
        copy=copy,
        time=time,
        json=json,
        SMOKE=True,
        device=torch.device("cpu"),
        OUT=tmp_path,
        model=model,
        optimizer=optimizer,
    )
    exec(cell(notebook("_pytorch"), 2), scope)
    exec(cell(notebook("_pytorch"), 8), scope)
    exec(cell(notebook("_pytorch"), 10), scope)
    scope["model_for_pruning"] = copy.deepcopy(scope["model"])
    scope["optimizer_prune"] = optimizer
    scope["pruning_controller"] = type(
        "AlreadyStripped", (), {"strip": lambda self: None}
    )()
    exec(cell(notebook("_pytorch"), 28), scope)
    return scope


@pytest.mark.parametrize("filename", ["original_file", "pruned_file"])
def test_size_benchmark_state_dict_artifacts_exclude_training_state(
    tiny_export_scope, filename
):
    scope = tiny_export_scope
    path = scope[filename]
    assert path.is_file()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    forbidden = {
        "optimizer_state",
        "epoch",
        "smoke",
        "pruning_state",
        "controller_state",
        "mask",
    }
    assert not forbidden.intersection(
        payload
    ), f"Inference artifact contains training state: {sorted(forbidden.intersection(payload))}"
    restored = TinyModel()
    restored.load_state_dict(payload["model_state"])
    example = torch.tensor(scope["input_feature"], dtype=torch.float32)
    with torch.no_grad():
        torch.testing.assert_close(
            restored(example), scope["model"](example), rtol=0, atol=0
        )


def test_training_checkpoint_separately_retains_optimizer_state(
    tiny_export_scope, tmp_path
):
    scope = tiny_export_scope
    path = tmp_path / "training_latest.pt"
    scope["save_checkpoint"](scope["model"], scope["optimizer"], path, 7)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["optimizer_state"]["state"]
    assert payload["epoch"] == 7


def test_compressed_size_uses_original_zip_deflated_method(tmp_path):
    path = tmp_path / "inference.pt"
    path.write_bytes(bytes(range(256)) * 8)
    scope = dict(Path=Path)
    exec(cell(notebook("_pytorch"), 31), scope)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(path)
    assert scope["get_gzipped_model_size"](path) == len(stream.getvalue())
