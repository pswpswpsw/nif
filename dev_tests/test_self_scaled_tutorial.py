"""Public regressions for the portable experimental self-scaled tutorial."""
import hashlib
import importlib.util
import json
import os
import py_compile
import subprocess
import sys
import tempfile
import venv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def support_module(name):
    path = ROOT / "examples/self_scaled" / (name + ".py")
    spec = importlib.util.spec_from_file_location("nif_ss_public_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_source_rejects_tampered_cache_without_executing(tmp_path):
    source = support_module("source")
    cached = tmp_path / source.AUTHOR_COMMIT / "_optimize.py"
    cached.parent.mkdir()
    sentinel = tmp_path / "executed"
    cached.write_text(f"open({str(sentinel)!r}, 'w').write('bad')")
    with pytest.raises(ValueError, match="SHA256"):
        source.load_author(tmp_path)
    assert not sentinel.exists()


def test_runtime_source_executes_verified_bytes_instead_of_cached_bytecode(
    tmp_path, monkeypatch
):
    source = support_module("source")
    path = tmp_path / "_optimize.py"
    path.write_bytes(b"MARKER = 'cached'\n")
    original_stat = path.stat()
    py_compile.compile(str(path), doraise=True)
    verified = b"MARKER = 'source'\n"
    path.write_bytes(verified)
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    monkeypatch.setattr(source, "AUTHOR_BYTES", len(verified))
    monkeypatch.setattr(source, "AUTHOR_SHA256", hashlib.sha256(verified).hexdigest())
    monkeypatch.setattr(source, "fetch_source", lambda cache_dir: path)
    module = source.load_author(tmp_path)
    assert module.MARKER == "source"


def test_original_grid_and_physical_inverse_are_preserved():
    benchmark = support_module("benchmark")
    data = benchmark.dataset()
    assert data["train_features"].shape == (2000, 2)
    assert data["train_target"].shape == (2000, 1)
    np.testing.assert_allclose(data["train_t"][:, 0], np.arange(10) * 10)
    np.testing.assert_allclose(data["train_x"][0], np.arange(200) * 0.005, atol=3e-8)
    restored = data["train_target"].ravel() * data["std"][-1] + data["mean"][-1]
    np.testing.assert_allclose(restored, data["train_truth_physical"], atol=8e-9)
    dense_raw = data["dense_features"] * data["std"][:2] + data["mean"][:2]
    np.testing.assert_allclose(dense_raw[:, 0], data["dense_t"].ravel(), atol=1e-14)
    np.testing.assert_allclose(dense_raw[:, 1], data["dense_x"].ravel(), atol=1e-14)


def test_plot_grid_rejects_shape_compatible_coordinate_permutation():
    report = support_module("report")
    xx, tt = np.meshgrid([0.0, 0.3, 0.9, 1.0], [0.0, 2.0, 7.0])
    got_t, got_x = report.coordinate_grid(tt, xx, (3, 4))
    np.testing.assert_array_equal(100 * got_t + got_x, 100 * tt + xx)
    with pytest.raises(ValueError, match="time-major"):
        report.coordinate_grid(tt.T.ravel(), xx.T.ravel(), (3, 4))


def test_nonfinite_dense_state_cannot_be_success():
    benchmark = support_module("benchmark")
    result = SimpleNamespace(
        x=np.ones(2),
        jac=np.zeros(2),
        fun=0.0,
        hess_inv=np.array([[1.0, 0.0], [0.0, np.nan]]),
        success=True,
        status=0,
        message="success",
        nit=1,
        nfev=2,
        njev=2,
    )
    status = benchmark.author_status(result)
    assert status["backend_success"]
    assert not status["state_finite"]
    assert not status["converged"]
    assert status["termination"] == "nonfinite_optimizer_state"


def test_real_lbfgs_short_run_and_saved_initial_state(tmp_path):
    benchmark = support_module("benchmark")
    summary = benchmark.run_experiment(
        tmp_path / "real-run",
        warmup_epochs=1,
        max_iterations=2,
        methods=("lbfgs",),
        render=False,
    )
    arm = summary["arms"]["lbfgs"]
    assert summary["mode"] == "CUSTOM"
    assert summary["shared_start_verified"]
    assert arm["evaluations"] >= arm["iterations"]
    assert arm["iterations"] > 0
    assert arm["train"]["rmse"] >= 0
    manifest = json.loads((tmp_path / "real-run/manifest.json").read_text())
    checkpoint = torch.load(tmp_path / "real-run/shared_start.pt", weights_only=True)
    assert checkpoint["parameter_hash"] == manifest["shared_start_parameter_hash"]
    with np.load(tmp_path / "real-run/lbfgs_parameters.npz") as values:
        assert benchmark.array_hash(values["initial"]) == checkpoint["parameter_hash"]
    with pytest.raises(FileExistsError):
        benchmark.run_experiment(
            tmp_path / "real-run", methods=("lbfgs",), render=False
        )


def test_cli_fails_when_requested_backend_raises_after_shared_initial_check(
    tmp_path, monkeypatch
):
    benchmark = support_module("benchmark")
    original_support = benchmark.support_module

    def broken_memo(_function):
        raise RuntimeError("Deliberately unavailable author backend")

    adapter = SimpleNamespace(
        load_author=lambda _cache: SimpleNamespace(MemoizeJac=broken_memo),
        provenance=lambda: {"test": "backend failure"},
    )
    monkeypatch.setattr(
        benchmark,
        "support_module",
        lambda name: adapter if name == "source" else original_support(name),
    )
    output = tmp_path / "failed-run"
    monkeypatch.setattr(
        sys, "argv", ["benchmark.py", "--smoke", "--no-render", "--output", str(output)]
    )
    with pytest.raises(RuntimeError, match="invalid"):
        benchmark.main()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["shared_start_verified"]
    assert summary["arms"]["ssbfgs"]["status"]["termination"] == "exception"
    assert not summary["execution_valid"]


def test_notebook_report_display_works_with_non_utf8_default_locale(
    tmp_path, monkeypatch
):
    notebook = json.loads(
        (ROOT / "tutorial/9_self_scaled_optimizers_pytorch.ipynb").read_text(
            encoding="utf-8"
        )
    )
    display_source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "report_text =" in "".join(cell["source"])
    )
    (tmp_path / "summary.json").write_text(
        json.dumps(
            dict(
                mode="SMOKE",
                shared_start_verified=True,
                warmup_elapsed_seconds=0.1,
                arms={},
            )
        ),
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(dict(shared_start_parameter_hash="fixture", threads=2)),
        encoding="utf-8",
    )
    (tmp_path / "REPORT.md").write_text("√", encoding="utf-8")
    read_text = Path.read_text

    def localized_read(path, encoding=None, errors=None):
        return read_text(path, encoding=encoding or "cp936", errors=errors)

    monkeypatch.setattr(Path, "read_text", localized_read)
    exec(
        compile(display_source, "tutorial9-report-display", "exec"),
        dict(RUN_DIR=tmp_path, MODE="SMOKE", json=json),
    )


def test_notebook_bootstrap_without_ensurepip_targets_only_child_environment(
    tmp_path, monkeypatch
):
    notebook = json.loads(
        (ROOT / "tutorial/9_self_scaled_optimizers_pytorch.ipynb").read_text(
            encoding="utf-8"
        )
    )
    bootstrap = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "venv.EnvBuilder" in "".join(cell["source"])
    )
    real_builder = venv.EnvBuilder
    real_output = subprocess.check_output
    created = []
    installs = []

    def without_ensurepip(*args, **kwargs):
        if kwargs.get("with_pip"):
            raise subprocess.CalledProcessError(1, ["python", "-m", "ensurepip"])
        created.append(kwargs)
        return real_builder(*args, **kwargs)

    def capture_installs(command, **kwargs):
        assert "install" in command
        assert command[:4] == [sys.executable, "-m", "pip", "--python"]
        child = Path(command[4]).resolve()
        assert child.is_relative_to(tmp_path.resolve())
        assert child != Path(sys.executable).resolve()
        installs.append(command)
        return 0

    def bootstrap_output(command, **kwargs):
        if "import json,sys,nif,torch" in " ".join(command):
            return json.dumps(dict(nif="2.0.0", numpy="2.1.3", scipy="1.16.3"))
        return real_output(command, **kwargs)

    monkeypatch.setenv("NIF_SS_LOCAL_REPO", str(ROOT))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(venv, "EnvBuilder", without_ensurepip)
    monkeypatch.setattr(subprocess, "check_call", capture_installs)
    monkeypatch.setattr(subprocess, "check_output", bootstrap_output)
    namespace = dict(os=os, SMOKE=True, MODE="SMOKE")
    exec(compile(bootstrap, "tutorial9-bootstrap", "exec"), namespace)
    assert created and all(not item.get("with_pip") for item in created)
    assert len(installs) == 3
    assert all(command[4] == str(namespace["SS_PYTHON"]) for command in installs)
