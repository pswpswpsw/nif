"""Native runner selection and output isolation, using tiny fake kernel clients."""
import importlib.util
import json
import os
from pathlib import Path

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]


def windows_long_path(path):
    return Path("\\\\?\\" + str(path)) if os.name == "nt" else path


def helper():
    path = ROOT / "dev_tests/run_native_notebooks.py"
    assert "def run(" in path.read_text(encoding="utf-8"), "Runner must be import-safe"
    spec = importlib.util.spec_from_file_location("nif_native_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checkout(tmp_path):
    root = windows_long_path(tmp_path / "checkout")
    (root / "tutorial").mkdir(parents=True)
    catalog = json.loads((ROOT / "tutorial/catalog.json").read_text(encoding="utf-8"))
    (root / "tutorial/catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    for entry in catalog["tutorials"]:
        for key in ("notebook", "original", "compatibility"):
            if key in entry:
                (root / entry[key]).parent.mkdir(parents=True, exist_ok=True)
                notebook = nbformat.v4.new_notebook(
                    cells=[nbformat.v4.new_code_cell("pass")]
                )
                nbformat.write(notebook, root / entry[key])
    return root


class FakeClient:
    calls = []

    def __init__(self, notebook, **kwargs):
        self.notebook = notebook
        self.calls.append(
            (
                kwargs,
                {
                    key: os.environ.get(key)
                    for key in ("NIF_SMOKE", "NIF_OUTPUT_DIR", "MPLBACKEND")
                },
            )
        )

    def execute(self):
        self.notebook.cells[0].execution_count = 1
        return self.notebook


def test_catalog_drives_default_scope_and_explicit_subset_order(checkout):
    runner = helper()
    assert [entry["id"] for entry in runner.select_tutorials(checkout)] == list(
        range(1, 9)
    )
    assert [entry["id"] for entry in runner.select_tutorials(checkout, "3,1")] == [3, 1]


@pytest.mark.parametrize("numbers", ["9", "1,1", "", "1,x", "-1"])
def test_invalid_selection_creates_no_output(checkout, tmp_path, numbers):
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError):
        helper().run(root=checkout, numbers=numbers, output=output)
    assert not output.exists()


def test_unique_outputs_preserve_previous_runs_and_kernel_contract(
    checkout, monkeypatch
):
    runner = helper()
    FakeClient.calls = []
    monkeypatch.setattr(runner, "NotebookClient", FakeClient)
    monkeypatch.setenv("NIF_SMOKE", "caller-value")
    old = checkout / "dev_tests/artifacts/smoke/execution_summary.json"
    old.parent.mkdir(parents=True)
    old.write_text("preserve previous scientific record", encoding="utf-8")
    first, summary = runner.run(root=checkout, numbers="1")
    second, _ = runner.run(root=checkout, numbers="1", full=True)
    assert (
        first != second
        and first.name.startswith("smoke-")
        and second.name.startswith("full-")
    )
    assert old.read_text(encoding="utf-8") == "preserve previous scientific record"
    assert summary[0]["status"] == "passed"
    assert os.environ["NIF_SMOKE"] == "caller-value"
    for index, (kwargs, environment) in enumerate(FakeClient.calls):
        assert kwargs == dict(
            timeout=7200,
            kernel_name="python3",
            resources={"metadata": {"path": str(checkout)}},
        )
        assert environment["NIF_SMOKE"] == str(1 - index)
        assert environment["NIF_OUTPUT_DIR"] == str((first, second)[index])
        assert environment["MPLBACKEND"] == "Agg"


def test_explicit_output_rejects_old_results_and_accepts_empty_directory(
    checkout, tmp_path, monkeypatch
):
    runner = helper()
    monkeypatch.setattr(runner, "NotebookClient", FakeClient)
    existing = windows_long_path(tmp_path / "existing")
    existing.mkdir()
    sentinel = existing / "previous.txt"
    sentinel.write_text("original", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        runner.run(root=checkout, numbers="1", output=existing)
    assert sentinel.read_text(encoding="utf-8") == "original"
    empty = windows_long_path(tmp_path / "empty")
    empty.mkdir()
    output, _ = runner.run(root=checkout, numbers="1", output=empty)
    assert output == empty.resolve()
    assert (empty / "execution_summary.json").is_file()


def test_failure_records_evidence_and_restores_environment(
    checkout, tmp_path, monkeypatch
):
    runner = helper()

    class FailingClient(FakeClient):
        def execute(self):
            raise RuntimeError("intentional kernel failure")

    monkeypatch.setattr(runner, "NotebookClient", FailingClient)
    monkeypatch.setenv("NIF_OUTPUT_DIR", "caller-directory")
    output, summary = runner.run(
        root=checkout, numbers="1", output=windows_long_path(tmp_path / "failed")
    )
    assert summary[0]["status"] == "failed"
    assert "intentional kernel failure" in summary[0]["error"]
    assert (output / summary[0]["notebook"]).is_file()
    assert len(list(output.glob("*.error.txt"))) == 1
    assert (
        json.loads((output / "execution_summary.json").read_text())[0]["status"]
        == "failed"
    )
    assert os.environ["NIF_OUTPUT_DIR"] == "caller-directory"
