"""Execute every tutorial-1 cell in separate CPU smoke mode; retain prior runs."""
import hashlib
import json
import os
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "tutorial/1_simple_1d_wave_pytorch.ipynb"
    output = ROOT / "dev_tests/artifacts/tutorial1_gc/smoke"
    output.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        NIF_SMOKE="1", NIF_DEVICE="cpu", NIF_OUTPUT_DIR=str(output), MPLBACKEND="Agg"
    )
    notebook = nbformat.read(path, as_version=4)
    started = time.monotonic()
    result = dict(
        notebook=path.name,
        mode="smoke",
        device="cpu",
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    try:
        NotebookClient(
            notebook,
            timeout=600,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
        code = [c for c in notebook.cells if c.cell_type == "code" and c.source.strip()]
        assert all(c.execution_count is not None for c in code)
        result.update(
            status="passed",
            executed_code_cells=len(code),
            metrics=json.loads((output / "tutorial1/metrics.json").read_text()),
            artifacts=sorted(p.name for p in (output / "tutorial1").iterdir()),
        )
        assert result["metrics"]["adabelief_epochs"] == 2
        assert result["metrics"]["lbfgs_rounds"] == 1
        assert result["metrics"]["lbfgs_max_iter"] == 2
        assert {
            "traveling_wave.pt",
            "fine_tuned.pt",
            "parameter_to_latent.pt",
            "latent.png",
            "fine_tune_loss.png",
            "fine_tuned_reconstruction.png",
        } <= set(result["artifacts"])
    except Exception as exc:
        result.update(status="failed", error=str(exc))
    result["seconds"] = time.monotonic() - started
    nbformat.write(notebook, output / path.name)
    (ROOT / "dev_tests/tutorial1_gc_smoke_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
