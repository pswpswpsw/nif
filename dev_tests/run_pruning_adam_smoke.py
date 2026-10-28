"""Separate CPU smoke outputs; original full scientific defaults are unchanged."""
import hashlib
import json
import os
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
out = root / "dev_tests/artifacts/pruning_adam/smoke"
out.mkdir(parents=True, exist_ok=True)
os.environ.update(
    NIF_SMOKE="1", NIF_DEVICE="cpu", NIF_OUTPUT_DIR=str(out), MPLBACKEND="Agg"
)
summary = []
for number in (2, 5, 6, 7, 8):
    path = next((root / "tutorial").glob(f"{number}_*_pytorch.ipynb"))
    notebook = nbformat.read(path, as_version=4)
    start = time.monotonic()
    result = dict(
        notebook=path.name,
        mode="smoke",
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    try:
        NotebookClient(
            notebook,
            timeout=600,
            kernel_name="python3",
            resources={"metadata": {"path": str(root)}},
        ).execute()
        result["status"] = "passed"
    except Exception as exc:
        result.update(status="failed", error=str(exc))
    result["seconds"] = time.monotonic() - start
    nbformat.write(notebook, out / path.name)
    summary.append(result)
    (out / "execution_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(result), flush=True)
if any(r["status"] != "passed" for r in summary):
    raise SystemExit(1)
