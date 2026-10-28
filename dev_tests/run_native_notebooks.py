"""Execute native tutorials in clean kernels; smoke and full outputs are kept separate."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--full", action="store_true")
p.add_argument("--numbers", default="1,2,3,4,5,6,7,8")
a = p.parse_args()
mode = "full" if a.full else "smoke"
artifact_root = ROOT / "dev_tests" / "artifacts" / mode
artifact_root.mkdir(parents=True, exist_ok=True)
os.environ["NIF_SMOKE"] = "0" if a.full else "1"
os.environ["NIF_OUTPUT_DIR"] = str(artifact_root)
os.environ["MPLBACKEND"] = "Agg"
summary = []
for number in a.numbers.split(","):
    path = next((ROOT / "tutorial").glob(number + "_*_pytorch.ipynb"))
    notebook = nbformat.read(path, as_version=4)
    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    started = time.monotonic()
    try:
        NotebookClient(
            notebook,
            timeout=7200,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
        result = {
            "notebook": path.name,
            "status": "passed",
            "seconds": time.monotonic() - started,
        }
    except Exception as exc:
        result = {
            "notebook": path.name,
            "status": "failed",
            "seconds": time.monotonic() - started,
            "error": str(exc),
        }
        (artifact_root / (path.stem + ".error.txt")).write_text(
            str(exc), encoding="utf-8"
        )
    nbformat.write(notebook, artifact_root / path.name)
    result["source_sha256"] = source_sha256
    summary.append(result)
    print(json.dumps(result), flush=True)
    (artifact_root / "execution_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
if any(r["status"] == "failed" for r in summary):
    raise SystemExit(1)
