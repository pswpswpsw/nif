"""Build a browser-pastable, full-settings Colab validation harness.

The payload contains notebook sources, with no saved outputs or attachments.
The delivered tutorials themselves are not modified by this QA harness.
"""
import base64
import hashlib
import json
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".codex/validation/tutorial-suite"


def main():
    notebooks = []
    for number in range(2, 8):
        path = next((ROOT / "tutorial").glob(f"{number}_*_colab.ipynb"))
        original = json.loads(path.read_text(encoding="utf-8"))
        clean = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": []}
        for index, cell in enumerate(original["cells"]):
            copied = {
                "id": f"original-{index}",
                "cell_type": cell["cell_type"],
                "source": "".join(cell["source"]),
                "metadata": {},
            }
            if copied["cell_type"] == "code":
                copied.update(execution_count=None, outputs=[])
            clean["cells"].append(copied)
        notebooks.append(
            {
                "name": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "notebook": clean,
            }
        )
    payload = base64.b64encode(zlib.compress(json.dumps(notebooks).encode())).decode()
    template = r"""import base64, hashlib, json, os, pathlib, subprocess, sys, time, zipfile, zlib
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet", "nbclient==0.11.0", "nbformat==5.11.1"
])
import nbformat
from nbclient import NotebookClient
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"
items = json.loads(zlib.decompress(base64.b64decode("PAYLOAD")))
root = pathlib.Path("/content/nif-tutorial-validation")
root.mkdir(exist_ok=True)
results = []
for item in items:
    name = item["name"]
    work = root / name.removesuffix(".ipynb")
    work.mkdir(exist_ok=True)
    nb = nbformat.from_dict(item["notebook"])
    destination = root / name.replace(".ipynb", ".full.executed.ipynb")
    started = time.monotonic()
    events = []
    def begin(cell, cell_index, **kwargs):
        if cell.cell_type == "code" and cell.source.strip():
            print(f"START {name} cell {cell_index}", flush=True)
    def completed(cell, cell_index, **kwargs):
        if cell.cell_type == "code" and cell.source.strip():
            events.append({"cell": cell_index, "count": cell.execution_count,
                           "elapsed": round(time.monotonic()-started, 2)})
            nbformat.write(nb, destination)
            print(f"DONE {name} cell {cell_index} at {events[-1]['elapsed']}s", flush=True)
    client = NotebookClient(nb, timeout=21600, startup_timeout=120, allow_errors=False,
                            resources={"metadata": {"path": str(work)}},
                            on_cell_start=begin, on_cell_executed=completed)
    try:
        client.execute()
        code = [c for c in nb.cells if c.cell_type == "code" and c.source.strip()]
        assert all(c.execution_count is not None for c in code)
        assert not any(o.output_type == "error" for c in code for o in c.outputs)
        result = {"name": name, "input_sha256": item["sha256"], "full_settings": True,
                  "cells_passed": len(code), "seconds": round(time.monotonic()-started, 2),
                  "python": sys.version, "events": events}
        results.append(result)
        (root / "results.json").write_text(json.dumps(results, indent=2))
        print("FULL PASS", json.dumps(result), flush=True)
    finally:
        nbformat.write(nb, destination)
        (root / (name + ".events.json")).write_text(json.dumps(events, indent=2))
archive = root / "nif-tutorials-2-7-full-colab-validation.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
    for path in root.iterdir():
        if path.is_file() and path != archive:
            output.write(path, path.name)
    for path in root.rglob("*.png"):
        output.write(path, path.relative_to(root))
print("ALL SIX TUTORIALS PASSED AT ORIGINAL SETTINGS", flush=True)
from google.colab import files
files.download(str(archive))
"""
    script = template.replace("PAYLOAD", payload)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "colab-suite-cell.py").write_text(script, encoding="utf-8")
    (OUT / "colab-suite-input-manifest.json").write_text(
        json.dumps(
            [
                {k: value for k, value in entry.items() if k != "notebook"}
                for entry in notebooks
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Built {len(notebooks)} full-settings notebooks, Colab cell {len(script)} characters."
    )


if __name__ == "__main__":
    main()
