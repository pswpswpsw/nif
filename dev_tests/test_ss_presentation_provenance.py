"""Editorial notebook changes retain the executed scientific code and figures."""
import base64
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "docs/source/_static/self_scaled"


def test_tutorial9_editorial_revision_preserves_executed_science():
    source = ROOT / "tutorial/9_self_scaled_optimizers_pytorch.ipynb"
    notebook = json.loads(source.read_bytes())
    receipt = json.loads((STATIC / "tutorial9_validation.json").read_bytes())
    revision = json.loads((STATIC / "tutorial9_presentation.json").read_bytes())
    assert revision["previous_notebook_sha256"] == receipt["published_notebook_sha256"]
    assert (
        revision["current_notebook_sha256"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    code = {
        cell["id"]: hashlib.sha256("".join(cell["source"]).encode()).hexdigest()
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    }
    assert code == receipt["published_code_cell_hashes_by_id"]
    assert revision["code_cell_hashes_by_id"] == code
    pngs = []
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            png = output.get("data", {}).get("image/png")
            if png:
                pngs.append(hashlib.sha256(base64.b64decode("".join(png))).hexdigest())
    assert pngs == receipt["published_png_sha256"]
    assert revision["png_sha256"] == pngs
    assert revision["additional_optimization"] is False
