"""Build a compatibility copy while leaving tutorial 7's scientific code intact."""
import copy
import hashlib
import json
from pathlib import Path

from notebook_metadata import clear_execution_metadata

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "tutorial" / "7_model_pruning_and_quantization.ipynb"
output = source.with_name(source.stem + "_colab.ipynb")
notebook = copy.deepcopy(json.loads(source.read_text(encoding="utf-8")))
bootstrap = (ROOT / "tutorial" / "colab_support" / "bootstrap_tutorials.py").read_text(
    encoding="utf-8"
)
notebook["cells"][0]["source"] = bootstrap.splitlines(keepends=True)
for index in (44, 47):
    notebook["cells"][index]["source"] = (
        "".join(notebook["cells"][index]["source"])
        .replace("ord=2", "ord='fro'")
        .splitlines(keepends=True)
    )
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
        for key in ("executionInfo", "outputId"):
            cell.get("metadata", {}).pop(key, None)
notebook.setdefault("metadata", {}).setdefault("colab", {})["name"] = output.name
notebook["metadata"]["nif_colab_compatibility"] = {
    "original_file": source.name,
    "original_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "original_training_epochs": 5000,
    "original_pruning_epochs": 50000,
    "modified_code_cells": [0, 44, 47],
}
clear_execution_metadata(notebook)
output.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
)
print(
    json.dumps(
        {
            "copy": str(output),
            "original_sha256": notebook["metadata"]["nif_colab_compatibility"][
                "original_sha256"
            ],
            "code_cells": sum(
                cell["cell_type"] == "code" for cell in notebook["cells"]
            ),
        }
    )
)
