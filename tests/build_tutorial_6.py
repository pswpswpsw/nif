"""Build tutorial 6's install-only compatibility copy; preserve the source."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from notebook_metadata import clear_execution_metadata

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tutorial" / "6_revisit_multi_scale_NIF_with_L1_L2_regularization.ipynb"
OUTPUT = SOURCE.with_name(SOURCE.stem + "_colab.ipynb")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-output", type=Path)
    args = parser.parse_args()
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    notebook["cells"][0]["source"] = (
        (ROOT / "tutorial" / "colab_support" / "bootstrap_tutorials.py")
        .read_text(encoding="utf-8")
        .splitlines(keepends=True)
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
            for name in ("executionInfo", "outputId"):
                cell.get("metadata", {}).pop(name, None)
    notebook["metadata"].setdefault("colab", {})["name"] = OUTPUT.name
    notebook["metadata"]["nif_colab_compatibility"] = {
        "original_file": SOURCE.name,
        "original_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "original_training_epochs": 3000,
        "original_training_runs": 9,
    }
    clear_execution_metadata(notebook)
    OUTPUT.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    if args.smoke_output:
        smoke = copy.deepcopy(notebook)
        for cell in smoke["cells"]:
            if cell["cell_type"] == "code":
                cell["source"] = (
                    "".join(cell["source"])
                    .replace("nepoch = 3000", "nepoch = 2")
                    .splitlines(keepends=True)
                )
        smoke["metadata"]["nif_qa_execution"] = {"mode": "smoke", "epochs_per_run": 2}
        args.smoke_output.parent.mkdir(parents=True, exist_ok=True)
        args.smoke_output.write_text(
            json.dumps(smoke, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "copy": str(OUTPUT),
                "code_cells": sum(c["cell_type"] == "code" for c in notebook["cells"]),
                "training_runs": 9,
                "epochs_per_run": 3000,
            }
        )
    )


if __name__ == "__main__":
    main()
