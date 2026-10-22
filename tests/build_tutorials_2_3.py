"""Build compatibility copies without altering tutorial 2 or 3 research cells."""
import copy
import hashlib
import json
from pathlib import Path

from notebook_metadata import clear_execution_metadata

ROOT = Path(__file__).resolve().parents[1]
STEMS = ("2_multi_scale_NIF", "3_multi_scale_linear_NIF")


def main():
    bootstrap = (
        ROOT / "tutorial" / "colab_support" / "bootstrap_tutorials.py"
    ).read_text(encoding="utf-8")
    for stem in STEMS:
        source = ROOT / "tutorial" / (stem + ".ipynb")
        target = source.with_name(stem + "_colab.ipynb")
        notebook = copy.deepcopy(json.loads(source.read_text(encoding="utf-8")))
        notebook["cells"][0]["source"] = bootstrap.splitlines(keepends=True)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
                for key in ("executionInfo", "outputId"):
                    cell.get("metadata", {}).pop(key, None)
        notebook["metadata"]["colab"]["name"] = target.name
        notebook["metadata"].pop("accelerator", None)
        notebook["metadata"]["nif_colab_compatibility"] = {
            "original_file": source.name,
            "original_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "modified_source_cells": [0],
        }
        clear_execution_metadata(notebook)
        target.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "notebook": target.name,
                    "original_sha256": notebook["metadata"]["nif_colab_compatibility"][
                        "original_sha256"
                    ],
                    "code_cells": sum(
                        c["cell_type"] == "code" for c in notebook["cells"]
                    ),
                }
            )
        )


if __name__ == "__main__":
    main()
