"""Create tutorial 8's compatibility copy and preserve its scientific settings."""
import copy
import hashlib
import json
from pathlib import Path

from notebook_metadata import clear_execution_metadata

ROOT = Path(__file__).resolve().parents[1]
STEM = "8_revisit_multi_scale_NIF_with_sobolov_training"


def main():
    source = ROOT / "tutorial" / (STEM + ".ipynb")
    output = ROOT / "tutorial" / (STEM + "_colab.ipynb")
    notebook = copy.deepcopy(json.loads(source.read_text(encoding="utf-8")))
    bootstrap = (
        ROOT / "tutorial" / "colab_support" / "bootstrap_tutorials.py"
    ).read_text(encoding="utf-8")
    notebook["cells"][0]["source"] = bootstrap.splitlines(keepends=True)
    prediction = "".join(notebook["cells"][36]["source"])
    prediction = prediction.replace(
        ".reshape(10,x_new_res)", ".reshape(10,x_new_res)*tw.std[2]+tw.mean[2]"
    )
    notebook["cells"][36]["source"] = prediction.splitlines(keepends=True)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
            for key in ("executionInfo", "outputId"):
                cell.get("metadata", {}).pop(key, None)
    notebook["metadata"].setdefault("colab", {})["name"] = output.name
    notebook["metadata"]["nif_colab_compatibility"] = {
        "original_file": source.name,
        "original_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "original_training_epochs": [50000, 50000],
    }
    clear_execution_metadata(notebook)
    output.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"copy": str(output), **notebook["metadata"]["nif_colab_compatibility"]}
        )
    )


if __name__ == "__main__":
    main()
