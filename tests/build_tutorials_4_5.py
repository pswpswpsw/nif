"""Build copy-preserving modern-Colab variants of tutorials 4 and 5."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path

from notebook_metadata import clear_execution_metadata

ROOT = Path(__file__).resolve().parents[1]
NAMES = (
    "4_get_gradients_by_wrapping_model_with_layer",
    "5_large_scale_training_on_tensorflow_record_data",
)


def build():
    bootstrap = (ROOT / "tutorial/colab_support/bootstrap_tutorials.py").read_text(
        encoding="utf-8"
    )
    for name in NAMES:
        original = ROOT / "tutorial" / (name + ".ipynb")
        repaired = deepcopy(json.loads(original.read_text(encoding="utf-8")))
        repaired["cells"][0]["source"] = bootstrap.splitlines(keepends=True)
        if name.startswith("5_"):
            listing = (
                "from pathlib import Path\n"
                'for path in sorted(Path("TFR_dir").glob("*.tfrecord"), '
                "key=lambda path: path.stat().st_mtime):\n"
                '    print(path.name, path.stat().st_size, "bytes")'
            )
            repaired["cells"][13]["source"] = listing.splitlines(keepends=True)
        for cell in repaired["cells"]:
            if cell["cell_type"] == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
        repaired["metadata"]["nif_colab_compatibility"] = {
            "original_notebook": original.name,
            "original_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
            "nif_commit": "ad9113df14590189b12068b6b3fa7c8c8d9187d4",
        }
        destination = original.with_name(original.stem + "_colab.ipynb")
        clear_execution_metadata(repaired)
        destination.write_text(
            json.dumps(repaired, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        print(destination)


if __name__ == "__main__":
    build()
