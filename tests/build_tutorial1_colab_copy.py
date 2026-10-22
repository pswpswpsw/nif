"""Create the reviewable notebook copy without overwriting tutorial 1."""
import copy
import hashlib
import json
from pathlib import Path

from notebook_metadata import clear_execution_metadata

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "tutorial" / "1_simple_1d_wave.ipynb"
output = ROOT / "tutorial" / "1_simple_1d_wave_colab.ipynb"
notebook = json.loads(source.read_text(encoding="utf-8"))
original_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
notebook = copy.deepcopy(notebook)
bootstrap = (ROOT / "tutorial" / "colab_support" / "bootstrap_tutorial1.py").read_text(
    encoding="utf-8"
)
notebook["cells"][0]["source"] = bootstrap.splitlines(keepends=True)
# GPU detection should also succeed on a CPU-only Colab session.
notebook["cells"][4]["source"] = [
    "import shutil, subprocess\n",
    "if shutil.which('nvidia-smi'):\n",
    "    subprocess.run(['nvidia-smi', '-L'], check=True)\n",
    "else:\n",
    "    print('No NVIDIA GPU attached; this tutorial also runs on CPU.')\n",
]
# This Adam instance is only used when compiling the restored inference model.
# The training optimizer remains the original AdaBelief implementation.
restoration = "".join(notebook["cells"][37]["source"])
notebook["cells"][37]["source"] = restoration.replace(
    "tf.keras.optimizers.Adam(1e-5)", "tf.keras.optimizers.legacy.Adam(1e-5)"
).splitlines(keepends=True)
notebook["cells"][39]["source"] += [
    "\n",
    "np.testing.assert_allclose(\n",
    "    model_opt.predict(train_data[:, :2], verbose=0),\n",
    "    new_model.predict(train_data[:, :2], verbose=0),\n",
    "    rtol=1e-5, atol=1e-6,\n",
    ")\n",
    "print('Checkpoint restore: predictions match.')\n",
]
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
        for name in ("executionInfo", "outputId"):
            cell.get("metadata", {}).pop(name, None)
notebook["metadata"]["colab"]["name"] = output.name
# Default to the CPU runtime used for the live environment probe and validation.
notebook["metadata"].pop("accelerator", None)
notebook["metadata"]["nif_colab_compatibility"] = {
    "original_file": source.name,
    "original_sha256": original_sha256,
    "original_training_epochs": 5000,
    "original_lbfgs_rounds": 200,
    "original_lbfgs_max_iter": 1000,
}
clear_execution_metadata(notebook)
output.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
)
print(
    json.dumps(
        {
            "copy": str(output),
            "original_sha256": original_sha256,
            "code_cells": sum(c["cell_type"] == "code" for c in notebook["cells"]),
        }
    )
)
