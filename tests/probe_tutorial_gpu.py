"""Record a real GPU calculation in the isolated Linux tutorial environment."""
import json
import os
import runpy
import sys
from pathlib import Path

os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
root = Path(__file__).resolve().parents[1]
runpy.run_path(str(root / "tutorial/colab_support/bootstrap_tutorials.py"))
import tensorflow as tf  # noqa: E402 - bootstrap configures legacy Keras first.

devices = tf.config.list_physical_devices("GPU")
assert devices, "No TensorFlow GPU is available."
with tf.device("/GPU:0"):
    result = tf.linalg.matmul(tf.ones((128, 128)), tf.ones((128, 128)))
assert "GPU:0" in result.device
tf.debugging.assert_equal(result, tf.fill((128, 128), 128.0))
manifest = {
    "python": sys.version,
    "tensorflow": tf.__version__,
    "gpu_devices": [device.name for device in devices],
    "calculation_device": result.device,
    "build": tf.sysconfig.get_build_info(),
}
output = root / ".codex/validation/tutorial-suite/wsl-gpu-environment.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
print(json.dumps(manifest, indent=2, default=str))
