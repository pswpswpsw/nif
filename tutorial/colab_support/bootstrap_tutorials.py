import importlib.metadata
import os
import subprocess
import sys
import tempfile
from pathlib import Path

if "tensorflow" in sys.modules and os.environ.get("TF_USE_LEGACY_KERAS") != "1":
    raise RuntimeError(
        "Restart the Colab session, then run this notebook from its first cell. "
        "NIF requires the legacy Keras setting before TensorFlow is imported."
    )
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

COLAB_REQUIREMENTS = [
    "tensorflow==2.21.0",
    "tf-keras==2.21.0",
    "keras==3.13.2",
    "tensorflow-probability==0.25.0",
    "tensorflow-model-optimization==0.8.1",
    "numpy==2.1.3",
    "scipy==1.16.3",
    "matplotlib==3.10.0",
    "tensorboard==2.21.0",
    "absl-py==1.4.0",
]
NIF_COMMIT = "ad9113df14590189b12068b6b3fa7c8c8d9187d4"

if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
    raise RuntimeError(
        "Use a Colab Python 3.11–3.13 runtime for this pinned environment."
    )

missing = []
for requirement in COLAB_REQUIREMENTS:
    package, expected = requirement.split("==")
    try:
        installed = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed != expected:
        missing.append(requirement)
if missing:
    package_modules = {
        "tensorflow": "tensorflow",
        "tf-keras": "tf_keras",
        "keras": "keras",
        "numpy": "numpy",
        "scipy": "scipy",
        "matplotlib": "matplotlib",
        "tensorflow-probability": "tensorflow_probability",
        "tensorflow-model-optimization": "tensorflow_model_optimization",
        "tensorboard": "tensorboard",
        "absl-py": "absl",
    }
    changing = {item.split("==")[0] for item in missing}
    loaded = [
        module
        for package, module in package_modules.items()
        if package in changing and module in sys.modules
    ]
    if loaded:
        raise RuntimeError(
            "Restart the Colab session before changing packages; already imported: "
            + ", ".join(loaded)
        )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            *COLAB_REQUIREMENTS,
        ]
    )

runtime_root = Path(tempfile.gettempdir()) / ("nif-tutorials-v2-" + NIF_COMMIT[:12])
stamp = runtime_root / ".nif-source-commit"
if not stamp.exists() or stamp.read_text().strip() != NIF_COMMIT:
    runtime_root.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-deps",
            "--upgrade",
            "--disable-pip-version-check",
            "--target",
            str(runtime_root),
            "git+https://github.com/pswpswpsw/nif.git@" + NIF_COMMIT,
        ]
    )
    stamp.write_text(NIF_COMMIT)


def replace_compatibility(path, old, new):
    content = path.read_text(encoding="utf-8")
    if new in content:
        return
    if old in content:
        path.write_text(content.replace(old, new), encoding="utf-8")
    else:
        raise RuntimeError("Pinned NIF source changed unexpectedly: " + str(path))


optimizer_file = runtime_root / "nif" / "optimizers" / "external_optimizers.py"
replace_compatibility(
    optimizer_file,
    "from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx",
    "from tensorflow.python.distribute import distribute_lib as distribute_ctx",
)
for optimizer_class in ("AdaBeliefOptimizer", "L4Adam"):
    replace_compatibility(
        optimizer_file,
        "class " + optimizer_class + "(tf.keras.optimizers.Optimizer):",
        "class " + optimizer_class + "(tf.keras.optimizers.legacy.Optimizer):",
    )
for name in ("lbfgs.py", "lbfgs_V2.py"):
    path = runtime_root / "nif" / "optimizers" / name
    content = path.read_text(encoding="utf-8")
    if "np.product(" in content:
        path.write_text(content.replace("np.product(", "np.prod("), encoding="utf-8")

tfr_file = runtime_root / "nif" / "data" / "tfr_dataset.py"
replace_compatibility(
    tfr_file,
    "target = tf.transpose(\n"
    "                batch_file[self.n_feature : self.n_feature + self.n_target]\n            )",
    "target = tf.transpose(\n"
    "                tf.squeeze(tf.stack("
    "batch_file[self.n_feature : self.n_feature + self.n_target], 0), 1)\n"
    "            )",
)
replace_compatibility(
    tfr_file,
    "target = tf.transpose(batch_file[-self.n_target :])",
    "target = tf.transpose(\n"
    "                tf.squeeze(tf.stack(batch_file[-self.n_target :], 0), 1)\n            )",
)

gc_file = runtime_root / "nif" / "optimizers" / "gtcf.py"
replace_compatibility(
    gc_file,
    "    def get_centralized_gradients_for_optimizer(loss, params):\n",
    """    if not getattr(optimizer, "_nif_gradient_centralization", False):
        original_get_gradients = optimizer._get_gradients

        def centralized_tape_gradients(*args, **kwargs):
            gradients_and_variables = original_get_gradients(*args, **kwargs)
            return [
                (gradient - tf.reduce_mean(
                    gradient, axis=list(range(len(gradient.shape) - 1)), keepdims=True
                )
                if gradient is not None and len(gradient.shape) > 1 else gradient, variable)
                for gradient, variable in gradients_and_variables
            ]

        optimizer._get_gradients = centralized_tape_gradients
        optimizer._nif_gradient_centralization = True

    def get_centralized_gradients_for_optimizer(loss, params):
""",
)

if str(runtime_root) not in sys.path:
    sys.path.insert(0, str(runtime_root))
if "nif" in sys.modules:
    loaded_nif = Path(sys.modules["nif"].__file__).resolve()
    if not loaded_nif.is_relative_to(runtime_root.resolve()):
        raise RuntimeError(
            "Restart the session: an unrelated NIF installation is already imported."
        )
print("Python:", sys.version.split()[0])
for requirement in COLAB_REQUIREMENTS:
    package = requirement.split("==")[0]
    print(package + ":", importlib.metadata.version(package))
print("NIF source commit:", NIF_COMMIT)
print("Compatibility copy:", runtime_root / "nif")
