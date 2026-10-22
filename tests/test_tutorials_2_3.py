"""Resolve the actual install cells of tutorials 2 and 3 for Colab Python 3.13."""
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEMS = ("2_multi_scale_NIF", "3_multi_scale_linear_NIF")


class TutorialTwoThreeColab(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("NIF_RUN_RUNTIME_TESTS") == "1",
        "Set NIF_RUN_RUNTIME_TESTS=1 in the matched TensorFlow environment",
    )
    def test_tutorial3_fit_actually_centralizes_gradients(self):
        helper = ROOT / "tutorial" / "colab_support" / "bootstrap_tutorials.py"
        exec(compile(helper.read_text(encoding="utf-8"), str(helper), "exec"), {})
        import numpy as np
        import tensorflow as tf
        from nif.optimizers import (
            AdaBeliefOptimizer,
            centralized_gradients_for_optimizer,
        )

        inputs = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32)
        targets = np.zeros((2, 2), dtype=np.float32)
        for eager in (True, False):
            with self.subTest(run_eagerly=eager):
                model = tf.keras.Sequential(
                    [
                        tf.keras.layers.Input((2,)),
                        tf.keras.layers.Dense(
                            2, kernel_initializer="ones", bias_initializer="zeros"
                        ),
                    ]
                )
                with tf.GradientTape() as tape:
                    loss = tf.reduce_mean(tf.square(model(inputs) - targets))
                raw = [
                    value.numpy()
                    for value in tape.gradient(loss, model.trainable_variables)
                ]
                self.assertGreater(
                    float(np.max(np.abs(raw[0].mean(axis=0)))),
                    1.0,
                    "Fixture needs non-centered raw gradients",
                )
                optimizer = AdaBeliefOptimizer(1e-3)
                optimizer.get_gradients = centralized_gradients_for_optimizer(optimizer)
                captured = [
                    tf.Variable(tf.zeros_like(variable), trainable=False)
                    for variable in model.trainable_variables
                ]
                apply_gradients = optimizer.apply_gradients

                def capture(grads_and_vars, *args, **kwargs):
                    pairs = list(grads_and_vars)
                    assigned = [
                        saved.assign(gradient)
                        for saved, (gradient, _) in zip(captured, pairs)
                    ]
                    with tf.control_dependencies(assigned):
                        return apply_gradients(pairs, *args, **kwargs)

                optimizer.apply_gradients = capture
                model.compile(optimizer, loss="mse", run_eagerly=eager)
                model.fit(
                    inputs, targets, batch_size=2, epochs=1, shuffle=False, verbose=0
                )
                self.assertEqual(int(optimizer.iterations), 1)
                self.assertEqual(len(captured), 2)
                np.testing.assert_allclose(
                    captured[0].numpy(),
                    raw[0] - raw[0].mean(axis=0, keepdims=True),
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    captured[0].numpy().mean(axis=0), 0.0, atol=1e-6
                )
                np.testing.assert_allclose(captured[1].numpy(), raw[1], atol=1e-6)
        import nif

        installed = Path(nif.__file__).resolve().parent
        before = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in installed.rglob("*.py")
        }
        exec(compile(helper.read_text(encoding="utf-8"), str(helper), "exec"), {})
        after = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in installed.rglob("*.py")
        }
        self.assertEqual(
            before,
            after,
            "Running the first notebook cell again must not repatch source",
        )

    def test_original_research_and_comments_are_preserved(self):
        for stem in STEMS:
            with self.subTest(notebook=stem):
                original_path = ROOT / "tutorial" / (stem + ".ipynb")
                original = json.loads(original_path.read_text(encoding="utf-8"))
                repaired = json.loads(
                    (ROOT / "tutorial" / (stem + "_colab.ipynb")).read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(len(original["cells"]), len(repaired["cells"]))
                self.assertEqual(
                    repaired["metadata"]["nif_colab_compatibility"]["original_sha256"],
                    hashlib.sha256(original_path.read_bytes()).hexdigest(),
                )
                for index, (before, after) in enumerate(
                    zip(original["cells"], repaired["cells"])
                ):
                    self.assertEqual(before["cell_type"], after["cell_type"])
                    if index != 0:
                        self.assertEqual(
                            before["source"],
                            after["source"],
                            f"Research source/comment changed in cell {index}",
                        )
                    if after["cell_type"] == "code":
                        self.assertIsNone(after["execution_count"])
                        self.assertEqual(after["outputs"], [])

    def test_install_cells_resolve_on_colab(self):
        for stem in STEMS:
            with self.subTest(notebook=stem):
                suffix = "" if os.environ.get("NIF_TEST_ORIGINALS") else "_colab"
                path = ROOT / "tutorial" / (stem + suffix + ".ipynb")
                if not path.exists():
                    path = ROOT / "tutorial" / (stem + ".ipynb")
                notebook = json.loads(path.read_text(encoding="utf-8"))
                source = "".join(notebook["cells"][0]["source"])
                match = re.search(r"COLAB_REQUIREMENTS\s*=\s*(\[[\s\S]*?\])", source)
                if match:
                    requirements = ast.literal_eval(match.group(1))
                else:
                    requirements = [
                        re.search(
                            r"[!%]pip install\s+(git\+https://\S+)", source
                        ).group(1)
                    ]
                result = subprocess.run(
                    [
                        shutil.which("uv"),
                        "pip",
                        "compile",
                        "--python-version",
                        "3.13",
                        "--python-platform",
                        "x86_64-unknown-linux-gnu",
                        "-",
                    ],
                    input="\n".join(requirements),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=180,
                )
                self.assertEqual(
                    result.returncode, 0, path.name + ":\n" + result.stderr[-10000:]
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
