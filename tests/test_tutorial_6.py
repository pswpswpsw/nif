"""Tutorial 6 dependency and actual regularized-model regression checks."""
import ast
import json
import os
import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tutorial" / "6_revisit_multi_scale_NIF_with_L1_L2_regularization.ipynb"
COPY = SOURCE.with_name(SOURCE.stem + "_colab.ipynb")


def load_notebook():
    return json.loads(
        Path(
            os.environ.get("TUTORIAL6_NOTEBOOK", COPY if COPY.exists() else SOURCE)
        ).read_text(encoding="utf-8")
    )


class TutorialSix(unittest.TestCase):
    def test_source_and_comments_preserved(self):
        original = json.loads(SOURCE.read_text(encoding="utf-8"))
        notebook = load_notebook()
        self.assertEqual(len(notebook["cells"]), len(original["cells"]))
        for index, (old, new) in enumerate(zip(original["cells"], notebook["cells"])):
            self.assertEqual(old["cell_type"], new["cell_type"])
            if index != 0:
                self.assertEqual(old["source"], new["source"], f"cell {index} changed")
        self.assertEqual(
            "".join(notebook["cells"][0]["source"]),
            (ROOT / "tutorial" / "colab_support" / "bootstrap_tutorials.py").read_text(
                encoding="utf-8"
            ),
        )

    def test_colab_dependencies_resolve(self):
        notebook = load_notebook()
        code = "\n".join(
            "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
        )
        match = re.search(r"COLAB_REQUIREMENTS\s*=\s*(\[[\s\S]*?\])", code)
        if match:
            requirements = ast.literal_eval(match.group(1))
        else:
            install = re.search(r"[!%]pip install\s+(git\+https://\S+)", code)
            self.assertIsNotNone(install)
            requirements = [install.group(1)]
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
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-12000:])

    def test_every_regularization_case_affects_loss_and_gradients(self):
        os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
        os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        bootstrap = ROOT / "tutorial" / "colab_support" / "bootstrap_tutorials.py"
        exec(compile(bootstrap.read_text(encoding="utf-8"), str(bootstrap), "exec"), {})
        import numpy as np
        import tensorflow as tf
        import nif
        from nif.demo import TravelingWaveHighFreq

        notebook = load_notebook()
        shape_config = None
        parameter_configs = []
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            try:
                tree = ast.parse("".join(cell["source"]))
            except SyntaxError:
                continue
            for node in tree.body:
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                ):
                    if node.targets[0].id == "cfg_shape_net":
                        shape_config = ast.literal_eval(node.value)
                    elif node.targets[0].id == "cfg_parameter_net":
                        parameter_configs.append(ast.literal_eval(node.value))
        self.assertEqual(len(parameter_configs), 9)
        data = TravelingWaveHighFreq().data
        x = tf.convert_to_tensor(data[:16, :2], dtype=tf.float32)
        for config in parameter_configs:
            with self.subTest(config=config):
                tf.keras.backend.clear_session()
                model = nif.NIFMultiScale(shape_config, config, "float32").build()
                with tf.GradientTape(persistent=True) as tape:
                    predictions = model(x)
                    mse = tf.reduce_mean(
                        tf.square(
                            predictions - tf.cast(data[:16, 2:3], predictions.dtype)
                        )
                    )
                    reg = tf.add_n(model.losses) if model.losses else tf.constant(0.0)
                    total = mse + reg
                if "l1_reg" not in config and "l2_reg" not in config:
                    self.assertEqual(float(reg), 0.0)
                    continue
                strength = config.get("l1_reg", config.get("l2_reg"))
                expected = (
                    sum(
                        float(
                            tf.reduce_sum(
                                tf.abs(w) if "l1_reg" in config else tf.square(w)
                            )
                        )
                        for w in model.trainable_weights
                    )
                    * strength
                )
                self.assertGreater(float(reg), 0.0)
                np.testing.assert_allclose(float(reg), expected, rtol=2e-6, atol=1e-8)
                total_grad = tape.gradient(total, model.trainable_weights)
                mse_grad = tape.gradient(mse, model.trainable_weights)
                reg_grad = tape.gradient(reg, model.trainable_weights)
                self.assertTrue(all(g is not None for g in reg_grad))
                self.assertTrue(any(np.any(np.abs(g.numpy()) > 0) for g in reg_grad))
                for weight, g_total, g_mse, g_reg in zip(
                    model.trainable_weights, total_grad, mse_grad, reg_grad
                ):
                    self.assertTrue(np.isfinite(g_total.numpy()).all())
                    analytic = strength * (
                        tf.sign(weight) if "l1_reg" in config else 2 * weight
                    )
                    np.testing.assert_allclose(
                        g_reg.numpy(), analytic.numpy(), rtol=1e-5, atol=1e-9
                    )
                    np.testing.assert_allclose(
                        g_total.numpy() - g_mse.numpy(),
                        g_reg.numpy(),
                        rtol=1e-3,
                        atol=2e-6,
                    )
                model.compile(tf.keras.optimizers.Adam(1e-3), loss="mse")
                actual = model.test_on_batch(x, data[:16, 2:3])
                np.testing.assert_allclose(actual, float(total), rtol=2e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
