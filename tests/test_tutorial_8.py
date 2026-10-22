"""Regression checks for tutorial 8's unchanged Sobolev experiment."""
import ast
import hashlib
import io
import json
import os
import tokenize
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("MPLBACKEND", "Agg")


ROOT = Path(__file__).resolve().parents[1]
STEM = "8_revisit_multi_scale_NIF_with_sobolov_training"
ORIGINAL = ROOT / "tutorial" / (STEM + ".ipynb")
COPY = ROOT / "tutorial" / (STEM + "_colab.ipynb")


def notebook():
    path = (
        ORIGINAL if os.environ.get("NIF_TEST_ORIGINALS") or not COPY.exists() else COPY
    )
    return json.loads(path.read_text(encoding="utf-8"))


def source(index, nb=None):
    return "".join((nb or notebook())["cells"][index]["source"])


def normalization():
    data = np.load(ROOT / "nif" / "demo" / "dataset" / "traveling_wave_high_freq.npz")[
        "data"
    ]
    mean, scale = data.mean(axis=0), data.std(axis=0)
    mean[:2] = (data[:, :2].min(axis=0) + data[:, :2].max(axis=0)) / 2
    scale[:2] = (data[:, :2].max(axis=0) - data[:, :2].min(axis=0)) / 2
    scale[2] = np.abs(data[:, 2]).max()
    return mean, scale


class TutorialEightRegression(unittest.TestCase):
    def test_jacobian_layer_and_derivative_normalization(self):
        bootstrap = ROOT / "tutorial" / "colab_support" / "bootstrap_tutorials.py"
        exec(compile(bootstrap.read_text(encoding="utf-8"), str(bootstrap), "exec"), {})
        import tensorflow as tf
        from nif.layers import JacobianLayer

        mean, scale = normalization()
        physical = np.array(
            [[0.0, 0.198], [20.0, 0.321], [40.0, 0.445], [60.0, 0.558]],
            dtype=np.float32,
        )
        normalized = (physical - mean[:2]) / scale[:2]
        inputs = tf.keras.Input((2,))
        z = (
            inputs[:, 1:2] * scale[1]
            + mean[1]
            - 0.2
            - 0.006 * (inputs[:, :1] * scale[0] + mean[0])
        )
        physical_u = tf.exp(-1000 * z**2) * tf.sin(400 * z)
        model = tf.keras.Model(inputs, (physical_u - mean[2]) / scale[2])
        values, gradients = JacobianLayer(model, [0], [0, 1])(
            tf.constant(normalized, tf.float32)
        )
        delta = physical[:, 1] - 0.2 - 0.006 * physical[:, 0]
        dx = np.exp(-1000 * delta**2) * (
            -2000 * delta * np.sin(400 * delta) + 400 * np.cos(400 * delta)
        )
        expected = np.column_stack(
            (-0.006 * dx * scale[0] / scale[2], dx * scale[1] / scale[2])
        )
        np.testing.assert_allclose(
            gradients.numpy()[:, 0, :], expected, rtol=1e-4, atol=2e-3
        )
        self.assertTrue(np.isfinite(values.numpy()).all())
        self.assertTrue(np.isfinite(gradients.numpy()).all())

    def test_superresolution_returns_physical_wave(self):
        mean, scale = normalization()

        class Dataset:
            pass

        tw = Dataset()
        tw.mean, tw.std = mean, scale
        scope = dict(np=np, tw=tw, x0=0.2, c=0.12 / 20, omega=400)
        exec(source(35), scope)

        class ExactNormalizedWave:
            def predict(self, inputs):
                physical = inputs * scale[:2] + mean[:2]
                z = physical[:, 1] - 0.2 - 0.12 / 20 * physical[:, 0]
                u = np.exp(-1000 * z**2) * np.sin(400 * z)
                return np.column_stack(
                    ((u - mean[2]) / scale[2], np.zeros((len(u), 2)))
                )

        scope["model_opt_ng"] = scope["model_opt"] = ExactNormalizedWave()
        exec(source(36), scope)
        for name in ("u_ng_test", "u_wg_test"):
            self.assertTrue(np.isfinite(scope[name]).all())
            np.testing.assert_allclose(
                scope[name], scope["u_test"], atol=1e-12, rtol=1e-12
            )

    def test_analytic_derivatives(self):
        from matplotlib import pyplot as plt

        scope = dict(np=np, plt=plt)
        exec(source(4), scope)
        exec(source(6), scope)
        z = scope["xx"] - scope["x0"] - scope["c"] * scope["tt"]

        def wave(v):
            return np.exp(-1000 * v**2) * np.sin(scope["omega"] * v)

        step = 1e-7
        fd_dx = (wave(z + step) - wave(z - step)) / (2 * step)
        np.testing.assert_allclose(scope["dudx"], fd_dx, atol=2e-6, rtol=2e-7)
        np.testing.assert_allclose(
            scope["dudt"], -scope["c"] * scope["dudx"], atol=1e-12
        )
        plt.close("all")

    def test_original_markdown_comments_and_training_settings(self):
        original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
        current = notebook()
        self.assertEqual(len(original["cells"]), len(current["cells"]))
        for index, (before, after) in enumerate(
            zip(original["cells"], current["cells"])
        ):
            self.assertEqual(before["cell_type"], after["cell_type"])
            if before["cell_type"] == "markdown":
                self.assertEqual(before["source"], after["source"], index)
            elif index:

                def comments(text):
                    return [
                        tok.string
                        for tok in tokenize.generate_tokens(io.StringIO(text).readline)
                        if tok.type == tokenize.COMMENT
                    ]

                self.assertEqual(
                    comments("".join(before["source"])),
                    comments("".join(after["source"])),
                    index,
                )
        self.assertIn("nepoch = 50000", source(8, current))
        self.assertIn("epochs=nepoch", source(20, current))
        self.assertIn("epochs=nepoch", source(22, current))
        self.assertIn(
            "sd_grad =  tf.square(y_true[:, n_output+1:] - y_pred[:, n_output+1:])",
            source(20, current),
        )
        if COPY.exists() and not os.environ.get("NIF_TEST_ORIGINALS"):
            self.assertEqual(
                current["metadata"]["nif_colab_compatibility"]["original_sha256"],
                hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                source(0, current),
                (
                    ROOT / "tutorial" / "colab_support" / "bootstrap_tutorials.py"
                ).read_text(encoding="utf-8"),
            )

    def test_sobolev_loss_supervises_only_spatial_derivative(self):
        import tensorflow as tf

        scope = dict(tf=tf, n_output=1, coef_grad=1e-3)
        definitions = [
            node
            for node in ast.parse(source(20)).body[1].body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        ]
        exec(
            compile(
                ast.Module(body=definitions, type_ignores=[]), "tutorial8-loss", "exec"
            ),
            scope,
        )
        truth = tf.zeros((2, 3))
        prediction = tf.constant([[1.0, 10.0, 3.0], [1.0, 10.0, 3.0]])
        np.testing.assert_allclose(
            scope["Sobolov_MSE"]()(truth, prediction).numpy(), 1.009, rtol=1e-6
        )
        np.testing.assert_allclose(
            scope["Sobolov_MSE_dudt"](truth, prediction).numpy(), [0.1, 0.1], rtol=1e-6
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
