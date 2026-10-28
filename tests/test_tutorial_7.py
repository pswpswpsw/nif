"""Short CPU regression for the original NIF pruning and TFLite pipeline."""
import contextlib
import hashlib
import io
import json
import logging
import os
import tempfile
import tokenize
import unittest
from pathlib import Path

os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "tutorial" / "7_model_pruning_and_quantization.ipynb"
COPY = ORIGINAL.with_name(ORIGINAL.stem + "_colab.ipynb")


def comments(source):
    source = "\n".join(
        line for line in source.splitlines() if not line.startswith(("!", "%"))
    )
    return [
        token.string
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT
    ]


class Tutorial7(unittest.TestCase):
    def test_original_content_and_training_preserved(self):
        if not COPY.exists():
            self.skipTest("Compatibility copy has not been built yet")
        original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
        repaired = json.loads(COPY.read_text(encoding="utf-8"))
        self.assertEqual(len(original["cells"]), len(repaired["cells"]))
        for index, (before, after) in enumerate(
            zip(original["cells"], repaired["cells"])
        ):
            self.assertEqual(before["cell_type"], after["cell_type"])
            if before["cell_type"] == "markdown":
                self.assertEqual(before["source"], after["source"], index)
            else:
                self.assertEqual(
                    comments("".join(before["source"])),
                    comments("".join(after["source"])),
                    index,
                )
                if index != 0:
                    self.assertEqual(before["source"], after["source"], index)
        self.assertEqual(
            "".join(repaired["cells"][0]["source"]),
            (ROOT / "tutorial" / "colab_support" / "bootstrap_tutorials.py").read_text(
                encoding="utf-8"
            ),
        )
        self.assertIn("nepoch = 5000", "".join(repaired["cells"][4]["source"]))
        self.assertIn("nepoch=5000*10", "".join(repaired["cells"][14]["source"]))
        self.assertEqual(
            hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
            repaired["metadata"]["nif_colab_compatibility"]["original_sha256"],
        )

    def test_relative_mse_preserves_reference_spectral_norm(self):
        import numpy as np

        selected = Path(
            os.environ.get(
                "TUTORIAL7_NOTEBOOK", str(COPY if COPY.exists() else ORIGINAL)
            )
        )
        notebook = json.loads(selected.read_text(encoding="utf-8"))
        baseline = np.array([[1.0, 2.0], [3.0, 5.0]])
        pruned = np.array([[1.0, 0.0], [0.0, 5.0]])
        quantized = np.array([[1.1, 2.0], [3.0, 4.8]])
        namespace = {
            "np": np,
            "y_pred_unpruned": baseline,
            "y_pred": pruned,
            "y_pred_quant": quantized,
        }
        expected_quantized = (
            np.linalg.norm(quantized - baseline, ord=2) ** 2
            / np.linalg.norm(baseline, ord=2) ** 2
        )
        reported = eval("".join(notebook["cells"][44]["source"]), namespace)
        self.assertAlmostEqual(
            float(reported),
            float(expected_quantized),
            places=12,
            msg="Preserve the original tutorial's squared matrix spectral-norm ratio",
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec("".join(notebook["cells"][47]["source"]), namespace)
        rows = output.getvalue().splitlines()
        expected_pruned = (
            np.linalg.norm(pruned - baseline, ord=2) ** 2
            / np.linalg.norm(baseline, ord=2) ** 2
        )
        self.assertAlmostEqual(
            float(rows[0].split(" is ")[1]), float(expected_pruned), places=12
        )
        self.assertAlmostEqual(
            float(rows[1].split(" is ")[1]), float(expected_quantized), places=12
        )

    def test_pruning_stripping_and_quantized_tflite_inference(self):
        selected = Path(
            os.environ.get(
                "TUTORIAL7_NOTEBOOK", str(COPY if COPY.exists() else ORIGINAL)
            )
        )
        notebook = json.loads(selected.read_text(encoding="utf-8"))
        namespace = {"__name__": "__main__"}
        bootstrap = (
            "".join(notebook["cells"][0]["source"])
            if selected != ORIGINAL
            else (
                ROOT / "tutorial" / "colab_support" / "bootstrap_tutorial1.py"
            ).read_text(encoding="utf-8")
        )
        exec(bootstrap, namespace)
        previous = Path.cwd()
        with tempfile.TemporaryDirectory(prefix="nif-tutorial7-") as directory:
            os.chdir(directory)
            try:
                for index in [
                    1,
                    4,
                    6,
                    8,
                    10,
                    12,
                    14,
                    18,
                    20,
                    22,
                    24,
                    26,
                    30,
                    32,
                    34,
                    36,
                    38,
                    40,
                    41,
                    42,
                ]:
                    source = "".join(notebook["cells"][index]["source"])
                    if index == 4:
                        source = source.replace("nepoch = 5000", "nepoch = 2")
                    if index == 8:
                        source = source.replace(
                            "train_data = raw_data", "train_data = raw_data[:64]"
                        )
                    if index in (10, 14):
                        source = source.replace("ckpt-4999/ckpt", "ckpt-1/ckpt")
                    if index == 14:
                        source = source.replace("nepoch=5000*10", "nepoch=2").replace(
                            "prune_epoch=100*10", "prune_epoch=1"
                        )
                    print(f"Tutorial 7 regression cell {index}", flush=True)
                    exec(
                        compile(source, f"{selected.name} cell {index}", "exec"),
                        namespace,
                    )
                import numpy as np
                import tensorflow as tf

                wrappers = [
                    layer
                    for layer in namespace["model_for_pruning"].layers
                    if hasattr(layer, "pruning_vars")
                ]
                self.assertEqual(len(wrappers), 1)
                weight, mask, threshold = wrappers[0].pruning_vars[0]
                sparsity = np.mean(mask.numpy() == 0)
                self.assertGreaterEqual(sparsity, 0.89)
                self.assertLessEqual(sparsity, 0.91)
                self.assertFalse(
                    any(
                        hasattr(layer, "pruning_vars")
                        for layer in namespace["model_for_export"].layers
                    )
                )
                features = namespace["input_feature"].astype(np.float32)
                pruned = namespace["model_for_pruning"](
                    features, training=False
                ).numpy()
                stripped = namespace["model_for_export"](
                    features, training=False
                ).numpy()
                np.testing.assert_allclose(pruned, stripped, rtol=1e-5, atol=1e-6)
                for key in ("pruned_tflite_model", "quantized_and_pruned_tflite_model"):
                    interpreter = tf.lite.Interpreter(model_content=namespace[key])
                    interpreter.allocate_tensors()
                    inputs = interpreter.get_input_details()[0]
                    outputs = interpreter.get_output_details()[0]
                    interpreter.set_tensor(inputs["index"], features[:1])
                    interpreter.invoke()
                    prediction = interpreter.get_tensor(outputs["index"])
                    self.assertEqual(prediction.shape, (1, 1))
                    self.assertTrue(np.isfinite(prediction).all())
                    if key == "pruned_tflite_model":
                        np.testing.assert_allclose(
                            prediction, stripped[:1], rtol=1e-4, atol=1e-5
                        )
                    if key.startswith("quantized"):
                        quantized = [
                            item
                            for item in interpreter.get_tensor_details()
                            if item["dtype"] == np.int8
                            and len(item["quantization_parameters"]["scales"]) > 0
                        ]
                        self.assertTrue(
                            quantized,
                            "TFLite model must contain genuinely quantized weights",
                        )
                self.assertEqual(namespace["y_pred_quant"].shape, (200, 10))
                self.assertTrue(np.isfinite(namespace["y_pred_quant"]).all())
                print(
                    json.dumps(
                        {
                            "mask_sparsity": float(sparsity),
                            "quantized_tensors": len(quantized),
                            "float_bytes": len(namespace["pruned_tflite_model"]),
                            "quantized_bytes": len(
                                namespace["quantized_and_pruned_tflite_model"]
                            ),
                        }
                    )
                )
            finally:
                logging.shutdown()
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main(verbosity=2)
