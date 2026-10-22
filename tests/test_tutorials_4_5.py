"""Preservation and numeric regressions for tutorial 4 and 5 compatibility copies."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    4: "4_get_gradients_by_wrapping_model_with_layer",
    5: "5_large_scale_training_on_tensorflow_record_data",
}


def notebook(number, compatibility=True):
    suffix = "_colab" if compatibility else ""
    path = ROOT / "tutorial" / (NAMES[number] + suffix + ".ipynb")
    if compatibility and not path.exists():
        path = ROOT / "tutorial" / (NAMES[number] + ".ipynb")
    return json.loads(path.read_text(encoding="utf-8"))


class CopyPreservationTests(unittest.TestCase):
    def test_copies_embed_pinned_bootstrap_and_preserve_cells(self):
        bootstrap = (ROOT / "tutorial/colab_support/bootstrap_tutorials.py").read_text(
            encoding="utf-8"
        )
        for number in NAMES:
            with self.subTest(tutorial=number):
                original = notebook(number, False)
                repaired = notebook(number)
                self.assertEqual(len(original["cells"]), len(repaired["cells"]))
                self.assertEqual("".join(repaired["cells"][0]["source"]), bootstrap)
                for before, after in zip(original["cells"], repaired["cells"]):
                    self.assertEqual(before["cell_type"], after["cell_type"])
                    if before["cell_type"] == "markdown":
                        self.assertEqual(before["source"], after["source"])
                    before_comments = [
                        s
                        for s in "".join(before["source"]).splitlines()
                        if s.lstrip().startswith("#")
                    ]
                    after_comments = [
                        s
                        for s in "".join(after["source"]).splitlines()
                        if s.lstrip().startswith("#")
                    ]
                    self.assertEqual(before_comments, after_comments)

    def test_full_data_and_training_settings_preserved(self):
        original = notebook(5, False)
        repaired = notebook(5)
        for index in (3, 11, 16, 18, 20):
            self.assertEqual(
                original["cells"][index]["source"], repaired["cells"][index]["source"]
            )

    def test_tfrecord_listing_runs_as_portable_python(self):
        listing = "".join(notebook(5)["cells"][13]["source"])
        before = Path.cwd()
        with tempfile.TemporaryDirectory(prefix="nif-listing-") as tmp:
            try:
                os.chdir(tmp)
                Path("TFR_dir").mkdir()
                Path("TFR_dir/case1_0.tfrecord").write_bytes(b"test")
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    exec(listing, {})
                self.assertIn("case1_0.tfrecord", output.getvalue())
            finally:
                os.chdir(before)


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
        os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        bootstrap = ROOT / "tutorial/colab_support/bootstrap_tutorials.py"
        exec(compile(bootstrap.read_text(encoding="utf-8"), str(bootstrap), "exec"), {})
        import numpy as np
        import tensorflow as tf

        cls.np, cls.tf = np, tf

    def test_jacobian_and_hessian_match_tanh_analytic_derivatives(self):
        from nif.layers import JacobianLayer, HessianLayer

        np, tf = self.np, self.tf
        x = tf.keras.Input(
            4,
        )
        dense = tf.keras.layers.Dense(5, activation="tanh")
        model = tf.keras.Model([x], [dense(x)])
        weight = np.arange(20, dtype="float32").reshape(4, 5) / 30 - 0.3
        bias = np.arange(5, dtype="float32") / 20
        dense.set_weights([weight, bias])
        data = tf.constant(np.arange(12, dtype="float32").reshape(3, 4) / 10)
        xi, yi = [3, 1], [4, 0, 2]
        y, jac = JacobianLayer(model, yi, xi)(data)
        y2, jac2, hess = HessianLayer(model, yi, xi)(data)
        expected_y = np.tanh(data.numpy() @ weight + bias)
        w = weight[np.array(xi)][:, np.array(yi)].T
        chosen_y = expected_y[:, yi]
        expected_jac = (1 - chosen_y**2)[..., None] * w
        expected_hess = (
            (-2 * chosen_y * (1 - chosen_y**2))[..., None, None]
            * w[None, :, :, None]
            * w[None, :, None, :]
        )
        np.testing.assert_allclose(y, expected_y, rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(y2, expected_y, rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(jac, expected_jac, rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(jac2, expected_jac, rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(hess, expected_hess, rtol=3e-6, atol=1e-7)

    def test_tfrecord_roundtrip_retains_rows_and_target_shape(self):
        from nif.data.tfr_dataset import TFRDataset

        np = self.np
        source = np.arange(91, dtype="float64").reshape(13, 7) / 100
        with tempfile.TemporaryDirectory(prefix="nif-tfr-roundtrip-") as tmp:
            tmp = Path(tmp)
            np.savez(tmp / "source.npz", data=source)
            handler = TFRDataset(n_feature=4, n_target=3)
            handler.create_from_npz(
                5, str(tmp / "source.npz"), "data", str(tmp / "records"), "case"
            )
            rows = []
            batches = 0
            for record in handler.get_tfr_meta_dataset(str(tmp / "records"), epoch=2):
                batches += 1
                for features, target in handler.gen_dataset_from_batch_file(
                    record, batch_size=2
                ):
                    self.assertEqual(features.shape[1:], (4,))
                    self.assertEqual(target.shape[1:], (3,))
                    rows.append(
                        np.concatenate((features.numpy(), target.numpy()), axis=1)
                    )
            self.assertEqual(batches, 6)
            rows = np.concatenate(rows)
            self.assertEqual(rows.shape, (26, 7))
            expected = np.repeat(source, 2, axis=0)
            np.testing.assert_allclose(
                rows[np.argsort(rows[:, 0])], expected, rtol=1e-6, atol=1e-7
            )

    def test_weighted_batch_has_pointwise_target_and_weight(self):
        from nif.data.tfr_dataset import TFRDataset

        np, tf = self.np, self.tf
        source = np.arange(40, dtype="float32").reshape(5, 8) / 100
        batch_file = [tf.constant(source[:, column][None, :]) for column in range(8)]
        handler = TFRDataset(n_feature=4, n_target=3, area_weight=True)
        actual = []
        for features, target, weight in handler.gen_dataset_from_batch_file(
            batch_file, batch_size=2
        ):
            self.assertEqual(features.shape[1:], (4,))
            self.assertEqual(target.shape[1:], (3,))
            self.assertEqual(weight.shape[1:], (1,))
            actual.append(
                np.concatenate(
                    (features.numpy(), target.numpy(), weight.numpy()), axis=1
                )
            )
        actual = np.concatenate(actual)
        np.testing.assert_array_equal(actual[np.argsort(actual[:, 0])], source)


if __name__ == "__main__":
    unittest.main()
