"""Check that the cloud validation payload contains the delivered cell sources."""
import ast
import base64
import hashlib
import importlib.util
import json
import tempfile
import unittest
import zlib
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


class SuitePayloadTests(unittest.TestCase):
    def test_payload_preserves_all_cells_and_full_training(self):
        builder_path = Path(__file__).with_name("build_colab_suite_validation.py")
        spec = importlib.util.spec_from_file_location(
            "colab_suite_builder", builder_path
        )
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        builder.ROOT = ROOT
        with tempfile.TemporaryDirectory() as temporary:
            builder.OUT = Path(temporary)
            builder.main()
            script = builder.OUT / "colab-suite-cell.py"
            tree = ast.parse(script.read_text(encoding="utf-8"))
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "items"
                for target in node.targets
            )
        )
        compressed = assignment.value.args[0].args[0].args[0].value
        items = json.loads(zlib.decompress(base64.b64decode(compressed)))
        self.assertEqual(len(items), 6)
        for number, item in enumerate(items, start=2):
            with self.subTest(tutorial=number):
                path = ROOT / "tutorial" / item["name"]
                self.assertTrue(path.name.startswith(str(number) + "_"))
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"]
                )
                delivered = nbformat.read(path, as_version=4)
                reconstructed = nbformat.from_dict(item["notebook"])
                nbformat.validate(reconstructed)
                self.assertEqual(
                    [(c.cell_type, c.source) for c in delivered.cells],
                    [(c.cell_type, c.source) for c in reconstructed.cells],
                )
                self.assertTrue(all(not c.get("outputs") for c in reconstructed.cells))


if __name__ == "__main__":
    unittest.main()
