"""Compatibility copies must not publish inherited runtime output metadata."""
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NotebookMetadataTests(unittest.TestCase):
    def test_copies_exclude_execution_metadata(self):
        notebooks = sorted((ROOT / "tutorial").glob("*_colab.ipynb"))
        self.assertEqual(len(notebooks), 8)
        for path in notebooks:
            with self.subTest(notebook=path.name):
                notebook = json.loads(path.read_text(encoding="utf-8"))
                self.assertFalse(notebook["metadata"].get("widgets"))
                self.assertFalse(
                    notebook["metadata"].get("colab", {}).get("provenance")
                )
                for index, cell in enumerate(notebook["cells"]):
                    self.assertFalse(
                        set(cell.get("metadata", {}))
                        & {"colab", "executionInfo", "outputId"},
                        f"Stored runtime output metadata in cell {index}",
                    )


if __name__ == "__main__":
    unittest.main()
