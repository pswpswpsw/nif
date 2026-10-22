"""The execution harness must preserve empty cells without treating them as failures."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import nbformat


class NotebookRunner(unittest.TestCase):
    def test_preserved_empty_code_cell_does_not_break_full_execution(self):
        with tempfile.TemporaryDirectory(prefix="nif-runner-") as temporary:
            folder = Path(temporary)
            notebook = nbformat.v4.new_notebook(
                cells=[
                    nbformat.v4.new_code_cell(""),
                    nbformat.v4.new_code_cell("# preserved comment"),
                    nbformat.v4.new_code_cell("assert 2 + 2 == 4"),
                ]
            )
            source, output = folder / "input.ipynb", folder / "executed.ipynb"
            nbformat.write(notebook, source)
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("execute_tutorial1.py")),
                    "--notebook",
                    str(source),
                    "--output",
                    str(output),
                    "--workdir",
                    str(folder / "run"),
                ],
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            executed = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(executed["cells"]), 3)
            self.assertIsNone(executed["cells"][0]["execution_count"])
            self.assertEqual(
                [cell["execution_count"] for cell in executed["cells"][1:]], [1, 2]
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
