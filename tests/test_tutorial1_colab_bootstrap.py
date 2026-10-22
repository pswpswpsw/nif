"""Resolve tutorial 1's real installation requirements for a fresh Colab Python.

Set TUTORIAL1_NOTEBOOK to compare the preserved original with the repaired copy.
This integration test uses uv's resolver; it does not install into the host Python.
"""
import ast
import json
import os
import re
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TutorialOneColabBootstrap(unittest.TestCase):
    def test_bootstrap_has_installable_colab_dependencies(self):
        default = ROOT / "tutorial" / "1_simple_1d_wave_colab.ipynb"
        if not default.exists():
            default = ROOT / "tutorial" / "1_simple_1d_wave.ipynb"
        notebook_path = Path(os.environ.get("TUTORIAL1_NOTEBOOK", default))
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        match = re.search(r"COLAB_REQUIREMENTS\s*=\s*(\[[\s\S]*?\])", code)
        if match:
            requirements = ast.literal_eval(match.group(1))
        else:
            install = re.search(r"[!%]pip install\s+(git\+https://\S+)", code)
            self.assertIsNotNone(
                install, "No install cell or declared Colab requirements"
            )
            requirements = [install.group(1)]
        uv = shutil.which("uv")
        self.assertIsNotNone(uv, "The resolver integration test requires uv")
        process = subprocess.run(
            [
                uv,
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
        self.assertEqual(
            process.returncode,
            0,
            f"{notebook_path.name}: installation fails on Colab Python 3.13\n"
            + process.stderr[-12000:],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
