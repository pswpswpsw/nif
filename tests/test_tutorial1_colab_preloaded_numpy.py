"""Fresh Colab preloads NumPy; installing unrelated extras must remain Run-all safe."""
import ast
import importlib.metadata
import os
import re
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


BOOTSTRAP = (
    Path(__file__).resolve().parents[1]
    / "tutorial"
    / "colab_support"
    / "bootstrap_tutorial1.py"
)


class PreloadedColabNumpy(unittest.TestCase):
    def run_install_prefix(self, numpy_version):
        source = BOOTSTRAP.read_text(encoding="utf-8").split("\nruntime_root =", 1)[0]
        requirements = ast.literal_eval(
            re.search(r"COLAB_REQUIREMENTS\s*=\s*(\[[\s\S]*?\])", source).group(1)
        )
        versions = dict(item.split("==") for item in requirements)
        versions["numpy"] = numpy_version

        def version(name):
            if name == "tensorflow-model-optimization":
                raise importlib.metadata.PackageNotFoundError(name)
            return versions[name]

        calls = []
        with mock.patch.dict(
            sys.modules, {"numpy": types.ModuleType("numpy")}
        ), mock.patch.dict(os.environ, {"TF_USE_LEGACY_KERAS": "1"}), mock.patch(
            "importlib.metadata.version", side_effect=version
        ), mock.patch(
            "subprocess.check_call", side_effect=lambda args: calls.append(args)
        ):
            exec(compile(source, str(BOOTSTRAP), "exec"), {})
        return calls

    def test_matching_preloaded_numpy_allows_installing_nif_extra(self):
        calls = self.run_install_prefix("2.1.3")
        self.assertEqual(len(calls), 1)

    def test_changing_preloaded_numpy_requires_restart(self):
        with self.assertRaisesRegex(RuntimeError, "Restart"):
            self.run_install_prefix("2.2.6")


if __name__ == "__main__":
    unittest.main(verbosity=2)
