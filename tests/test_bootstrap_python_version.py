"""Keep the bootstrap's Python guard consistent with the pinned SciPy wheel."""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


class BootstrapPythonVersion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (ROOT / "tutorial/colab_support/bootstrap_tutorials.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        guards = [
            node
            for node in tree.body
            if isinstance(node, ast.If)
            and any(
                isinstance(item, ast.Attribute)
                and isinstance(item.value, ast.Name)
                and item.value.id == "sys"
                and item.attr == "version_info"
                for item in ast.walk(node.test)
            )
        ]
        if len(guards) != 1:
            raise AssertionError("Expected one explicit Python version guard")
        cls.guard = compile(
            ast.Module(body=guards, type_ignores=[]), "bootstrap-version-guard", "exec"
        )

    def test_python_310_is_rejected_before_dependency_installation(self):
        with self.assertRaisesRegex(RuntimeError, "3.11.*3.13"):
            exec(self.guard, {"sys": SimpleNamespace(version_info=(3, 10, 14))})

    def test_supported_colab_python_versions_are_accepted(self):
        for minor in (11, 12, 13):
            with self.subTest(minor=minor):
                exec(self.guard, {"sys": SimpleNamespace(version_info=(3, minor, 0))})

    def test_unsupported_versions_are_rejected(self):
        for version in ((3, 9, 0), (3, 14, 0)):
            with self.subTest(version=version), self.assertRaises(RuntimeError):
                exec(self.guard, {"sys": SimpleNamespace(version_info=version)})


if __name__ == "__main__":
    unittest.main(verbosity=2)
