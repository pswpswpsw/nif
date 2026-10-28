"""Exercise the scope test's own source reads with UTF-8 temporary copies."""
import ast
from pathlib import Path

from dev_tests import test_tutorial1_gc_scope as scope_tests
from dev_tests.lint_preservation import read_lint_bytes
from dev_tests.lint_preservation import read_lint_text


def test_scope_test_source_reads_are_utf8_under_gbk_default(monkeypatch, tmp_path):
    # Add the same harmless Unicode comment to temporary source copies. This
    # preserves the exact phase transform while testing decoding independently
    # of whether today's particular test source happens to be all ASCII.
    files = [
        "dev_tests/fixtures/tutorial1_gc_scope_v1/test_pruning_adam_state.py.txt",
        "dev_tests/test_pruning_adam_state.py",
    ]
    for name in files:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            read_lint_bytes(scope_tests.ROOT, name)
            + "\n# Preserved UTF-8 scope – reader control\n".encode("utf-8")
        )
    # This deliberately modified minimal fixture tests the original source
    # reader. Replay its hash-verified historical function with every assertion
    # intact; the current exact lint manifest must not accept arbitrary fixtures.
    source = read_lint_text(scope_tests.ROOT, "dev_tests/test_tutorial1_gc_scope.py")
    function_name = (
        "test_phase_guard_v2_changes_only_authorized_historical_scope_assertion"
    )
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    namespace = {"ROOT": tmp_path}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]), "<frozen-reader>", "exec"
        ),
        namespace,
    )
    original = Path.read_text

    def gbk_default(path, encoding=None, errors=None, **kwargs):
        return original(path, encoding=encoding or "gbk", errors=errors, **kwargs)

    monkeypatch.setattr(Path, "read_text", gbk_default)
    namespace[function_name]()
