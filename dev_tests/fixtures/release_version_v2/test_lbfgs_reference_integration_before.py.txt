import ast
import json
from pathlib import Path

import pytest

from dev_tests.lint_preservation import read_lint_bytes
from dev_tests.lint_preservation import read_lint_text

ROOT = Path(__file__).resolve().parents[1]


def test_actual_tutorial_constructor_is_explicit_reference_lbfgs():
    notebook = json.loads(
        (ROOT / "tutorial/1_simple_1d_wave_pytorch.ipynb").read_bytes()
    )
    code = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    imports = [
        node
        for source in code
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module == "nif.optimizers"
    ]
    assert any(
        any(alias.name == "ReferenceLBFGS" for alias in node.names) for node in imports
    )
    constructors = [
        node
        for source in code
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "fine_tuner" for t in node.targets)
    ]
    assert len(constructors) == 1
    call = constructors[0].value
    assert isinstance(call.func, ast.Name) and call.func.id == "ReferenceLBFGS"
    teaching = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    assert f"- then we use native `{call.func.id}` to wrap around our model" in teaching
    generator = read_lint_text(ROOT, "dev_tests/generate_native_notebooks.py")
    assert f"'`TFPLBFGS`':'native `{call.func.id}`'" in generator
    assert {arg.arg for arg in call.keywords} == {
        "max_iter",
        "num_correction_pairs",
        "tolerance",
        "x_tolerance",
        "f_relative_tolerance",
        "f_absolute_tolerance",
        "max_line_search_iterations",
    }


def test_exact_reference_scope_successor_exists():
    import importlib.util

    path = ROOT / "dev_tests/lbfgs_reference_scope.py"
    assert path.is_file(), "bounded ReferenceLBFGS public scope successor is required"
    spec = importlib.util.spec_from_file_location("reference_scope_red", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.assert_reference_lbfgs_scope(ROOT)


def test_original_strong_wolfe_regression_replays_all_assertions(tmp_path):
    import runpy

    archive = ROOT / "dev_tests/fixtures/lbfgs_reference_v1"
    for name in (
        "tutorial/1_simple_1d_wave_pytorch.ipynb",
        "dev_tests/generate_native_notebooks.py",
    ):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((archive / (name.replace("/", "__") + ".txt")).read_bytes())
    old = runpy.run_path(str(archive / "dev_tests__test_tutorial1_lbfgs.py.txt"))
    regression = old["test_tutorial1_preserves_public_lbfgs_controls"]
    regression.__globals__["ROOT"] = tmp_path
    for source in ("notebook", "generator"):
        for smoke in (False, True):
            regression(source, smoke)


def test_generator_and_notebook_optimizer_cells_are_identical():
    from dev_tests.test_tutorial1_lbfgs import solver_cells

    assert solver_cells("notebook") == solver_cells("generator")


def test_frozen_stopping_and_phase0_inputs_are_unchanged():
    import hashlib

    assert hashlib.sha256(
        read_lint_bytes(ROOT, "dev_tests/test_lbfgs_stopping_contract.py")
    ).hexdigest() == (
        "16a203349aa115fef48785bd8af47f8e0a93962df9d4abf0d9c1c6ac74b8424c"
    )
    from dev_tests.test_lbfgs_reference_phase0_records import (
        test_frozen_public_inputs_and_pending_recipe_are_byte_exact,
    )

    test_frozen_public_inputs_and_pending_recipe_are_byte_exact()


@pytest.fixture(scope="module")
def reference_projection(tmp_path_factory):
    from dev_tests.publication_projection import project_public_files

    return project_public_files(ROOT, tmp_path_factory.mktemp("reference_public"))


def test_reference_scope_runs_in_public_projection(reference_projection):
    from dev_tests.lbfgs_reference_scope import assert_reference_lbfgs_scope

    assert not (reference_projection / ".git").exists()
    assert not (reference_projection / "dev_tests/artifacts").exists()
    assert assert_reference_lbfgs_scope(reference_projection)


@pytest.mark.parametrize(
    "mutation", ["code", "output", "metadata", "cell-metadata", "reorder"]
)
def test_reference_scope_rejects_other_notebook_changes(reference_projection, mutation):
    from dev_tests.lbfgs_reference_scope import assert_reference_lbfgs_scope

    path = reference_projection / "tutorial/1_simple_1d_wave_pytorch.ipynb"
    original = path.read_bytes()
    notebook = json.loads(original)
    if mutation == "code":
        next(c for c in notebook["cells"] if c["cell_type"] == "code")["source"].append(
            "# unauthorized\n"
        )
    elif mutation == "output":
        next(c for c in notebook["cells"] if c["cell_type"] == "code")[
            "outputs"
        ].append(
            {"output_type": "stream", "name": "stdout", "text": ["unauthorized\n"]}
        )
    elif mutation == "metadata":
        notebook["metadata"]["unauthorized"] = True
    elif mutation == "cell-metadata":
        notebook["cells"][0]["metadata"]["unauthorized"] = True
    else:
        notebook["cells"][0], notebook["cells"][1] = (
            notebook["cells"][1],
            notebook["cells"][0],
        )
    try:
        path.write_text(json.dumps(notebook), encoding="utf-8")
        with pytest.raises(
            AssertionError, match="tutorial/1_simple_1d_wave_pytorch.ipynb"
        ):
            assert_reference_lbfgs_scope(reference_projection)
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize(
    "name",
    [
        ".gitattributes",
        "MANIFEST.in",
        "dev_tests/lbfgs_reference_scope.py",
        "nif/optimizers/TFP_LICENSE.txt",
        "nif/optimizers/TFP_NOTICE.txt",
        "nif/optimizers/lbfgs.py",
        "nif/optimizers/lbfgs_V2.py",
        "nif/optimizers/__init__.py",
        "nif/optimizers/reference_lbfgs.py",
        "nif/optimizers/_hager_zhang.py",
        "dev_tests/lbfgs_reference_v1.json",
        "dev_tests/fixtures/lbfgs_reference_v1/dev_tests__test_tutorial1_lbfgs.py.txt",
    ],
)
def test_reference_scope_rejects_source_record_and_archive_tampering(
    reference_projection, name
):
    from dev_tests.lbfgs_reference_scope import assert_reference_lbfgs_scope

    path = reference_projection / name
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n# unauthorized\n")
        with pytest.raises(AssertionError):
            assert_reference_lbfgs_scope(reference_projection)
    finally:
        path.write_bytes(original)


def test_new_required_archive_missing_fails(reference_projection):
    from dev_tests.lbfgs_reference_scope import assert_reference_lbfgs_scope

    path = (
        reference_projection
        / "dev_tests/fixtures/lbfgs_reference_v1/dev_tests__test_tutorial1_lbfgs.py.txt"
    )
    original = path.read_bytes()
    try:
        path.unlink()
        with pytest.raises(FileNotFoundError):
            assert_reference_lbfgs_scope(reference_projection)
    finally:
        path.write_bytes(original)


def test_three_exact_checkout_rules_and_two_license_includes_only():
    archive = ROOT / "dev_tests/fixtures/lbfgs_reference_v1"
    attributes = (archive / "gitattributes.txt").read_bytes()
    assert (ROOT / ".gitattributes").read_bytes() == attributes + (
        b"/MANIFEST.in text eol=lf\n"
        b"/nif/optimizers/TFP_LICENSE.txt text eol=lf\n"
        b"/nif/optimizers/TFP_NOTICE.txt text eol=lf\n"
    )
    manifest = (archive / "MANIFEST.in.txt").read_bytes()
    assert (ROOT / "MANIFEST.in").read_bytes() == manifest + (
        b"include nif/optimizers/TFP_LICENSE.txt\n"
        b"include nif/optimizers/TFP_NOTICE.txt\n"
    )


def test_original_portability_constructor_expectation_replays_under_gbk(
    monkeypatch, tmp_path
):
    import runpy

    archive = ROOT / "dev_tests/fixtures/lbfgs_reference_v1"
    generator = tmp_path / "dev_tests/generate_native_notebooks.py"
    generator.parent.mkdir(parents=True)
    generator.write_bytes(
        (archive / "dev_tests__generate_native_notebooks.py.txt").read_bytes()
    )
    original_solver = runpy.run_path(
        str(archive / "dev_tests__test_tutorial1_lbfgs.py.txt")
    )["solver_cells"]
    original_solver.__globals__["ROOT"] = tmp_path
    regression = runpy.run_path(
        str(archive / "dev_tests__test_test_portability.py.txt")
    )["test_generator_source_read_uses_utf8_under_gbk_default"]
    regression.__globals__["solver_cells"] = original_solver
    regression(monkeypatch)
