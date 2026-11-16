"""Current tutorial/generator integration; historical scope replay is archived."""
import ast
import json
from pathlib import Path


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
    generator = (ROOT / "dev_tests/generate_native_notebooks.py").read_text(
        encoding="utf-8"
    )
    assert any(
        isinstance(node, ast.Dict)
        and any(
            isinstance(key, ast.Constant)
            and key.value == "`TFPLBFGS`"
            and isinstance(value, ast.Constant)
            and value.value == f"native `{call.func.id}`"
            for key, value in zip(node.keys, node.values)
        )
        for node in ast.walk(ast.parse(generator))
    )
    assert {arg.arg for arg in call.keywords} == {
        "max_iter",
        "num_correction_pairs",
        "tolerance",
        "x_tolerance",
        "f_relative_tolerance",
        "f_absolute_tolerance",
        "max_line_search_iterations",
    }


def test_generator_and_notebook_optimizer_cells_are_identical():
    from dev_tests.test_tutorial1_lbfgs import solver_cells

    assert solver_cells("notebook") == solver_cells("generator")
