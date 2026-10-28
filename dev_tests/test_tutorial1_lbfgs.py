"""Public-baseline regression for Tutorial 1's actual native solver setup.

Execute the setup and constructor from both the notebook and its generator,
then check the instantiated optimizer and run a real quadratic update. No full
training data or independent evaluator fixtures are used.
"""
import ast
from pathlib import Path

import nbformat
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def solver_cells(source):
    if source == "notebook":
        notebook = nbformat.read(
            ROOT / "tutorial/1_simple_1d_wave_pytorch.ipynb", as_version=4
        )
        code = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
        return (
            next(text for text in code if "rounds, max_iter =" in text),
            next(text for text in code if "fine_tuner = ReferenceLBFGS(" in text),
        )
    tree = ast.parse(
        (ROOT / "dev_tests/generate_native_notebooks.py").read_text(encoding="utf-8")
    )
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "emit"
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == 1
    )
    replacements = {
        ast.literal_eval(key): value
        for key, value in zip(call.args[1].keys, call.args[1].values)
    }
    return ast.literal_eval(replacements[46]), ast.literal_eval(replacements[48])


@pytest.mark.parametrize("source", ["notebook", "generator"])
@pytest.mark.parametrize("smoke", [False, True])
def test_tutorial1_preserves_public_lbfgs_controls(source, smoke):
    setup, training = solver_cells(source)
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, -3.0]], dtype=torch.float64))
    scope = {"torch": torch, "SMOKE": smoke, "new_model": model}
    exec(setup, scope)
    assert (scope["rounds"], scope["max_iter"]) == ((1, 2) if smoke else (200, 1000))
    tree = ast.parse(training)
    assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "fine_tuner"
            for target in node.targets
        )
    )
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), source, "exec"), scope)
    optimizer = scope["fine_tuner"]
    options = optimizer.param_groups[0]
    from nif.optimizers import ReferenceLBFGS

    assert isinstance(optimizer, ReferenceLBFGS)
    assert options["tolerance"] == 1e-15, "Public TFPLBFGS tolerance must be retained"
    assert options["x_tolerance"] == 1e-15
    assert options["f_relative_tolerance"] == 1e-15
    assert options["f_absolute_tolerance"] == 0.0
    assert options["num_correction_pairs"] == 20, "Public num_correction_pairs is 20"
    assert options["max_iter"] == scope["max_iter"]
    assert options["max_line_search_iterations"] == 100
    assert not {"max_eval", "tolerance_change", "line_search_fn"} & options.keys()
    evaluations = []

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = model.weight.square().sum()
        loss.backward()
        evaluations.append(float(loss.detach()))
        return loss

    initial = float(model.weight.detach().square().sum())
    optimizer.step(closure)
    assert len(evaluations) >= 2
    assert torch.isfinite(model.weight).all()
    assert float(model.weight.detach().square().sum()) < initial
