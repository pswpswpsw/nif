"""Public analytic reference contract for T1's actual full-mode constructor.

The chosen scalar quadratics have a finite downhill step while their gradients
exceed the shared 1e-15 threshold. The original TFP oracle makes progress on both.
This requires progress, not matching solver trajectories or parameter bits.
"""
import ast
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorial/1_simple_1d_wave_pytorch.ipynb"


def tutorial1_optimizer(parameter):
    """Execute only the public settings and optimizer constructor, no model cell."""
    notebook = json.loads(NOTEBOOK.read_bytes())
    code = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    setup = next(source for source in code if "rounds, max_iter =" in source)

    # Supply the tiny parameter through the same parameters() interface.
    class Point:
        def parameters(self):
            return iter([parameter])

    scope = {"torch": torch, "SMOKE": False, "new_model": Point()}
    exec(compile(setup, str(NOTEBOOK), "exec"), scope)
    assignments = [
        node
        for source in code
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "fine_tuner"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    constructor = assignments[0]
    # If the public constructor names an imported optimizer, resolve only its
    # own binding. No notebook data, NIF model or training cell is executed.
    symbol = constructor.value.func
    while isinstance(symbol, ast.Attribute):
        symbol = symbol.value
    if isinstance(symbol, ast.Name) and symbol.id not in scope:
        for source in code:
            for node in ast.parse(source).body:
                if isinstance(node, ast.ImportFrom):
                    aliases = [
                        alias
                        for alias in node.names
                        if (alias.asname or alias.name) == symbol.id
                    ]
                    if aliases:
                        statement = ast.ImportFrom(
                            module=node.module, names=aliases, level=node.level
                        )
                        tree = ast.fix_missing_locations(
                            ast.Module(body=[statement], type_ignores=[])
                        )
                        exec(compile(tree, str(NOTEBOOK), "exec"), scope)
    exec(
        compile(ast.Module(body=[constructor], type_ignores=[]), str(NOTEBOOK), "exec"),
        scope,
    )
    return scope["fine_tuner"], scope["tolerance"]


@pytest.mark.parametrize("scale", [1.0, 1e-8], ids=["unit-scale", "small-scale"])
def test_full_t1_constructor_makes_progress_before_stopping_on_nonstationary_quadratic(
    scale,
):
    point = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64, device="cpu"))
    optimizer, gradient_tolerance = tutorial1_optimizer(point)
    evaluations = []

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = 0.5 * scale * point.square().sum()
        loss.backward()
        evaluations.append(float(loss.detach()))
        return loss

    initial_loss = 0.5 * scale
    assert scale > gradient_tolerance
    optimizer.step(closure)
    final_loss = float((0.5 * scale * point.detach().square().sum()).item())
    diagnostic = {
        "scale": scale,
        "initial_x": 1.0,
        "final_x": float(point.detach().item()),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "initial_gradient": scale,
        "final_gradient": scale * float(point.detach().item()),
        "gradient_tolerance": gradient_tolerance,
        "evaluations": len(evaluations),
    }
    print(json.dumps(diagnostic, sort_keys=True))
    assert final_loss < initial_loss, diagnostic
