"""Each released notebook carries only the helpers its workload uses."""
import ast
import copy
import json
from pathlib import Path

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]
COMMON = {
    "epochs",
    "tensor",
    "loader",
    "predict",
    "centralize_gradients",
    "save_checkpoint",
    "save_and_check",
    "train",
    "write_metrics",
}
EXPECTED = {
    1: COMMON | {"wave_figure"},
    2: COMMON | {"wave_figure"},
    3: COMMON,
    4: {"dense_tanh", "tensor", "save_checkpoint", "save_and_check", "write_metrics"},
    5: {"dense_tanh", "tensor", "save_checkpoint", "save_and_check", "write_metrics"},
    6: COMMON,
    7: COMMON - {"save_and_check"} | {"save_inference_and_check"},
    8: COMMON,
}


def notebook(number):
    path = next((ROOT / "tutorial").glob(f"{number}_*_pytorch.ipynb"))
    return json.loads(path.read_text(encoding="utf-8"))


def functions(source):
    return {
        node.name: ast.get_source_segment(source, node)
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    }


@pytest.mark.parametrize("number", range(1, 9))
def test_helper_payload_matches_the_workload(number):
    source = "".join(notebook(number)["cells"][2]["source"])
    assert set(functions(source)) == EXPECTED[number]


def test_shared_helpers_remain_verbatim_generator_definitions():
    source = (ROOT / "dev_tests/generate_native_notebooks.py").read_text(
        encoding="utf-8"
    )
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"HELPERS", "T7_INFERENCE_HELPERS"}
    }
    canonical = functions(constants["HELPERS"])
    canonical.update(functions(constants["T7_INFERENCE_HELPERS"]))
    for number in range(1, 9):
        actual = functions("".join(notebook(number)["cells"][2]["source"]))
        for name, definition in actual.items():
            if number == 7 and name in {"train", "save_checkpoint"}:
                # T7 has its original pruning hook and non-tensor state handling.
                continue
            assert definition == canonical[name], (number, name)


def test_helper_selection_preserves_callback_and_default_dependencies():
    generator = ast.parse(
        (ROOT / "dev_tests/generate_native_notebooks.py").read_text(encoding="utf-8")
    )
    selector = next(
        node
        for node in generator.body
        if isinstance(node, ast.FunctionDef) and node.name == "select_helpers"
    )
    scope = {"ast": ast}
    exec(
        compile(ast.Module(body=[selector], type_ignores=[]), "selector", "exec"), scope
    )
    source = (
        "def prepare():\n    return 7\n\n"
        "def unused():\n    return -1\n\n"
        "def run(callback=prepare):\n    return callback()\n"
    )
    selected = scope["select_helpers"](source, ["callbacks = [run]"])
    assert list(functions(selected)) == ["prepare", "run"]
    namespace = {}
    exec(selected, namespace)
    assert namespace["run"]() == 7
    for name, definition in functions(selected).items():
        assert definition == functions(source)[name]


@pytest.mark.parametrize("fresh", [False, True])
def test_generator_reproduces_current_teaching_and_preserves_existing_cells(
    fresh, tmp_path, monkeypatch
):
    generator = ROOT / "dev_tests/generate_native_notebooks.py"
    source = generator.read_text(encoding="utf-8")
    generator_root = ROOT
    if fresh:
        generator_root = tmp_path
        destination = tmp_path / "tutorial"
        destination.mkdir()
        for number in range(1, 9):
            path = next(
                path
                for path in (ROOT / "tutorial").glob(f"{number}_*.ipynb")
                if not path.stem.endswith(("_colab", "_pytorch"))
                and "self_scaled_comparison" not in path.stem
            )
            reference = nbformat.read(path, as_version=4)
            for cell in reference.cells:
                if cell.cell_type == "code":
                    cell.outputs = []
                    cell.execution_count = None
            nbformat.write(reference, destination / path.name)
    generated = {}
    monkeypatch.setattr(
        nbformat,
        "write",
        lambda value, path: generated.update({Path(path).name: copy.deepcopy(value)}),
    )
    exec(
        compile(source, str(generator), "exec"),
        {"__file__": str(generator_root / "dev_tests/generate_native_notebooks.py")},
    )
    assert len(generated) == 8
    for number in range(1, 9):
        path = next((ROOT / "tutorial").glob(f"{number}_*_pytorch.ipynb"))
        expected = nbformat.read(path, as_version=4)
        actual = generated[path.name]
        if fresh:
            assert [(cell.cell_type, cell.source) for cell in actual.cells] == [
                (cell.cell_type, cell.source) for cell in expected.cells
            ]
        else:
            assert actual == expected


@pytest.mark.parametrize("number", [4, 5, 7])
def test_checkpoint_helpers_keep_their_transitive_dependencies(number, tmp_path):
    import torch

    scope = {
        "torch": torch,
        "nn": torch.nn,
        "device": torch.device("cpu"),
        "copy": copy,
        "SMOKE": True,
        "OUT": tmp_path,
    }
    exec("".join(notebook(number)["cells"][2]["source"]), scope)

    def factory():
        return torch.nn.Linear(2, 1)

    model = factory()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.tensor([[0.25, -0.75], [0.125, 0.5]])
    if number == 7:
        restored = scope["save_inference_and_check"](
            model, "roundtrip", x, factory=factory
        )
    else:
        restored = scope["save_and_check"](
            model, optimizer, "roundtrip", x, factory=factory
        )
    torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)
