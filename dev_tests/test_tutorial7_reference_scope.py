"""Exact T7 succession, portable containers, and mandatory negative controls."""
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from dev_tests.publication_projection import project_public_files
from dev_tests.tutorial7_reference_scope import assert_tutorial7_reference_scope
from dev_tests.tutorial7_reference_scope import historical_tutorial7_bytes
from dev_tests.tutorial7_reference_scope import INVENTORY
from dev_tests.tutorial7_reference_scope import RECORD
from dev_tests.tutorial7_reference_scope import SCOPE
from dev_tests.tutorial7_reference_scope import transition_record
from dev_tests.tutorial7_reference_scope import TRANSITIONS

ROOT = Path(__file__).resolve().parents[1]
NATIVE = "tutorial/7_model_pruning_and_quantization_pytorch.ipynb"
COLAB = "tutorial/7_model_pruning_and_quantization_colab.ipynb"
FROZEN = "dev_tests/test_tutorial7_reference_outputs.py"


@pytest.fixture(scope="module")
def public_root(tmp_path_factory):
    return project_public_files(ROOT, tmp_path_factory.mktemp("t7_public"))


@contextmanager
def preserve(path):
    before = path.read_bytes()
    try:
        yield before
    finally:
        path.write_bytes(before)


def test_current_t7_scope_and_historical_predecessors_are_exact(public_root):
    assert assert_tutorial7_reference_scope(ROOT) == set(TRANSITIONS)
    assert assert_tutorial7_reference_scope(public_root) == set(TRANSITIONS)
    record = transition_record(public_root)
    for name, entry in record["transitions"].items():
        historical = historical_tutorial7_bytes(
            public_root, name, (public_root / name).read_bytes()
        )
        assert historical == (public_root / entry["archive"]).read_bytes()
    assert not (public_root / ".git").exists()
    assert not (public_root / "dev_tests/artifacts").exists()


@pytest.mark.parametrize("form", ["lf", "crlf"])
def test_only_bound_lf_and_crlf_notebook_containers_are_accepted(public_root, form):
    paths = [public_root / name for name in (NATIVE, COLAB)]
    before = [path.read_bytes() for path in paths]
    try:
        for path, raw in zip(paths, before):
            lf = raw.replace(b"\r\n", b"\n")
            path.write_bytes(lf if form == "lf" else lf.replace(b"\n", b"\r\n"))
        assert assert_tutorial7_reference_scope(public_root) == set(TRANSITIONS)
    finally:
        for path, raw in zip(paths, before):
            path.write_bytes(raw)


@pytest.mark.parametrize(
    "mutation",
    [
        "other-cell",
        "training-budget",
        "authorized-cell-extra",
        "output",
        "metadata",
        "cell-metadata",
        "cell-id",
        "reorder",
        "format",
        "mixed-eol",
        "bare-cr",
        "trailing-space",
    ],
)
def test_native_rejects_every_unapproved_source_or_container_change(
    public_root, mutation
):
    path = public_root / NATIVE
    with preserve(path) as raw:
        notebook = json.loads(raw)
        if mutation == "other-cell":
            notebook["cells"][14]["source"].append("# unauthorized\n")
        elif mutation == "training-budget":
            source = "".join(notebook["cells"][12]["source"])
            assert "pruning_epochs=50000" in source
            notebook["cells"][12]["source"] = source.replace(
                "pruning_epochs=50000", "pruning_epochs=1"
            ).splitlines(keepends=True)
        elif mutation == "authorized-cell-extra":
            notebook["cells"][37]["source"].append("# unauthorized\n")
        elif mutation == "output":
            notebook["cells"][8]["outputs"].append(
                {"output_type": "stream", "name": "stdout", "text": ["unauthorized\n"]}
            )
        elif mutation == "metadata":
            notebook["metadata"]["unauthorized"] = True
        elif mutation == "cell-metadata":
            notebook["cells"][37]["metadata"]["unauthorized"] = True
        elif mutation == "cell-id":
            notebook["cells"][37]["id"] = "unauthorized"
        elif mutation == "reorder":
            notebook["cells"][1], notebook["cells"][2] = (
                notebook["cells"][2],
                notebook["cells"][1],
            )
        if mutation == "mixed-eol":
            lf = raw.replace(b"\r\n", b"\n")
            changed = lf.replace(b"\n", b"\r\n", 1)
        elif mutation == "bare-cr":
            changed = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r", 1)
        elif mutation == "trailing-space":
            changed = raw + b" "
        else:
            changed = (
                json.dumps(notebook, indent=2 if mutation == "format" else 1) + "\n"
            ).encode("utf-8")
        path.write_bytes(changed)
        with pytest.raises(AssertionError, match=NATIVE):
            assert_tutorial7_reference_scope(public_root)


@pytest.mark.parametrize(
    "name,old,new",
    [
        (
            "dev_tests/generate_native_notebooks.py",
            "def epochs(full):",
            "def epochs(unauthorized):",
        ),
        (
            "dev_tests/generate_native_notebooks.py",
            "pruning_epochs=50000",
            "pruning_epochs=1",
        ),
        ("dev_tests/generate_native_notebooks.py", "seen += len(xb)", "seen += 1"),
        ("tests/build_tutorial_7.py", "modified_code_cells", "unauthorized_code_cells"),
        ("tests/test_tutorial_7.py", "places=12", "places=1"),
    ],
)
def test_python_sources_reject_changes_outside_exact_hunks(public_root, name, old, new):
    path = public_root / name
    with preserve(path) as raw:
        assert old.encode() in raw
        path.write_bytes(raw.replace(old.encode(), new.encode(), 1))
        with pytest.raises(AssertionError, match=name):
            assert_tutorial7_reference_scope(public_root)


@pytest.mark.parametrize(
    "name",
    [
        COLAB,
        SCOPE,
        RECORD,
        INVENTORY,
        FROZEN,
        "dev_tests/test_tutorial7_inference_artifacts.py",
        "dev_tests/test_tutorial7_reference_scope.py",
        "dev_tests/lbfgs_reference_scope.py",
        "dev_tests/publication_projection.py",
        "dev_tests/test_publication_checkout_eol.py",
    ],
)
def test_bound_sources_records_and_regressions_reject_tampering(public_root, name):
    path = public_root / name
    with preserve(path) as raw:
        path.write_bytes(raw + b"\n# unauthorized\n")
        with pytest.raises(AssertionError, match=name):
            assert_tutorial7_reference_scope(public_root)


@pytest.mark.parametrize(
    "name",
    [
        RECORD,
        INVENTORY,
        FROZEN,
        SCOPE,
        "dev_tests/fixtures/tutorial7_reference_v1/dev_tests__generate_native_notebooks.py.txt",
    ],
)
def test_new_required_files_are_mandatory_with_structured_identity(public_root, name):
    path = public_root / name
    with preserve(path):
        path.unlink()
        with pytest.raises(FileNotFoundError) as error:
            assert_tutorial7_reference_scope(public_root)
        assert Path(error.value.filename) == path


def test_new_archive_cannot_be_modified_or_used_to_relax_a_transition(public_root):
    name = (
        "dev_tests/fixtures/tutorial7_reference_v1/"
        "tutorial__7_model_pruning_and_quantization_pytorch.ipynb.txt"
    )
    path = public_root / name
    with preserve(path) as raw:
        path.write_bytes(raw + b"\n")
        with pytest.raises(AssertionError, match=name):
            assert_tutorial7_reference_scope(public_root)


def test_record_cannot_authorize_extra_cells_or_new_hashes(public_root):
    path = public_root / RECORD
    with preserve(path) as raw:
        record = json.loads(raw)
        record["transitions"][NATIVE]["notebook_patches"].append(
            {"cell": 5, "before": [], "after": []}
        )
        path.write_bytes(json.dumps(record).encode("utf-8"))
        with pytest.raises(AssertionError, match=RECORD):
            assert_tutorial7_reference_scope(public_root)


def test_inventory_preserves_every_previous_public_input(public_root):
    current = json.loads((public_root / INVENTORY).read_bytes())
    previous = json.loads(
        (public_root / "dev_tests/lbfgs_reference_v1_inventory.json").read_bytes()
    )
    assert current[: len(previous)] == previous
    assert len(current) == len(set(current))
    assert FROZEN in current and SCOPE in current and RECORD in current
