"""Eight published Markdown edits only; exact successors and substantive negatives."""
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from dev_tests import publication_windows_scope as windows_scope
from dev_tests import tutorial1_gc_scope as teaching_scope
from dev_tests.lbfgs_reference_scope import historical_bytes
from dev_tests.lint_preservation import read_lint_bytes
from dev_tests.publication_projection import project_public_files

ROOT = Path(__file__).resolve().parents[1]
RECORD = "dev_tests/teaching_markdown_v1.json"
SOURCES = "dev_tests/teaching_markdown_v1_sources.json"
RECORD_HASH = "aad2865228b9f93d9d47cd56c720f6c3fed56abb7ae1d430311970f0f428adfa"
SOURCES_HASH = "52a42c9f5c04e41a4d259094a52cdcd5a06cd6ea6c3d0aa00a4551325273d832"
T1 = "tutorial/1_simple_1d_wave_pytorch.ipynb"


def frozen(name, digest):
    raw = (ROOT / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest, name
    return json.loads(raw)


@pytest.fixture
def teaching_root(tmp_path):
    record = frozen(RECORD, RECORD_HASH)
    names = [RECORD, *record["required_archives"], *record["notebooks"]]
    names += [
        "dev_tests/lbfgs_reference_v1.json",
        "dev_tests/fixtures/lbfgs_reference_v1/tutorial__1_simple_1d_wave_pytorch.ipynb.txt",
    ]
    for name in names:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    return tmp_path


def validate(root):
    function = getattr(teaching_scope, "assert_teaching_markdown_scope", None)
    assert callable(function), "exact published Markdown JSON guard is required"
    return function(root)


def test_published_eight_cells_and_all_exact_source_successors():
    record = frozen(RECORD, RECORD_HASH)
    sources = frozen(SOURCES, SOURCES_HASH)
    assert record["published_commit"] == "5f8a88b41ce41e09bbd5aaef8456950ecb3b0501"
    assert record["before_commit"] == "541c7feacd041b086c181d151076f443bd2d15d7"
    assert sum(len(item["changes"]) for item in record["notebooks"].values()) == 8
    for name, patches in sources["exact_transforms"].items():
        current = historical_bytes(ROOT, name, (ROOT / name).read_bytes())
        assert (
            hashlib.sha256(current).hexdigest() == sources["source_after_sha256"][name]
        ), name
        for patch in reversed(patches):
            old, new = patch["old"].encode(), patch["new"].encode()
            assert current.count(new) == 1
            current = current.replace(new, old, 1)
        assert (
            hashlib.sha256(current).hexdigest() == sources["source_before_sha256"][name]
        ), name
        assert current == (ROOT / record["source_archive"][name]).read_bytes(), name
    assert hashlib.sha256(
        read_lint_bytes(ROOT, "nif/optimizers/external_optimizers.py")
    ).hexdigest() == (
        "956a2f80f6dfee6c5603a3ed981d5bc0e2d537e156cc6e1e95e8fba6519674a4"
    )
    assert (
        hashlib.sha256(
            read_lint_bytes(ROOT, "dev_tests/test_adabelief_final_update_order.py")
        ).hexdigest()
        == sources["frozen_parameter_stage_test_sha256"]
    )
    assert (
        hashlib.sha256(
            (ROOT / "dev_tests/adabelief_final_update_v5.json").read_bytes()
        ).hexdigest()
        == sources["frozen_ada_record_sha256"]
    )
    assert sources["diff_path"] == "dev_tests/fixtures/teaching_markdown_v1/scope.diff"
    assert (
        hashlib.sha256((ROOT / sources["diff_path"]).read_bytes()).hexdigest()
        == sources["diff_sha256"]
    )
    assert validate(ROOT) == set(record["notebooks"])


def test_json_guard_accepts_without_git_or_generated_outputs(teaching_root):
    assert not (teaching_root / ".git").exists()
    assert not (teaching_root / "dev_tests/artifacts").exists()
    assert validate(teaching_root) == set(frozen(RECORD, RECORD_HASH)["notebooks"])


@pytest.mark.parametrize(
    "mutation",
    [
        "ninth-markdown",
        "authorized-cell-other-text",
        "code",
        "output",
        "metadata",
        "cell-metadata",
        "insert-cell",
        "delete-cell",
        "reorder-cells",
    ],
)
def test_rejects_every_unapproved_notebook_surface(teaching_root, mutation):
    path = teaching_root / T1
    notebook = json.loads(path.read_bytes())
    code = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    if mutation == "ninth-markdown":
        notebook["cells"][0]["source"].append("Unauthorized ninth edit.\n")
    elif mutation == "authorized-cell-other-text":
        notebook["cells"][8]["source"].append("Unauthorized extra text.\n")
    elif mutation == "code":
        code["source"].append("# unauthorized\n")
    elif mutation == "output":
        code["outputs"].append(
            {"output_type": "stream", "name": "stdout", "text": ["unauthorized\n"]}
        )
    elif mutation == "metadata":
        notebook["metadata"]["unauthorized"] = True
    elif mutation == "cell-metadata":
        notebook["cells"][8]["metadata"]["unauthorized"] = True
    elif mutation == "insert-cell":
        notebook["cells"].append(
            {"cell_type": "markdown", "metadata": {}, "source": ["new"]}
        )
    elif mutation == "delete-cell":
        notebook["cells"].pop()
    else:
        notebook["cells"][8], notebook["cells"][10] = (
            notebook["cells"][10],
            notebook["cells"][8],
        )
    path.write_text(json.dumps(notebook), encoding="utf-8")
    with pytest.raises(AssertionError, match=T1):
        validate(teaching_root)


ARCHIVES = list(frozen(RECORD, RECORD_HASH)["required_archives"])


@pytest.mark.parametrize("name", ARCHIVES)
def test_every_new_archive_is_mandatory(teaching_root, name):
    (teaching_root / name).unlink()
    with pytest.raises(FileNotFoundError) as error:
        validate(teaching_root)
    assert Path(error.value.filename) == teaching_root / name


@pytest.mark.parametrize("name", [ARCHIVES[0], ARCHIVES[-1]])
def test_notebook_and_helper_archives_reject_tampering(teaching_root, name):
    path = teaching_root / name
    path.write_bytes(path.read_bytes() + b"\nunauthorized\n")
    with pytest.raises(AssertionError, match=name):
        validate(teaching_root)


def test_teaching_record_cannot_authorize_a_ninth_change(teaching_root):
    path = teaching_root / RECORD
    record = json.loads(path.read_bytes())
    record["notebooks"][T1]["changes"].append({"cell": 0, "before": [], "after": []})
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(AssertionError, match=RECORD):
        validate(teaching_root)


def test_new_inventory_preserves_order_and_clean_projection(tmp_path):
    sources = frozen(SOURCES, SOURCES_HASH)
    name = "dev_tests/teaching_markdown_v1_inventory.json"
    raw = (ROOT / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == sources["inventory_sha256"]
    current = json.loads(raw)
    old = json.loads(
        (ROOT / "dev_tests/publication_windows_v2_inventory.json").read_bytes()
    )
    assert current[: len(old)] == old and len(current) == 407 and len(old) == 375
    assert len(current) == len(set(current))
    assert all(
        not Path(n).is_absolute()
        and ".." not in Path(n).parts
        and "artifacts" not in Path(n).parts
        and ".git" not in Path(n).parts
        for n in current
    )
    projected = project_public_files(ROOT, tmp_path / "public")
    assert (
        not (projected / ".git").exists()
        and not (projected / "dev_tests/artifacts").exists()
    )
    assert all((projected / n).is_file() for n in current)
    assert validate(projected) == set(frozen(RECORD, RECORD_HASH)["notebooks"])
    assert len(windows_scope.assert_publication_windows_scope_v2(projected)) == 8


def test_relaxed_json_guard_source_is_rejected(tmp_path):
    project_public_files(ROOT, tmp_path, files=["dev_tests/lbfgs_reference_v1.json"])
    path = tmp_path / "dev_tests/tutorial1_gc_scope.py"
    data = (ROOT / "dev_tests/tutorial1_gc_scope.py").read_bytes()
    old = b"assert json.loads(current) == expected, name"
    assert data.count(old) == 1
    path.write_bytes(data.replace(old, b"assert True", 1))
    shutil.copyfile(
        ROOT / "dev_tests/lbfgs_reference_v1.json",
        tmp_path / "dev_tests/lbfgs_reference_v1.json",
    )
    function = getattr(windows_scope, "assert_teaching_helper_source", None)
    assert callable(function)
    with pytest.raises(AssertionError, match="dev_tests/tutorial1_gc_scope.py"):
        function(tmp_path)


def test_new_readers_are_explicit_utf8_under_gbk_defaults(monkeypatch):
    original = Path.read_text

    def gbk_default(path, encoding=None, errors=None, **kwargs):
        return original(path, encoding=encoding or "gbk", errors=errors, **kwargs)

    monkeypatch.setattr(Path, "read_text", gbk_default)
    assert validate(ROOT) == set(frozen(RECORD, RECORD_HASH)["notebooks"])
    assert len(windows_scope.assert_publication_windows_scope_v2()) == 8
