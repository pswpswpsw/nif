"""Frozen history remains reproducible without binding it to current sources."""
import ast
import hashlib
import json
import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASE = "73b2ffcf71d9f0d019bd86d62cbf80989e9f7eac"


def snapshot(**kwargs):
    from dev_tests.provenance import Snapshot

    return Snapshot(ROOT, **kwargs)


def test_frozen_sources_are_independent_of_live_cleanup(tmp_path):
    saved = snapshot()
    assert saved.manifest["base_commit"] == BASE
    names = ["nif/model.py", "dev_tests/lint_preservation.py", ".README.md.un~"]
    view = saved.materialize(tmp_path / "view", names=names)
    assert not (view / ".git").exists()
    for name in names:
        assert (view / name).read_bytes() == saved.read(name)
    assert b"def _policy(policy):" in (view / "nif/model.py").read_bytes()
    assert b"RECORD_SHA256" in (view / "dev_tests/lint_preservation.py").read_bytes()


def test_equal_historical_evidence_uses_one_content_object():
    saved = snapshot()
    first = "dev_tests/adabelief_moment_red.log"
    second = "dev_tests/fixtures/adabelief_moment_v4/adabelief_moment_red.log.txt"
    assert (
        saved.manifest["files"][first]["sha256"]
        == saved.manifest["files"][second]["sha256"]
    )
    assert saved.read(first) == saved.read(second)
    assert saved.manifest["deduplicated_bytes"] >= 247886


@pytest.mark.parametrize(
    "name", ["../outside.py", "/absolute.py", "C:/outside.py", "nif\\model.py"]
)
def test_archive_paths_cannot_escape_the_projection(name, tmp_path):
    with pytest.raises(ValueError, match="relative POSIX"):
        snapshot().materialize(tmp_path / "view", names=[name])
    assert not (tmp_path / "outside.py").exists()


def test_modified_manifest_and_bundle_are_rejected(tmp_path):
    saved = snapshot()
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(saved.manifest_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="manifest digest"):
        snapshot(manifest_path=manifest)
    bundle = tmp_path / "objects.zip"
    bundle.write_bytes(saved.bundle_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="bundle digest"):
        snapshot(bundle_path=bundle)


def test_protected_originals_are_required_and_checked(tmp_path):
    from dev_tests.provenance import Snapshot

    name = "misc/myimage.gif"
    saved = Snapshot(tmp_path)
    with pytest.raises(FileNotFoundError):
        saved.read(name)
    target = tmp_path / name
    target.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / name, target)
    assert (
        hashlib.sha256(saved.read(name)).hexdigest()
        == saved.manifest["files"][name]["sha256"]
    )
    target.write_bytes(target.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="backing digest"):
        saved.read(name)


def test_original_notebook_accepts_only_exact_checkout_line_endings(tmp_path):
    from dev_tests.provenance import Snapshot

    name = "tutorial/7_model_pruning_and_quantization_colab.ipynb"
    raw = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
    target = tmp_path / name
    target.parent.mkdir(parents=True)
    target.write_bytes(raw.replace(b"\n", b"\r\n"))
    assert Snapshot(tmp_path).read(name) == raw
    target.write_bytes(raw.replace(b"\n", b"\r\n", 1))
    with pytest.raises(ValueError, match="backing digest"):
        Snapshot(tmp_path).read(name)


def test_current_pruning_numerical_assertions_remain_live_and_unchanged():
    name = "dev_tests/test_pruning_adam_state.py"
    old = ast.parse(snapshot().read(name))
    current = ast.parse((ROOT / name).read_text(encoding="utf-8"))
    historical = "test_red_test_fingerprints_unchanged_and_scientific_calls_migrated_v2"
    expected = {
        node.name: ast.dump(node)
        for node in old.body
        if isinstance(node, ast.FunctionDef) and node.name != historical
    }
    actual = {
        node.name: ast.dump(node)
        for node in current.body
        if isinstance(node, ast.FunctionDef)
    }
    assert actual == expected


def test_archived_paths_remain_recoverable_and_scientific_inputs_stay_current():
    saved = snapshot()
    migration = json.loads((ROOT / "dev_tests/provenance/migration.json").read_text())
    for name in migration["archived_live_paths"]:
        assert not (ROOT / name).exists(), name
        assert (
            hashlib.sha256(saved.read(name)).hexdigest()
            == saved.manifest["files"][name]["sha256"]
        )
    for name in migration["retained_current_scientific_inputs"]:
        assert (ROOT / name).read_bytes() == saved.read(name), name
