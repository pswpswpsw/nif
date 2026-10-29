"""Release metadata projection preserves the exact historical lint boundary."""
import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest

from dev_tests import lint_preservation as lint

ROOT = Path(__file__).resolve().parents[1]
ABOUT = "nif/__about__.py"
MANIFEST = "MANIFEST.in"
RELEASE_TEST = "dev_tests/test_lbfgs_reference_integration.py"
BEFORE = b'__version__ = "1.0.3"\n'
AFTER = b'__version__ = "2.0.0"\n'
READERS = (
    "historical_release_version_bytes",
    "historical_lint_bytes",
    "read_lint_bytes",
    "read_lint_text",
)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _read_release(root, reader, name=ABOUT):
    if reader.startswith("historical_"):
        return getattr(lint, reader)(root, name, (root / name).read_bytes())
    result = getattr(lint, reader)(root, name)
    return result.encode("utf-8") if isinstance(result, str) else result


@pytest.fixture
def release_root(tmp_path):
    record = lint.lint_record(ROOT)
    metadata = record["release_version_metadata"]
    names = (
        lint.SOURCE,
        lint.RECORD,
        ABOUT,
        MANIFEST,
        RELEASE_TEST,
        record["transitions"][RELEASE_TEST]["archive"],
        *metadata["archived_inputs"],
    )
    for name in names:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    return tmp_path


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize("ending", [b"\n", b"\r\n"])
def test_exact_release_metadata_projects_to_original_bytes(
    release_root, reader, ending
):
    current = AFTER.replace(b"\n", ending)
    (release_root / ABOUT).write_bytes(current)
    reader_path = release_root / lint.SOURCE
    reader_path.write_bytes(
        reader_path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", ending)
    )
    assert _read_release(release_root, reader) == BEFORE
    assert (release_root / ABOUT).read_bytes() == current


def test_projection_does_not_change_the_live_release_version():
    current = (ROOT / ABOUT).read_bytes()
    assert current in (AFTER, AFTER.replace(b"\n", b"\r\n"))
    assert lint.read_lint_bytes(ROOT, ABOUT) == BEFORE
    assert (ROOT / ABOUT).read_bytes() == current


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize("ending", [b"\n", b"\r\n"])
def test_exact_sdist_manifest_projects_to_original_bytes(release_root, reader, ending):
    metadata = lint.release_metadata_record(release_root)["sdist_manifest"]
    original = metadata["before_source"].encode("utf-8")
    path = release_root / MANIFEST
    current = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", ending)
    path.write_bytes(current)
    assert _read_release(release_root, reader, MANIFEST) == original
    assert path.read_bytes() == current


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize(
    "mutation",
    ["changed-input", "extra-include", "newline", "cr-only", "mixed", "original"],
)
def test_unrecorded_sdist_manifest_is_rejected(release_root, reader, mutation):
    path = release_root / MANIFEST
    current = path.read_bytes().replace(b"\r\n", b"\n")
    if mutation == "changed-input":
        current = current.replace(b"include README.md", b"include OTHER.md")
    elif mutation == "extra-include":
        current += b"include unexpected.txt\n"
    elif mutation == "newline":
        current += b"\n"
    elif mutation == "cr-only":
        current = current.replace(b"\n", b"\r")
    elif mutation == "mixed":
        current = current.replace(b"\n", b"\r\n", 1)
    else:
        metadata = lint.release_metadata_record(release_root)["sdist_manifest"]
        current = metadata["before_source"].encode("utf-8")
    path.write_bytes(current)
    with pytest.raises(AssertionError, match=re.escape(MANIFEST)):
        _read_release(release_root, reader, MANIFEST)


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize("ending", [b"\n", b"\r\n"])
def test_exact_test_read_boundary_projects_through_the_original_lint_transition(
    release_root, reader, ending
):
    record = lint.lint_record(release_root)
    metadata = lint.release_metadata_record(release_root)["test_reader"]
    path = release_root / RELEASE_TEST
    current = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", ending)
    path.write_bytes(current)
    archive = metadata["archive"]
    if reader != "historical_release_version_bytes":
        archive = record["transitions"][RELEASE_TEST]["archive"]
    original = (release_root / archive).read_bytes()
    if reader == "read_lint_text":
        original = original.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    assert _read_release(release_root, reader, RELEASE_TEST) == original
    assert path.read_bytes() == current


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize("mutation", ["added-code", "golden-assertion", "original"])
def test_unrecorded_test_read_boundary_is_rejected(release_root, reader, mutation):
    path = release_root / RELEASE_TEST
    current = path.read_bytes().replace(b"\r\n", b"\n")
    if mutation == "added-code":
        current += b"\nunrecorded = True\n"
    elif mutation == "golden-assertion":
        original = current
        current = current.replace(
            b"include nif/optimizers/TFP_LICENSE.txt",
            b"include nif/optimizers/OTHER.txt",
        )
        assert current != original
    else:
        metadata = lint.release_metadata_record(release_root)["test_reader"]
        current = (release_root / metadata["archive"]).read_bytes()
    path.write_bytes(current)
    with pytest.raises(AssertionError, match=re.escape(RELEASE_TEST)):
        _read_release(release_root, reader, RELEASE_TEST)


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize(
    "current",
    [
        b'__version__ = "2.0.1"\n',
        AFTER + b"unexpected = True\n",
        AFTER + b"\n",
        AFTER.replace(b"\n", b"\r"),
        BEFORE,
    ],
    ids=["wrong-version", "added-code", "extra-newline", "cr-only", "old-version"],
)
def test_unrecorded_release_source_is_rejected(release_root, reader, current):
    (release_root / ABOUT).write_bytes(current)
    with pytest.raises(AssertionError, match=re.escape(ABOUT)):
        _read_release(release_root, reader)


@pytest.mark.parametrize(
    "reader", ["historical_release_version_bytes", "historical_lint_bytes"]
)
def test_unrelated_paths_pass_through_without_release_provenance(tmp_path, reader):
    current = AFTER + b"unrelated bytes\r"
    assert getattr(lint, reader)(tmp_path, "unrelated.txt", current) == current


def test_release_metadata_records_only_the_exact_version_transition(release_root):
    metadata = lint.release_metadata_record(release_root)
    assert metadata == lint.lint_record(release_root)["release_version_metadata"]
    assert lint.RELEASE_VERSION_SOURCE == ABOUT
    assert lint.RELEASE_VERSION_BEFORE == BEFORE
    assert metadata["path"] == ABOUT
    assert metadata["before_version"] == "1.0.3"
    assert metadata["after_version"] == "2.0.0"
    assert metadata["before_sha256"] == _digest(BEFORE)
    assert metadata["after_sha256_by_eol"] == {
        "lf": _digest(AFTER),
        "crlf": _digest(AFTER.replace(b"\n", b"\r\n")),
    }
    assert set(metadata["archived_inputs"]) == {
        metadata["prior_lint_record"],
        metadata["prior_lint_reader"],
        metadata["test_reader"]["archive"],
    }
    for name, digest in metadata["archived_inputs"].items():
        assert _digest((release_root / name).read_bytes()) == digest


def test_sdist_record_contains_only_the_three_release_includes(release_root):
    metadata = lint.release_metadata_record(release_root)["sdist_manifest"]
    assert lint.RELEASE_MANIFEST_SOURCE == MANIFEST
    assert metadata["path"] == MANIFEST
    original = metadata["before_source"].encode("utf-8")
    current = (release_root / MANIFEST).read_bytes().replace(b"\r\n", b"\n")
    assert metadata["before_sha256"] == _digest(original)
    assert metadata["after_sha256_by_eol"] == {
        "lf": _digest(current),
        "crlf": _digest(current.replace(b"\n", b"\r\n")),
    }
    additions = {
        b"include CITATION.cff",
        b"include .zenodo.json",
        b"include scripts/check_release.py",
    }
    current_lines = current.splitlines()
    assert all(current_lines.count(line) == 1 for line in additions)
    assert [line for line in current_lines if line not in additions] == (
        original.splitlines()
    )


def test_test_reader_record_changes_only_the_manifest_read_boundary(release_root):
    metadata = lint.release_metadata_record(release_root)["test_reader"]
    assert lint.RELEASE_TEST_SOURCE == RELEASE_TEST
    assert metadata["path"] == RELEASE_TEST
    original = (release_root / metadata["archive"]).read_bytes()
    current = (release_root / RELEASE_TEST).read_bytes().replace(b"\r\n", b"\n")
    assert metadata["before_sha256"] == _digest(original)
    assert metadata["after_sha256_by_eol"] == {
        "lf": _digest(current),
        "crlf": _digest(current.replace(b"\n", b"\r\n")),
    }
    old = b'(ROOT / "MANIFEST.in").read_bytes()'
    new = b'read_lint_bytes(ROOT, "MANIFEST.in")'
    original_lf = original.replace(b"\r\n", b"\n")
    assert original_lf.count(old) == 1
    assert current == original_lf.replace(old, new, 1)


@pytest.mark.parametrize("reader", READERS)
def test_changed_test_reader_archive_is_rejected(release_root, reader):
    metadata = lint.release_metadata_record(release_root)["test_reader"]
    name = metadata["archive"]
    path = release_root / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(AssertionError, match=re.escape(name)):
        _read_release(release_root, reader, RELEASE_TEST)


@pytest.mark.parametrize("archive_key", ["prior_lint_record", "prior_lint_reader"])
@pytest.mark.parametrize("reader", ["metadata", "projection"])
@pytest.mark.parametrize("source", [ABOUT, MANIFEST])
def test_changed_prior_lint_evidence_is_rejected(
    release_root, archive_key, reader, source
):
    metadata = lint.lint_record(release_root)["release_version_metadata"]
    name = metadata[archive_key]
    path = release_root / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(AssertionError, match=re.escape(name)):
        if reader == "metadata":
            lint.release_metadata_record(release_root)
        else:
            lint.read_lint_bytes(release_root, source)


@pytest.mark.parametrize("mutation", ["bytes", "release-version", "sdist-manifest"])
def test_modified_release_manifest_fails_its_pinned_hash(release_root, mutation):
    path = release_root / lint.RECORD
    data = path.read_bytes()
    if mutation == "bytes":
        path.write_bytes(data + b"\n")
    else:
        record = json.loads(data)
        if mutation == "release-version":
            record["release_version_metadata"]["after_version"] = "2.0.1"
        else:
            record["release_version_metadata"]["sdist_manifest"][
                "before_source"
            ] = "include unrecorded.txt\n"
        path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(AssertionError, match=re.escape(lint.RECORD)):
        lint.read_lint_bytes(release_root, ABOUT)


@pytest.mark.parametrize(
    "mutation", ["source", "mixed-endings", "lone-carriage-return"]
)
@pytest.mark.parametrize("source", [ABOUT, MANIFEST])
def test_modified_release_reader_is_rejected(release_root, mutation, source):
    path = release_root / lint.SOURCE
    data = path.read_bytes().replace(b"\r\n", b"\n")
    if mutation == "source":
        data += b"\n# unrecorded change\n"
    elif mutation == "mixed-endings":
        data = data.replace(b"\n", b"\r\n", 1)
    else:
        data = data.replace(b"\n", b"\r", 1)
    path.write_bytes(data)
    with pytest.raises(AssertionError, match=re.escape(lint.SOURCE)):
        lint.read_lint_bytes(release_root, source)


def test_release_keeps_all_original_lint_transitions_and_public_inputs():
    record = lint.lint_record(ROOT)
    metadata = lint.release_metadata_record(ROOT)
    prior = json.loads((ROOT / metadata["prior_lint_record"]).read_bytes())
    assert len(prior["transitions"]) == 82
    assert len(prior["public_files"]) == 166
    assert record["transitions"] == prior["transitions"]
    assert record["public_files"] == prior["public_files"]
    assert record["base_commit"] == prior["base_commit"]
    public = lint.lint_public_files(ROOT)
    assert public[: len(prior["public_files"])] == prior["public_files"]
    assert set(public[len(prior["public_files"]) :]) == {
        ABOUT,
        MANIFEST,
        *metadata["archived_inputs"],
    }
    assert len(public) == len(set(public))
