"""Exact lint provenance rejects unrecorded source, archive, and reader changes."""
import json
import re
import shutil
from pathlib import Path

import pytest

from dev_tests import lint_preservation as lint

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = "dev_tests/generate_native_notebooks.py"


@pytest.fixture
def lint_root(tmp_path):
    entry = lint.lint_record(ROOT)["transitions"][SAMPLE]
    for name in (lint.SOURCE, lint.RECORD, SAMPLE, entry["archive"]):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    return tmp_path


@pytest.mark.parametrize("ending", [b"\n", b"\r\n"])
def test_exact_recorded_source_and_whole_checkout_endings(lint_root, ending):
    current = (lint_root / SAMPLE).read_bytes().replace(b"\r\n", b"\n")
    current = current.replace(b"\n", ending)
    reader = lint_root / lint.SOURCE
    reader.write_bytes(
        reader.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", ending)
    )
    entry = lint.lint_record(lint_root)["transitions"][SAMPLE]
    assert (
        lint.historical_lint_bytes(lint_root, SAMPLE, current)
        == (lint_root / entry["archive"]).read_bytes()
    )


def test_unrecorded_current_source_is_rejected(lint_root):
    current = (lint_root / SAMPLE).read_bytes() + b"\n"
    with pytest.raises(AssertionError, match=re.escape(SAMPLE)):
        lint.historical_lint_bytes(lint_root, SAMPLE, current)


@pytest.mark.parametrize("ending", [b"\r\n", b"\r"])
def test_text_reader_rejects_mixed_or_lone_carriage_returns(lint_root, ending):
    path = lint_root / SAMPLE
    data = path.read_bytes().replace(b"\r\n", b"\n")
    path.write_bytes(data.replace(b"\n", ending, 1))
    with pytest.raises(AssertionError, match=re.escape(SAMPLE)):
        lint.read_lint_text(lint_root, SAMPLE)


def test_modified_archive_is_rejected(lint_root):
    name = lint.lint_record(lint_root)["transitions"][SAMPLE]["archive"]
    path = lint_root / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(AssertionError, match=re.escape(name)):
        lint.read_lint_bytes(lint_root, SAMPLE)


@pytest.mark.parametrize("mutation", ["bytes", "drop-transition"])
def test_modified_manifest_is_rejected(lint_root, mutation):
    path = lint_root / lint.RECORD
    data = path.read_bytes()
    if mutation == "bytes":
        path.write_bytes(data + b"\n")
    else:
        record = json.loads(data)
        del record["transitions"][SAMPLE]
        path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(AssertionError, match=re.escape(lint.RECORD)):
        lint.read_lint_bytes(lint_root, SAMPLE)


@pytest.mark.parametrize(
    "mutation", ["source", "mixed-endings", "lone-carriage-return"]
)
def test_modified_reader_is_rejected(lint_root, mutation):
    path = lint_root / lint.SOURCE
    data = path.read_bytes().replace(b"\r\n", b"\n")
    if mutation == "source":
        data += b"\n# unrecorded change\n"
    elif mutation == "mixed-endings":
        data = data.replace(b"\n", b"\r\n", 1)
    else:
        data = data.replace(b"\n", b"\r", 1)
    path.write_bytes(data)
    with pytest.raises(AssertionError, match=re.escape(lint.SOURCE)):
        lint.read_lint_bytes(lint_root, SAMPLE)


def test_missing_archive_is_required(lint_root):
    name = lint.lint_record(lint_root)["transitions"][SAMPLE]["archive"]
    path = lint_root / name
    path.unlink()
    with pytest.raises(FileNotFoundError) as error:
        lint.read_lint_bytes(lint_root, SAMPLE)
    assert Path(error.value.filename) == path


def test_unmapped_names_pass_through_without_a_source_transition(tmp_path):
    data = b"unmapped immutable fixture"
    assert lint.historical_lint_bytes(tmp_path, "unmapped.txt", data) == data


def test_current_text_reader_explicitly_uses_utf8_under_gbk_default(
    monkeypatch, lint_root
):
    entry = lint.lint_record(lint_root)["transitions"][SAMPLE]
    expected = (lint_root / entry["archive"]).read_bytes().decode("utf-8")
    expected = expected.replace("\r\n", "\n").replace("\r", "\n")
    original = Path.read_text
    encodings = []

    def gbk_default(path, encoding=None, errors=None, **kwargs):
        if path == lint_root / SAMPLE:
            encodings.append(encoding)
        return original(path, encoding=encoding or "gbk", errors=errors, **kwargs)

    monkeypatch.setattr(Path, "read_text", gbk_default)
    assert lint.read_lint_text(lint_root, SAMPLE) == expected
    assert encodings == ["utf-8"]
