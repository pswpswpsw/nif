"""Exact public final-update succession; historical numerical checks stay fixed."""
import hashlib
import json
from pathlib import Path

import pytest

from dev_tests.adabelief_moment_scope import assert_adabelief_phase_scope_v4
from dev_tests.lbfgs_reference_scope import historical_bytes
from dev_tests.lint_preservation import read_lint_bytes
from dev_tests.publication_windows_scope import assert_publication_windows_scope_v2

ROOT = Path(__file__).resolve().parents[1]
RECORD = "dev_tests/adabelief_final_update_v5.json"
RECORD_SHA256 = "c57a3387831a0d6b31eab168ce629bcc208f19d638bba535b0e00b3fd2196b10"


def record():
    data = (ROOT / RECORD).read_bytes()
    assert hashlib.sha256(data).hexdigest() == RECORD_SHA256
    return json.loads(data)


@pytest.mark.parametrize(
    "name",
    [
        "dev_tests/adabelief_moment_scope.py",
        "dev_tests/publication_windows_scope.py",
        "dev_tests/test_publication_checkout_successor.py",
    ],
)
def test_exact_final_update_source_successors_and_historical_test_bytes(name):
    frozen = record()
    current = historical_bytes(ROOT, name, (ROOT / name).read_bytes())
    teaching = json.loads(
        (ROOT / "dev_tests/teaching_markdown_v1_sources.json").read_text(
            encoding="utf-8"
        )
    )
    entries = {
        key: teaching["ada_compatibility"][key]
        for key in [
            "dev_tests/adabelief_moment_scope.py",
            "dev_tests/publication_windows_scope.py",
            "dev_tests/test_publication_checkout_successor.py",
        ]
    }
    assert (
        hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
        == "69a0b3c1bc6394c36b1f7d4be69fd789fdb931cc7c0e91d635fd74ac950135b9"
    )
    entry = entries[name]
    assert hashlib.sha256(current).hexdigest() == entry["after_sha256"], name
    for patch in reversed(entry["transforms"]):
        old, new = patch["old"].encode(), patch["new"].encode()
        assert current.count(new) == 1
        current = current.replace(new, old, 1)
    assert hashlib.sha256(current).hexdigest() == entry["before_sha256"], name
    assert (
        hashlib.sha256(current).hexdigest() == frozen["source_after_sha256"][name]
    ), name
    change = frozen["exact_transforms"][name]
    assert current.count(change["new"].encode()) == 1
    previous = current.replace(change["new"].encode(), change["old"].encode(), 1)
    assert (
        hashlib.sha256(previous).hexdigest() == frozen["source_before_sha256"][name]
    ), name
    archived = (ROOT / frozen["historical_test_archive"]).read_bytes()
    assert (
        hashlib.sha256(archived).hexdigest()
        == frozen["source_before_sha256"][
            "dev_tests/test_publication_checkout_successor.py"
        ]
    )
    assert (
        hashlib.sha256(
            read_lint_bytes(ROOT, "dev_tests/test_adabelief_final_update_order.py")
        ).hexdigest()
        == frozen["frozen_final_update_test_sha256"]
    )


def test_windows_guard_accepts_only_the_exact_current_moment_helper():
    assert len(assert_publication_windows_scope_v2()) == 8


@pytest.mark.parametrize(
    "name,old,new",
    [
        (
            "nif/optimizers/external_optimizers.py",
            "p.sub_(direction * lr)",
            "p.add_(direction, alpha=-lr)",
        ),
        (
            "nif/optimizers/external_optimizers.py",
            "p.sub_(direction * lr)",
            "p.sub_(direction * (lr * 2))",
        ),
        ("nif/model.py", "class NIF", "class UnauthorizedNIF"),
        (
            "dev_tests/adabelief_moment_scope.py",
            "p.sub_(direction * lr)",
            "p.sub_(direction * (lr * 2))",
        ),
    ],
)
def test_final_update_successor_rejects_unrelated_optimizer_model_or_helper_changes(
    monkeypatch, name, old, new
):
    original = Path.read_text
    original_bytes = Path.read_bytes

    def changed(path, *args, **kwargs):
        source = original(path, *args, **kwargs)
        if path == ROOT / name:
            assert old in source
            return source.replace(old, new, 1)
        return source

    def changed_bytes(path):
        data = original_bytes(path)
        # Untransformed model integrity is deliberately checked as raw bytes.
        if name == "nif/model.py" and path == ROOT / name:
            assert old.encode() in data
            return data.replace(old.encode(), new.encode(), 1)
        return data

    monkeypatch.setattr(Path, "read_text", changed)
    monkeypatch.setattr(Path, "read_bytes", changed_bytes)
    with pytest.raises(AssertionError, match=name):
        assert_adabelief_phase_scope_v4()
