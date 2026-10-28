"""Public-only test projection and optional local historical-output audit.

Projection inputs are an explicit public inventory, never discovered at runtime
through Git or file-existence fallbacks. The local audit is separate from the
mandatory source/fixture/provenance scope guard.
"""
import hashlib
import json
import shutil
from pathlib import Path

from dev_tests.lint_preservation import lint_public_files

ROOT = Path(__file__).resolve().parents[1]
GENERATED_PREFIX = "dev_tests/artifacts/"


def project_public_files(source, destination, files=None):
    source, destination = Path(source), Path(destination)
    if files is None:
        files = json.loads(
            (source / "dev_tests/tutorial7_reference_v1_inventory.json").read_text(
                encoding="utf-8"
            )
        )
    files = list(dict.fromkeys([*files, *lint_public_files(source)]))
    for name in files:
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts, name
        assert relative.parts[0] != ".git" and not name.startswith(
            GENERATED_PREFIX
        ), name
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)  # Every inventory entry is required.
    assert not (destination / ".git").exists()
    assert not (destination / "dev_tests/artifacts").exists()
    return destination


def audit_optional_local_history(root=ROOT):
    """Check present historical generated outputs; explicitly report absences.

    This is OPTIONAL LOCAL integrity, not a publication requirement. Only the
    exact namespace recorded in the immutable moment-phase manifest qualifies.
    A present generated output must still match its original digest.
    """
    root = Path(root)
    fingerprints = json.loads(
        (root / "dev_tests/adabelief_moment_pre_fix_fingerprints.json").read_text(
            encoding="utf-8"
        )
    )
    result = {"namespace": GENERATED_PREFIX, "present": [], "missing": []}
    for name, digest in fingerprints.items():
        if not name.startswith(GENERATED_PREFIX):
            continue
        path = root / name
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
            result["present"].append(name)
        else:
            result["missing"].append(name)
    return result
