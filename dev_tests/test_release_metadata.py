"""Release metadata must identify the same software version as the package."""
import ast
import io
import json
import re
import shutil
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.check_release import check_distributions
from scripts.check_release import check_metadata

ROOT = Path(__file__).resolve().parents[1]


def test_release_checkout_ref_cannot_select_a_same_named_branch():
    workflow = (ROOT / ".github/workflows/release.yaml").read_text(encoding="utf-8")
    refs = re.findall(r"^\s+ref:\s*(.+)$", workflow, re.MULTILINE)
    assert len(refs) == 1, "The release build must explicitly select its source"
    expression = refs[0]
    # Checkout resolves an unqualified name as a branch before trying a tag.
    # Published releases therefore use the event commit; manual runs name the
    # tag namespace, so an existing branch called v2.0.0 cannot substitute code.
    assert (
        "github.event.release.tag_name" not in expression
    ), "Published releases must build github.sha, not a short tag name"
    assert re.search(
        r"format\(\s*['\"]refs/tags/\{0\}['\"]\s*,\s*inputs.release_tag\s*\)",
        expression,
    ), "Manual releases must explicitly select refs/tags/<release_tag>"
    assert expression.rstrip().endswith("|| github.sha }}")


def test_current_release_metadata_agrees_with_package():
    source = ast.parse((ROOT / "nif/__about__.py").read_text(encoding="utf-8"))
    version = next(
        ast.literal_eval(node.value)
        for node in source.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        )
    )
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    declared = re.search(r"^version:\s*['\"]?([^'\"\s]+)", citation, re.MULTILINE)
    assert declared, "CITATION.cff must declare the released software version"
    assert declared.group(1) == version
    assert check_metadata(ROOT, "v" + version) == version


@pytest.fixture
def release_source(tmp_path):
    for name in ("nif/__about__.py", "CITATION.cff", ".zenodo.json"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    return tmp_path


def test_tag_mismatch_is_rejected(release_source):
    with pytest.raises(ValueError, match="tag does not match"):
        check_metadata(release_source, "v0.0.0")


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("version", "0.0.0", "Citation version mismatch"),
        ("date-released", "2026-02-30", "day is out of range"),
        ("type", "article", "identify software"),
        ("doi", "10.48550/arXiv.2204.03216", "Software DOI"),
        ("repository-code", "https://example.org/other", "repository mismatch"),
    ],
)
def test_invalid_software_citation_is_rejected(release_source, field, value, message):
    path = release_source / "CITATION.cff"
    original = path.read_text(encoding="utf-8")
    changed, count = re.subn(
        r"^" + field + r":.*$",
        field + ": '" + value + "'",
        original,
        count=1,
        flags=re.MULTILINE,
    )
    assert count == 1
    path.write_text(changed, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        check_metadata(release_source)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("version", "0.0.0", "Zenodo version mismatch"),
        ("publication_date", "2020-01-01", "Zenodo release date mismatch"),
        ("title", "Another project", "Zenodo title mismatch"),
        ("doi", "10.5281/zenodo.7909312", "Do not reuse"),
    ],
)
def test_zenodo_metadata_drift_is_rejected(release_source, field, value, message):
    path = release_source / ".zenodo.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata[field] = value
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        check_metadata(release_source)


def write_distributions(
    root, *, wheel_version="2.0.0", source_version="2.0.0", omit=None, extra=None
):
    package = {
        "nif/__about__.py": b'__version__ = "2.0.0"\n',
        "nif/model.py": b"# test artifact\n",
        "nif/demo/dataset/cylinderflow.npz": b"dataset",
        "nif/demo/dataset/traveling_wave.npz": b"dataset",
        "nif/demo/dataset/traveling_wave_high_freq.npz": b"dataset",
        "nif/optimizers/TFP_LICENSE.txt": b"license",
        "nif/optimizers/TFP_NOTICE.txt": b"notice",
    }
    if omit:
        del package[omit]
    if extra:
        package[extra] = b"local file"
    with zipfile.ZipFile(root / "nif-2.0.0-py3-none-any.whl", "w") as archive:
        for name, content in package.items():
            archive.writestr(name, content)
        archive.writestr(
            "nif-2.0.0.dist-info/METADATA",
            f"Name: NIF\nVersion: {wheel_version}\nRequires-Dist: torch>=2.6\n\n",
        )
    source = dict(package)
    source.update(
        {
            name: b"metadata"
            for name in ("CITATION.cff", ".zenodo.json", "LICENSE", "README.md")
        }
    )
    source["PKG-INFO"] = f"Name: NIF\nVersion: {source_version}\n\n".encode()
    with tarfile.open(root / "nif-2.0.0.tar.gz", "w:gz") as archive:
        for name, content in source.items():
            member = tarfile.TarInfo("nif-2.0.0/" + name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))


def test_distribution_checker_accepts_complete_release_inputs(tmp_path):
    write_distributions(tmp_path)
    check_distributions(tmp_path, "2.0.0")


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"wheel_version": "1.0.3"}, "Wheel version mismatch"),
        ({"source_version": "1.0.3"}, "Sdist version mismatch"),
        ({"omit": "nif/demo/dataset/cylinderflow.npz"}, "Missing distribution files"),
        ({"omit": "nif/optimizers/TFP_NOTICE.txt"}, "Missing distribution files"),
        ({"extra": "experiments/private.py"}, "local/generated files"),
        ({"extra": "dev_tests/artifacts/local.json"}, "local/generated files"),
    ],
)
def test_distribution_checker_rejects_inconsistent_or_local_inputs(
    tmp_path, changes, message
):
    write_distributions(tmp_path, **changes)
    with pytest.raises(ValueError, match=message):
        check_distributions(tmp_path, "2.0.0")
