"""The tutorial catalog is the single source for public listing and test scope."""
import copy
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def helper():
    path = ROOT / "scripts/tutorial_catalog.py"
    assert path.is_file(), "A shared tutorial catalog renderer is required"
    spec = importlib.util.spec_from_file_location("nif_tutorial_catalog", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_catalog_separates_released_and_experimental_workloads():
    catalog = helper().load_catalog(ROOT)
    released = helper().tutorial_entries(ROOT, status="released")
    experimental = helper().tutorial_entries(ROOT, status="experimental")
    assert [entry["id"] for entry in released] == list(range(1, 9))
    assert [entry["id"] for entry in experimental] == [9]
    listed = {entry["notebook"] for entry in catalog["tutorials"]}
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tutorial").glob("*_pytorch.ipynb")
    }
    assert listed == actual
    for entry in released:
        assert helper().source_ref(catalog, entry, "docs") == "v2.0.0"
        assert helper().source_ref(catalog, entry, "readme") == "main"
    assert helper().source_ref(catalog, experimental[0], "docs") == "main"


def test_catalog_outputs_are_current_without_rewriting_source_files():
    assert helper().update_outputs(ROOT, check=True) == []


@pytest.mark.parametrize("mutation", ["duplicate-id", "traversal", "missing", "status"])
def test_invalid_catalog_entries_fail_before_rendering(mutation):
    module = helper()
    catalog = copy.deepcopy(module.load_catalog(ROOT))
    if mutation == "duplicate-id":
        catalog["tutorials"][1]["id"] = catalog["tutorials"][0]["id"]
    elif mutation == "traversal":
        catalog["tutorials"][0]["notebook"] = "tutorial/../../outside.ipynb"
    elif mutation == "missing":
        catalog["tutorials"][0]["notebook"] = "tutorial/missing_pytorch.ipynb"
    else:
        catalog["tutorials"][0]["status"] = "unknown"
    with pytest.raises(ValueError):
        module.validate_catalog(catalog, ROOT)


def test_readme_update_preserves_everything_outside_tutorial_section():
    module = helper()
    before = (ROOT / "README.md").read_text(encoding="utf-8")
    updated = module.replace_readme_section(before, "replacement tutorial listing\n")
    before_prefix, before_suffix = module.readme_outer_sections(before)
    after_prefix, after_suffix = module.readme_outer_sections(updated)
    assert (after_prefix, after_suffix) == (before_prefix, before_suffix)
    assert before.count("```") == updated.count("```")
    assert "./misc/myimage.gif" in updated and "./misc/compare.gif" in updated


def test_check_cli_reports_stale_listings_without_writing(tmp_path):
    module = helper()
    catalog = module.load_catalog(ROOT)
    (tmp_path / "tutorial").mkdir()
    (tmp_path / "tutorial/catalog.json").write_bytes(
        (ROOT / "tutorial/catalog.json").read_bytes()
    )
    for entry in catalog["tutorials"]:
        for key in ("notebook", "original", "compatibility"):
            if key in entry:
                (tmp_path / entry[key]).touch()
    readme = (
        "# Existing README\n\n## Google Colab Tutorial\n\nstale\n\n"
        "## Issues, bugs, requests, ideas\n\nKeep this footer.\n"
    )
    (tmp_path / "README.md").write_text(readme, encoding="utf-8")
    assert set(module.update_outputs(tmp_path, check=True)) == {
        "README.md",
        "docs/source/tutorials.rst",
        "tutorial/README.md",
    }
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == readme
    module.update_outputs(tmp_path, check=False)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/tutorial_catalog.py"),
            "--root",
            str(tmp_path),
            "--check",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    generated = tmp_path / "docs/source/tutorials.rst"
    generated.write_text(
        generated.read_text(encoding="utf-8") + "\nDrift\n", encoding="utf-8"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/tutorial_catalog.py"),
            "--root",
            str(tmp_path),
            "--check",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 1
    assert "docs/source/tutorials.rst" in result.stdout
    assert generated.read_text(encoding="utf-8").endswith("\nDrift\n")
