"""Check that CI selects a Python supported by the package it validates."""
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ci_python_version_meets_package_minimum():
    workflow = (ROOT / ".github/workflows/run-tests.yml").read_text(encoding="utf-8")
    setup = (ROOT / "setup.py").read_text(encoding="utf-8")
    selected = re.search(r"python-version:\s*['\"]?(\d+)\.(\d+)", workflow)
    minimum = re.search(r"python_requires\s*=\s*['\"]>=(\d+)\.(\d+)", setup)
    assert selected and minimum, "CI Python and package minimum must both be explicit"
    ci_version = tuple(map(int, selected.groups()))
    package_minimum = tuple(map(int, minimum.groups()))
    assert (
        ci_version >= package_minimum
    ), f"CI Python {ci_version} is older than package minimum {package_minimum}"


def test_ci_avoids_duplicate_pr_push_and_cancels_superseded_runs():
    workflow = (ROOT / ".github/workflows/run-tests.yml").read_text(encoding="utf-8")
    assert re.search(r"(?m)^  push:\s*\n    branches: \[main\]", workflow)
    assert re.search(r"(?m)^  pull_request:", workflow)
    assert re.search(r"(?m)^concurrency:", workflow)
    assert "cancel-in-progress: true" in workflow


def test_ci_runs_current_regressions_and_checks_the_tutorial_catalog():
    workflow = (ROOT / ".github/workflows/run-tests.yml").read_text(encoding="utf-8")
    assert "python -m pytest dev_tests -q" in workflow
    assert "python scripts/tutorial_catalog.py --check" in workflow
    assert "python -m pip install" in workflow
    assert "tensorboard" in workflow
    assert "CUDA_VISIBLE_DEVICES" in workflow
    assert "python dev_tests/run_native_notebooks.py --output" in workflow
    assert "NIF_DEVICE: cpu" in workflow
