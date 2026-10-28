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
    assert ci_version >= package_minimum, (
        f"CI Python {ci_version} is older than package minimum {package_minimum}"
    )
