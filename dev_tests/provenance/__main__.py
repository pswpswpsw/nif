"""Run the old integrity guards against their frozen public source tree."""
import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from . import Snapshot


HISTORICAL_FILES = (
    "test_adabelief_boundary_succession.py",
    "test_adabelief_dtype_preservation.py",
    "test_adabelief_final_update_successor.py",
    "test_lbfgs_reference_phase0_records.py",
    "test_lint_preservation.py",
    "test_publication_checkout_eol.py",
    "test_publication_checkout_successor.py",
    "test_publication_scope.py",
    "test_publication_transition.py",
    "test_publication_windows_successor.py",
    "test_release_version_scope.py",
    "test_scope_helper_utf8.py",
    "test_scope_helper_utf8_archives.py",
    "test_teaching_markdown_scope.py",
    "test_tutorial1_gc_scope.py",
    "test_tutorial7_reference_scope.py",
)
HISTORICAL_NODES = (
    "test_pruning_adam_state.py::test_red_test_fingerprints_unchanged_and_scientific_calls_migrated_v2",
    "test_lbfgs_reference_integration.py::test_exact_reference_scope_successor_exists",
    "test_lbfgs_reference_integration.py::test_original_strong_wolfe_regression_replays_all_assertions",
    "test_lbfgs_reference_integration.py::test_frozen_stopping_and_phase0_inputs_are_unchanged",
    "test_lbfgs_reference_integration.py::test_reference_scope_runs_in_public_projection",
    "test_lbfgs_reference_integration.py::test_reference_scope_rejects_other_notebook_changes",
    "test_lbfgs_reference_integration.py::test_reference_scope_rejects_source_record_and_archive_tampering",
    "test_lbfgs_reference_integration.py::test_new_required_archive_missing_fails",
    "test_lbfgs_reference_integration.py::test_three_exact_checkout_rules_and_two_license_includes_only",
    (
        "test_lbfgs_reference_integration.py::"
        "test_original_portability_constructor_expectation_replays_under_gbk"
    ),
    "test_publication_windows.py::test_actual_fingerprint_guard_accepts_windows_relative_paths",
    "test_publication_windows.py::test_missing_file_negative_uses_exact_windows_filename_identity",
    "test_publication_windows.py::test_different_windows_filename_is_never_accepted",
)


def audit(output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Audit output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    saved = Snapshot()
    started = time.perf_counter()
    view = saved.materialize(output / "frozen-public-tree")
    config = output / "pytest.ini"
    config.write_text("[pytest]\n", encoding="utf-8")
    selectors = ["dev_tests/" + name for name in (*HISTORICAL_FILES, *HISTORICAL_NODES)]
    script = (
        "import pathlib,sys; "
        "root=pathlib.Path.cwd().resolve(); sys.path.insert(0,str(root)); "
        "import nif,pytest; "
        "assert pathlib.Path(nif.__file__).resolve().is_relative_to(root); "
        "assert not (root/'.git').exists(); "
        "sys.exit(pytest.main(sys.argv[1:]))"
    )
    command = [
        sys.executable,
        "-I",
        "-c",
        script,
        "-c",
        str(config),
        "--confcutdir",
        str(view),
        "--basetemp",
        str(output / "t"),
        "-q",
        "-p",
        "no:cacheprovider",
        "--tb=short",
        "--junitxml",
        str(output / "junit.xml"),
        *selectors,
    ]
    environment = dict(
        os.environ,
        PYTHONUTF8="1",
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
    )
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        command,
        cwd=view,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=900,
    )
    (output / "pytest.stdout.log").write_text(result.stdout, encoding="utf-8")
    (output / "pytest.stderr.log").write_text(result.stderr, encoding="utf-8")
    junit = output / "junit.xml"
    suites = ET.parse(junit).getroot().findall("testsuite") if junit.is_file() else []
    report = {
        "result": "PASS" if result.returncode == 0 else "FAIL",
        "base_commit": saved.manifest["base_commit"],
        "bundle_sha256": saved.manifest["bundle_sha256"],
        "projected_files": len(saved.manifest["files"]),
        "historical_test_selectors": selectors,
        "tests": sum(int(suite.get("tests", 0)) for suite in suites),
        "failures": sum(int(suite.get("failures", 0)) for suite in suites),
        "errors": sum(int(suite.get("errors", 0)) for suite in suites),
        "skipped": sum(int(suite.get("skipped", 0)) for suite in suites),
        "elapsed_seconds": time.perf_counter() - started,
        "scope": (
            "Unchanged historical integrity tests in a frozen no-Git public tree; "
            "current implementation is tested separately"
        ),
        "git_checkout_tests": "Original real-checkout-only tests retain their no-Git skip condition",
        "platform_rng_tests": "Legacy RNG fingerprints retain the exact recorded platform/build requirement",
        "skip_details": [
            {"test": case.get("name"), "reason": skipped.get("message")}
            for suite in suites
            for case in suite.findall("testcase")
            for skipped in case.findall("skipped")
        ],
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(result.stdout[-18000:])
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    print(json.dumps(report, indent=2))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    runner = subparsers.add_parser("audit")
    runner.add_argument("--output", required=True, type=Path)
    extractor = subparsers.add_parser("extract")
    extractor.add_argument("--output", required=True, type=Path)
    extractor.add_argument("paths", nargs="*")
    args = parser.parse_args()
    if args.command == "extract":
        target = Snapshot().materialize(args.output, names=args.paths or None)
        print("Historical files extracted to:", target)
        return 0
    return audit(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
