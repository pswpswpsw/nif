"""Exact source view for historical guards after the lint-only cleanup.

The public manifest pins every permitted formatted source and its original
archive. Historical checks receive the original only after both byte hashes
pass. Runtime model imports continue to execute the current source.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = "dev_tests/fixtures/lint_formatting_v1/manifest.json"
RECORD_SHA256 = "2a07d432d8762ae547b641db51cdd895396277e036a21dd4932f0dd637ac572e"
SOURCE = "dev_tests/lint_preservation.py"
# BEGIN SEALED PATHS
LINT_PATHS = frozenset(
    {
        "dev_tests/adabelief_history.py",
        "dev_tests/adabelief_moment_history.py",
        "dev_tests/adabelief_moment_scope.py",
        "dev_tests/adabelief_phase_scope.py",
        "dev_tests/benchmark_reference_adam_cache.py",
        "dev_tests/diagnose_tutorial3_parity.py",
        "dev_tests/generate_native_notebooks.py",
        "dev_tests/lbfgs_reference_scope.py",
        "dev_tests/publication_projection.py",
        "dev_tests/publication_windows_scope.py",
        "dev_tests/run_adabelief_boundary_smoke.py",
        "dev_tests/run_adabelief_dtype_smoke.py",
        "dev_tests/run_adabelief_moment_smoke.py",
        "dev_tests/run_native_notebooks.py",
        "dev_tests/run_pruning_adam_smoke.py",
        "dev_tests/run_tutorial1_gc_smoke.py",
        "dev_tests/test_adabelief_boundary_succession.py",
        "dev_tests/test_adabelief_dtype_preservation.py",
        "dev_tests/test_adabelief_empty_gradients.py",
        "dev_tests/test_adabelief_final_update_order.py",
        "dev_tests/test_adabelief_final_update_successor.py",
        "dev_tests/test_adabelief_moment_checkpoint.py",
        "dev_tests/test_adabelief_moment_dtype.py",
        "dev_tests/test_adabelief_scalar_dtype.py",
        "dev_tests/test_ci_workflow.py",
        "dev_tests/test_layers.py",
        "dev_tests/test_lbfgs_reference_integration.py",
        "dev_tests/test_lbfgs_reference_phase0_records.py",
        "dev_tests/test_lbfgs_stopping_contract.py",
        "dev_tests/test_model_review.py",
        "dev_tests/test_models.py",
        "dev_tests/test_native_notebooks.py",
        "dev_tests/test_notebook_displays.py",
        "dev_tests/test_optimizer_epsilon.py",
        "dev_tests/test_optimizers_data.py",
        "dev_tests/test_pruning_adam_state.py",
        "dev_tests/test_public_api_regressions.py",
        "dev_tests/test_publication_checkout_eol.py",
        "dev_tests/test_publication_checkout_successor.py",
        "dev_tests/test_publication_scope.py",
        "dev_tests/test_publication_transition.py",
        "dev_tests/test_publication_windows.py",
        "dev_tests/test_publication_windows_successor.py",
        "dev_tests/test_reference_adam.py",
        "dev_tests/test_reference_adam_cache.py",
        "dev_tests/test_reference_adam_empty_gradients.py",
        "dev_tests/test_reference_lbfgs_api.py",
        "dev_tests/test_reference_lbfgs_contract.py",
        "dev_tests/test_reference_lbfgs_edges.py",
        "dev_tests/test_scope_helper_utf8.py",
        "dev_tests/test_scope_helper_utf8_archives.py",
        "dev_tests/test_teaching_markdown_scope.py",
        "dev_tests/test_test_portability.py",
        "dev_tests/test_tutorial1_gc_behavior.py",
        "dev_tests/test_tutorial1_gc_scope.py",
        "dev_tests/test_tutorial1_lbfgs.py",
        "dev_tests/test_tutorial7_inference_artifacts.py",
        "dev_tests/test_tutorial7_pruning.py",
        "dev_tests/test_tutorial7_reference_outputs.py",
        "dev_tests/test_tutorial7_reference_scope.py",
        "dev_tests/tutorial1_gc_scope.py",
        "dev_tests/tutorial7_reference_scope.py",
        "nif/__init__.py",
        "nif/data/point_wise_data.py",
        "nif/data/tfr_dataset.py",
        "nif/layers/__init__.py",
        "nif/layers/_common.py",
        "nif/layers/gradient.py",
        "nif/layers/mlp.py",
        "nif/layers/regularization.py",
        "nif/layers/siren.py",
        "nif/model.py",
        "nif/optimizers/__init__.py",
        "nif/optimizers/_hager_zhang.py",
        "nif/optimizers/external_optimizers.py",
        "nif/optimizers/gtcf.py",
        "nif/optimizers/lbfgs.py",
        "nif/optimizers/lbfgs_V2.py",
        "nif/optimizers/reference_adam.py",
        "nif/optimizers/reference_lbfgs.py",
        "nif/pruning.py",
        "tests/test_tutorial_7.py",
    }
)
# END SEALED PATHS


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def lint_record(root=ROOT):
    """Validate the manifest and the reader's own exact source template."""
    root = Path(root)
    raw = (root / RECORD).read_bytes()
    assert _digest(raw) == RECORD_SHA256, RECORD
    record = json.loads(raw)
    assert set(record["transitions"]) == LINT_PATHS, RECORD
    source = (root / SOURCE).read_bytes()
    # Sources use LF in Git; accept only its exact CRLF checkout equivalent.
    assert b"\r" not in source.replace(b"\r\n", b""), SOURCE
    if b"\r\n" in source:
        assert b"\n" not in source.replace(b"\r\n", b""), SOURCE
    source = source.replace(b"\r\n", b"\n")
    slot = ('RECORD_SHA256 = "' + RECORD_SHA256 + '"').encode("utf-8")
    assert source.count(slot) == 1, SOURCE
    template = source.replace(slot, b'RECORD_SHA256 = "<lint-record-sha256>"', 1)
    assert _digest(template) == record["reader_template_sha256"], SOURCE
    return record


def historical_lint_bytes(root, name, current):
    """Undo one exact recorded cleanup; arbitrary source changes fail closed."""
    if name not in LINT_PATHS:
        return current
    entry = lint_record(root)["transitions"][name]
    assert _digest(current) in entry["after_sha256_by_eol"].values(), name
    original = (Path(root) / entry["archive"]).read_bytes()
    assert _digest(original) == entry["before_sha256"], entry["archive"]
    return original


def read_lint_bytes(root, name):
    return historical_lint_bytes(root, name, (Path(root) / name).read_bytes())


def read_lint_text(root, name):
    # Keep the original text-read boundary for historical mutation tests.
    current = (Path(root) / name).read_text(encoding="utf-8").encode("utf-8")
    if name not in LINT_PATHS:
        return current.decode("utf-8")
    raw = (Path(root) / name).read_bytes()
    original = historical_lint_bytes(root, name, raw)
    assert current == raw.replace(b"\r\n", b"\n"), name
    return original.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


def lint_public_files(root=ROOT):
    """Explicit candidate and provenance inputs for Git-free projections."""
    record = lint_record(root)
    expected = [SOURCE, RECORD]
    for name, entry in sorted(record["transitions"].items()):
        expected.extend((name, entry["archive"]))
    assert record["public_files"] == expected, RECORD
    assert len(expected) == len(set(expected)), RECORD
    for name in expected:
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts, name
        assert ".git" not in relative.parts and "artifacts" not in relative.parts, name
        (Path(root) / name).stat()
        assert (Path(root) / name).is_file(), name
    return expected
