"""Authorized scope transition; no numerical expectations are changed."""
import hashlib
from dev_tests.lbfgs_reference_scope import historical_bytes
import json
from pathlib import Path
from dev_tests.tutorial1_gc_scope import assert_tutorial1_gc_scope_v2

ROOT=Path(__file__).resolve().parents[1]


def test_tutorial1_gc_only_changes_call_selector_and_teaching_note():
    assert_tutorial1_gc_scope_v2()


def test_other_notebooks_package_sources_and_frozen_expectations_phase_v4():
    previous=json.loads((ROOT/'dev_tests/tutorial1_gc_pre_fix_fingerprints.json').read_text(encoding='utf-8'))
    authorized={'dev_tests/test_pruning_adam_state.py','dev_tests/generate_native_notebooks.py',
                'tutorial/1_simple_1d_wave_pytorch.ipynb','docs/PYTORCH_CANDIDATE_REPORT.md',
                'tutorial/5_large_scale_training_on_tensorflow_record_data_pytorch.ipynb',
                'tutorial/8_revisit_multi_scale_NIF_with_sobolov_training_pytorch.ipynb'}
    # Coordinator documentation may evolve independently; our implementation scope
    # check covers all notebook/package/data/reference and prior public test files.
    # V4 adds only dtype moments/historical replay; numerical goldens stay fixed.
    from dev_tests.adabelief_moment_scope import assert_adabelief_phase_scope_v4
    superseded = assert_adabelief_phase_scope_v4()
    for name,digest in previous.items():
        if name in superseded:
            continue  # Already checked against the exact approved transformation.
        if name not in authorized and name.startswith(('tutorial/','nif/','dev_tests/','tests/')):
            assert hashlib.sha256(historical_bytes(ROOT, name, (ROOT/name).read_bytes())).hexdigest()==digest,name


def test_phase_guard_v2_changes_only_authorized_historical_scope_assertion():
    old=(ROOT/'dev_tests/fixtures/tutorial1_gc_scope_v1/test_pruning_adam_state.py.txt').read_text(encoding='utf-8')
    new=(ROOT/'dev_tests/test_pruning_adam_state.py').read_text(encoding='utf-8')
    old=old.replace('def test_red_test_fingerprints_unchanged_and_scientific_calls_migrated():',
                    'def test_red_test_fingerprints_unchanged_and_scientific_calls_migrated_v2():')
    old=old.replace('    for number in (1,3,4):',
                    '    # Scope v2: tutorial 1 has an authorized, exactly bounded GC default correction.\n'
                    '    from dev_tests.tutorial1_gc_scope import assert_tutorial1_gc_scope_v2\n'
                    '    assert_tutorial1_gc_scope_v2()\n'
                    '    for number in (3,4):')
    # Publication Windows v2: only portable lookup of immutable POSIX keys.
    old=old.replace('str(path.relative_to(root))','path.relative_to(root).as_posix()')
    assert new==old, 'All historical numerical assertions must remain unchanged'
