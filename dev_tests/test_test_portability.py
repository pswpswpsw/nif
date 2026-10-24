"""Exercise development checks under non-default text/build environments."""
from pathlib import Path
import platform

import pytest

import test_public_api_regressions as regressions
from test_tutorial1_lbfgs import solver_cells


def test_generator_source_read_uses_utf8_under_gbk_default(monkeypatch):
    original = Path.read_text

    def locale_read(path, encoding=None, errors=None, **kwargs):
        return original(path, encoding=encoding or 'gbk', errors=errors, **kwargs)

    monkeypatch.setattr(Path, 'read_text', locale_read)
    setup, training = solver_cells('generator')
    assert 'rounds, max_iter =' in setup
    assert 'fine_tuner = ReferenceLBFGS(' in training


def test_linux_fingerprints_skip_on_windows_with_same_torch_version(monkeypatch):
    monkeypatch.setattr(platform, 'system', lambda: 'Windows')
    with pytest.raises(pytest.skip.Exception, match='recorded.*build'):
        regressions.test_existing_2d_rng_and_default_training_trajectories_unchanged()


@pytest.fixture
def recorded_environment():
    import json
    return json.loads(Path(__file__).with_name(
        'public_api_pre_fix_environment.json').read_text(encoding='utf-8'))['environment']


def test_exact_fingerprints_run_on_matching_capture_build(monkeypatch, recorded_environment):
    monkeypatch.setattr(regressions, 'fingerprint_environment', lambda: recorded_environment.copy())
    regressions.require_recorded_fingerprint_environment(recorded_environment)


@pytest.mark.parametrize('field', [
    'system', 'machine', 'byteorder', 'pointer_bits', 'python_implementation',
    'python_version', 'python_compiler', 'numpy_version', 'torch_version',
    'torch_git_version', 'torch_build_config_sha256',
])
def test_exact_fingerprints_skip_each_mismatched_build_field(monkeypatch, recorded_environment, field):
    current = {**recorded_environment, field: 'different-build-value'}
    monkeypatch.setattr(regressions, 'fingerprint_environment', lambda: current)
    with pytest.raises(pytest.skip.Exception, match=field):
        regressions.require_recorded_fingerprint_environment(recorded_environment)


def test_exact_fingerprints_require_complete_capture_metadata(monkeypatch, recorded_environment):
    monkeypatch.setattr(regressions, 'fingerprint_environment', lambda: recorded_environment.copy())
    incomplete = recorded_environment.copy()
    del incomplete['torch_build_config_sha256']
    with pytest.raises(pytest.skip.Exception, match='torch_build_config_sha256'):
        regressions.require_recorded_fingerprint_environment(incomplete)
