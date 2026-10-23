"""Real Git-for-Windows checkout regression; original scientific gates stay intact."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_FILES = (
    '.gitattributes',
    'dev_tests/publication_windows_scope.py',
    'dev_tests/adabelief_moment_scope.py',
    'dev_tests/test_publication_checkout_eol.py',
    'dev_tests/fixtures/publication_checkout_v3/publication_windows_scope.py.txt',
)
BYTE_EXACT = (
    'tutorial/README.md',
    'dev_tests/adabelief_moment_scope.diff',
    'dev_tests/publication_v1_scope.diff',
    'dev_tests/publication_windows_v2_scope.diff',
)


def git(*args, cwd=None, data=None):
    result = subprocess.run([shutil.which('git'), *args], cwd=cwd, input=data,
                            capture_output=True)
    assert result.returncode == 0, result.stderr.decode('utf-8', errors='replace')
    return result.stdout.decode('utf-8').strip()


def test_real_git_checkout_with_autocrlf_true_preserves_integrity(tmp_path):
    if not shutil.which('git') or not (ROOT/'.git').exists():
        pytest.skip('integration requires a Git source checkout and installed Git')
    checkout = tmp_path/'checkout'
    git('-c', 'core.autocrlf=true', 'clone', '--no-checkout', '--no-hardlinks',
        str(ROOT), str(checkout))
    git('config', 'core.autocrlf', 'true', cwd=checkout)
    git('read-tree', 'HEAD', cwd=checkout)
    # Stage just the explicit candidate without commits or source-repo writes.
    # checkout-index then performs Git's actual index-to-worktree conversion.
    for name in CANDIDATE_FILES:
        oid = git('hash-object', '-w', '--no-filters', '--stdin', cwd=checkout,
                  data=(ROOT/name).read_bytes())
        git('update-index', '--add', '--cacheinfo', '100644', oid, name, cwd=checkout)
    git('checkout-index', '--all', '--force', cwd=checkout)
    assert git('config', '--get', 'core.autocrlf', cwd=checkout) == 'true'
    differences = [name for name in BYTE_EXACT
                   if (checkout/name).read_bytes() != (ROOT/name).read_bytes()]
    result = subprocess.run([
        sys.executable, '-m', 'pytest', '-q',
        'dev_tests/test_adabelief_boundary_succession.py::test_scope_v4_exact_successor_and_utf8_under_gbk_default',
        'dev_tests/test_publication_scope.py::test_clean_public_scope_runs_without_git_or_generated_outputs',
        'dev_tests/test_tutorial1_gc_scope.py::test_other_notebooks_package_sources_and_frozen_expectations_phase_v4',
        '--basetemp', str(tmp_path/'scope-pytest'),
    ], cwd=checkout, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
       capture_output=True, text=True, encoding='utf-8')
    assert not differences and result.returncode == 0, (
        f'Git checkout byte deltas: {differences}\n{result.stdout}\n{result.stderr}')


def test_checkout_scope_archive_retains_prior_public_digest():
    archive = ROOT/CANDIDATE_FILES[-1]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == (
        '39d7780ffe31d570cc2aabefd2b7dfffc4cd3fac1310adfa68ad409831136b1e')
