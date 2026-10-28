"""Exercise the scope test's own source reads with UTF-8 temporary copies."""
from pathlib import Path

from dev_tests import test_tutorial1_gc_scope as scope_tests


def test_scope_test_source_reads_are_utf8_under_gbk_default(monkeypatch,tmp_path):
    # Add the same harmless Unicode comment to temporary source copies. This
    # preserves the exact phase transform while testing decoding independently
    # of whether today's particular test source happens to be all ASCII.
    files=['dev_tests/fixtures/tutorial1_gc_scope_v1/test_pruning_adam_state.py.txt',
           'dev_tests/test_pruning_adam_state.py']
    for name in files:
        target=tmp_path/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((scope_tests.ROOT/name).read_bytes()+'\n# Preserved UTF-8 scope – reader control\n'.encode('utf-8'))
    monkeypatch.setattr(scope_tests,'ROOT',tmp_path)
    original=Path.read_text
    def gbk_default(path,encoding=None,errors=None,**kwargs):
        return original(path,encoding=encoding or 'gbk',errors=errors,**kwargs)
    monkeypatch.setattr(Path,'read_text',gbk_default)
    scope_tests.test_phase_guard_v2_changes_only_authorized_historical_scope_assertion()
