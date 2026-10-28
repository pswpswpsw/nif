"""Controlled GBK-default reader simulation for the scoped public helpers."""
from pathlib import Path

import pytest
from dev_tests import tutorial1_gc_scope as helper
from dev_tests import test_tutorial1_gc_scope as scope_tests


def test_representative_utf8_archive_rejects_gbk_decode():
    source=(helper.ARCHIVE/'generator_before.py.txt').read_bytes()
    assert b'\xe2\x80\x93' in source  # the preserved en dash
    with pytest.raises(UnicodeDecodeError):source.decode('gbk')


@pytest.mark.parametrize('target',['helper','scope_test'])
def test_new_scope_readers_work_with_gbk_default(monkeypatch,target):
    original=Path.read_text
    def gbk_default(path,encoding=None,errors=None,**kwargs):
        return original(path,encoding=encoding or 'gbk',errors=errors,**kwargs)
    monkeypatch.setattr(Path,'read_text',gbk_default)
    if target=='helper':helper.assert_tutorial1_gc_scope_v2()
    else:scope_tests.test_phase_guard_v2_changes_only_authorized_historical_scope_assertion()
