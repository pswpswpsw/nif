"""Bounded v2 guards preserve all numeric/source expectations and required files."""
from pathlib import Path

import pytest
from dev_tests.publication_windows_scope import ROOT,assert_publication_windows_scope_v2
from dev_tests.adabelief_moment_scope import assert_adabelief_phase_scope_v4


def test_exact_windows_successors_with_gbk_default_reader(monkeypatch):
    original=Path.read_text
    def gbk_default(path,encoding=None,errors=None,**kwargs):
        return original(path,encoding=encoding or 'gbk',errors=errors,**kwargs)
    monkeypatch.setattr(Path,'read_text',gbk_default)
    assert len(assert_publication_windows_scope_v2())==8


@pytest.mark.parametrize('name,old,new',[
    ('dev_tests/test_tutorial1_gc_behavior.py','lr=5e-3','lr=4e-3'),
    ('dev_tests/test_pruning_adam_state.py','list(range(4000,200001,4000))','list(range(0,200001,4000))'),
    ('dev_tests/test_publication_scope.py',"assert result['filename']==name","assert result['filename'] is not None"),
    ('dev_tests/adabelief_moment_scope.py',"name.startswith('dev_tests/artifacts/')","name.startswith('dev_tests/')"),
])
def test_portability_authorization_rejects_unrelated_or_weakened_assertions(monkeypatch,name,old,new):
    original=Path.read_text
    def changed(path,*args,**kwargs):
        source=original(path,*args,**kwargs)
        if path==ROOT/name:
            assert old in source
            return source.replace(old,new)
        return source
    monkeypatch.setattr(Path,'read_text',changed)
    with pytest.raises(AssertionError,match=name):assert_publication_windows_scope_v2()


def test_missing_new_public_archive_is_not_optional(monkeypatch):
    original=Path.read_bytes
    target=ROOT/'dev_tests/fixtures/publication_windows_v2/test_tutorial1_gc_behavior.py.txt'
    def missing(path):
        if path==target:raise FileNotFoundError(2,'controlled missing public fixture',str(path))
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',missing)
    with pytest.raises(FileNotFoundError) as error:assert_adabelief_phase_scope_v4()
    assert Path(error.value.filename)==target
