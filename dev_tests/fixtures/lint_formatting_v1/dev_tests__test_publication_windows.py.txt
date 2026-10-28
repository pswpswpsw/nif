"""Linux-hosted Windows path/GBK semantics, frozen before target-test edits.

These are public portability controls, not a claim of native Windows execution.
They exercise actual test functions and the actual subprocess reporting script.
"""
import ast
from contextlib import redirect_stdout
import io
from pathlib import Path, PureWindowsPath
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
from dev_tests import test_pruning_adam_state as pruning
from dev_tests import test_tutorial1_gc_behavior as behavior
from dev_tests import test_publication_scope as publication


NAMES=['dev_tests/fixtures/adabelief_dtype/native_before.py.txt',
       'dev_tests/adabelief_dtype_red.log','dev_tests/adabelief_boundary_full.log']


def test_actual_fingerprint_guard_accepts_windows_relative_paths(monkeypatch):
    original=Path.relative_to
    def windows_relative(path,*args,**kwargs):
        result=original(path,*args,**kwargs)
        if path.name in ('3_multi_scale_linear_NIF_pytorch.ipynb','4_evaluate_derivatives_pytorch.ipynb') or path.name.startswith('4_'):
            return PureWindowsPath(result.as_posix())
        return result
    monkeypatch.setattr(Path,'relative_to',windows_relative)
    pruning.test_red_test_fingerprints_unchanged_and_scientific_calls_migrated_v2()


@pytest.mark.parametrize('explicit_gc',[False,True],ids=['primary_default','optional_gc'])
def test_generator_reader_is_utf8_for_both_consumers(monkeypatch,explicit_gc):
    expected=behavior.sources('generator')
    original=Path.read_text
    def gbk_default(path,encoding=None,errors=None,**kwargs):
        return original(path,encoding=encoding or 'gbk',errors=errors,**kwargs)
    monkeypatch.setattr(Path,'read_text',gbk_default)
    actual=behavior.sources('generator')
    assert actual==expected
    for source in actual:ast.parse(source)
    # Both consumer choices use this unchanged source reader; no model training.
    selector=next(node for node in ast.parse(actual[-1]).body if isinstance(node,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=='use_gradient_centralization' for t in node.targets))
    choice=True if explicit_gc else ast.literal_eval(selector.value)
    assert choice is explicit_gc


def windows_error_runner(monkeypatch,missing_name):
    """Execute scope_result's exact child script with Windows path semantics."""
    class WindowsView(PureWindowsPath):
        @classmethod
        def cwd(cls):return cls('C:/publication')
        def resolve(self):return self
        def exists(self):return False
    root=WindowsView.cwd()
    def run(args,**kwargs):
        scope=SimpleNamespace(__file__=str(root/'dev_tests/adabelief_moment_scope.py'))
        def reject():raise FileNotFoundError(2,'No such file or directory',str(root/missing_name))
        scope.assert_adabelief_phase_scope_v4=reject
        package=ModuleType('dev_tests');package.adabelief_moment_scope=scope
        nif=ModuleType('nif');nif.__file__=str(root/'nif/__init__.py')
        import pathlib
        output=io.StringIO();saved_path=sys.path[:]
        try:
            with monkeypatch.context() as patch:
                patch.setattr(pathlib,'Path',WindowsView)
                patch.setitem(sys.modules,'nif',nif)
                patch.setitem(sys.modules,'dev_tests',package)
                with redirect_stdout(output):exec(compile(args[3],'<actual scope_result child script>','exec'),{})
        finally:sys.path[:]=saved_path
        return subprocess.CompletedProcess(args,0,stdout=output.getvalue(),stderr='')
    monkeypatch.setattr(publication.subprocess,'run',run)


@pytest.mark.parametrize('name',NAMES)
def test_missing_file_negative_uses_exact_windows_filename_identity(monkeypatch,tmp_path,name):
    path=tmp_path/name;path.parent.mkdir(parents=True);path.write_bytes(b'controlled public fixture')
    windows_error_runner(monkeypatch,name)
    publication.test_missing_required_public_fixture_or_provenance_fails(tmp_path,name)
    assert path.read_bytes()==b'controlled public fixture'


@pytest.mark.parametrize('name',NAMES)
def test_different_windows_filename_is_never_accepted(monkeypatch,tmp_path,name):
    path=tmp_path/name;path.parent.mkdir(parents=True);path.write_bytes(b'controlled public fixture')
    windows_error_runner(monkeypatch,'unrelated/'+name)
    with pytest.raises(AssertionError):
        publication.test_missing_required_public_fixture_or_provenance_fails(tmp_path,name)
    assert path.read_bytes()==b'controlled public fixture'
