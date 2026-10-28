"""Publication v1: real clean projection, mandatory preservation negatives.

These tests were frozen and failed before the scope helper's publication fix.
No scientific expectations or historical hash manifests are rewritten.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from dev_tests.publication_projection import project_public_files,audit_optional_local_history

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def public_root(tmp_path_factory):
    return project_public_files(ROOT,tmp_path_factory.mktemp('public_projection'))


def scope_result(root):
    script='''
import json, pathlib, sys
root=pathlib.Path.cwd().resolve()
sys.path.insert(0,str(root))
import nif
from dev_tests import adabelief_moment_scope as scope
assert pathlib.Path(nif.__file__).resolve().is_relative_to(root)
assert pathlib.Path(scope.__file__).resolve().is_relative_to(root)
assert not (root/'.git').exists()
assert not (root/'dev_tests/artifacts').exists()
try:
    result=scope.assert_adabelief_phase_scope_v4()
except Exception as exc:
    filename=getattr(exc,'filename',None)
    if filename is not None:
        filename=pathlib.Path(filename).relative_to(root).as_posix()
    print(json.dumps({'error':type(exc).__name__,'detail':str(exc),'filename':filename}))
else:
    print(json.dumps({'passed':True,'scope':sorted(result),'nif_origin':nif.__file__,'scope_origin':scope.__file__}))
'''
    # Absolute Python + isolated mode excludes inherited PYTHONPATH. Empty PATH
    # makes Git unavailable to runtime checks, even if the host installed it.
    proc=subprocess.run([sys.executable,'-I','-c',script],cwd=root,
        env={**os.environ,'PATH':'','CUDA_VISIBLE_DEVICES':''},capture_output=True,text=True,encoding='utf-8')
    assert proc.returncode==0,proc.stdout+proc.stderr
    return json.loads(proc.stdout.strip())


def test_clean_public_scope_runs_without_git_or_generated_outputs(public_root):
    result=scope_result(public_root)
    assert result.get('passed'),result


@pytest.mark.parametrize('name',[
    'dev_tests/fixtures/adabelief_dtype/native_before.py.txt',
    'dev_tests/adabelief_dtype_red.log',
    'dev_tests/adabelief_boundary_full.log',
])
def test_missing_required_public_fixture_or_provenance_fails(public_root,name):
    path=public_root/name;before=path.read_bytes()
    try:
        path.unlink()
        result=scope_result(public_root)
        assert result.get('error')=='FileNotFoundError',result
        assert result['filename']==name,result
    finally:path.write_bytes(before)


@pytest.mark.parametrize('name',[
    'nif/model.py',
    'dev_tests/public_api_pre_fix_fingerprints.json',
    'dev_tests/fixtures/adabelief_dtype/native_before.py.txt',
    'dev_tests/fixtures/adabelief_moment_v4/external_optimizers.py.txt',
])
def test_changed_public_source_golden_or_archived_fixture_fails(public_root,name):
    path=public_root/name;before=path.read_bytes()
    try:
        path.write_bytes(before+b'\n')
        result=scope_result(public_root)
        assert result.get('error')=='AssertionError',result
        # Archive errors may identify the original public source, not its copy.
        expected='nif/optimizers/external_optimizers.py' if name.endswith('adabelief_moment_v4/external_optimizers.py.txt') else name
        assert expected in result['detail'],result
    finally:path.write_bytes(before)


def test_optional_local_artifact_audit_reports_all_absent_in_projection(public_root):
    result=audit_optional_local_history(public_root)
    assert result['present']==[]
    assert len(result['missing'])==439
    assert all(n.startswith('dev_tests/artifacts/') for n in result['missing'])
    assert 'dev_tests/artifacts/adabelief_boundary/public_suite.junit.xml' in result['missing']


def test_optional_audit_rejects_present_changed_historical_artifact(tmp_path):
    manifest='dev_tests/adabelief_moment_pre_fix_fingerprints.json'
    (tmp_path/'dev_tests').mkdir()
    (tmp_path/manifest).write_bytes((ROOT/manifest).read_bytes())
    name='dev_tests/artifacts/adabelief_boundary/public_suite.junit.xml'
    path=tmp_path/name;path.parent.mkdir(parents=True);path.write_bytes(b'changed historical output')
    with pytest.raises(AssertionError,match=name):audit_optional_local_history(tmp_path)
