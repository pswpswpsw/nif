"""Publication Windows v2: exact nonnumerical test-source succession.

All original archives/manifests remain immutable. These transforms permit only
POSIX fingerprint lookup, UTF-8 reading, structured missing-file identity and
the necessary scope/projection wiring. No math assertion is changed.
"""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE='dev_tests/fixtures/publication_windows_v2'
TRANSITIONS={
 'dev_tests/test_pruning_adam_state.py':[
  ('fingerprints[str(path.relative_to(root))]',
   'fingerprints[path.relative_to(root).as_posix()]')],
 'dev_tests/test_tutorial1_gc_behavior.py':[
  ("(ROOT/'dev_tests/generate_native_notebooks.py').read_text()",
   "(ROOT/'dev_tests/generate_native_notebooks.py').read_text(encoding='utf-8')")],
 'dev_tests/test_publication_scope.py':[
  ("    print(json.dumps({'error':type(exc).__name__,'detail':str(exc)}))",
   "    filename=getattr(exc,'filename',None)\n"
   "    if filename is not None:\n"
   "        filename=pathlib.Path(filename).relative_to(root).as_posix()\n"
   "    print(json.dumps({'error':type(exc).__name__,'detail':str(exc),'filename':filename}))"),
  ("        assert name in result['detail'],result",
   "        assert result['filename']==name,result")],
 'dev_tests/test_tutorial1_gc_scope.py':[
  ("    assert new==old, 'All historical numerical assertions must remain unchanged'",
   "    # Publication Windows v2: only portable lookup of immutable POSIX keys.\n"
   "    old=old.replace('str(path.relative_to(root))','path.relative_to(root).as_posix()')\n"
   "    assert new==old, 'All historical numerical assertions must remain unchanged'")],
 'dev_tests/test_adabelief_boundary_succession.py':[
  ("        'dev_tests/test_adabelief_boundary_succession.py'}",
   "        'dev_tests/test_adabelief_boundary_succession.py',\n"
   "        'dev_tests/test_pruning_adam_state.py','dev_tests/test_tutorial1_gc_behavior.py',\n"
   "        'dev_tests/test_publication_scope.py','dev_tests/test_publication_transition.py',\n"
   "        'dev_tests/adabelief_moment_scope.py','dev_tests/publication_projection.py'}")],
 'dev_tests/test_publication_transition.py':[
  ('def test_publication_v1_is_only_exact_namespace_and_documentation_change():',
   'def test_publication_v1_history_and_windows_v2_exact_successor():'),
  ("    assert (ROOT/SOURCE).read_text(encoding='utf-8')==expected_publication_source(archive.decode('utf-8'))",
   "    # V1 remains checked verbatim; v2 checks the exact current successor.\n"
   "    historical=ROOT/'dev_tests/fixtures/publication_windows_v2/adabelief_moment_scope.py.txt'\n"
   "    assert historical.read_text(encoding='utf-8')==expected_publication_source(archive.decode('utf-8'))\n"
   "    from dev_tests.publication_windows_scope import assert_publication_windows_scope_v2\n"
   "    assert_publication_windows_scope_v2(ROOT)")],
 'dev_tests/adabelief_moment_scope.py':[
  ('public source, fixtures and provenance remain mandatory.\n',
   'public source, fixtures and provenance remain mandatory.\n'
   'Publication Windows v2 additionally verifies exact nonnumerical test-reader\n'
   'and scope successors before applying the unchanged scientific scope checks.\n'),
  ('def assert_adabelief_phase_scope_v4():\n',
   'def assert_adabelief_phase_scope_v4():\n'
   '    from dev_tests.publication_windows_scope import assert_publication_windows_scope_v2\n'
   '    portable = assert_publication_windows_scope_v2(ROOT)\n'),
  ('    for name in TRANSITIONS:\n',
   '    for name in TRANSITIONS:\n'
   '        if name in portable:\n'
   '            continue  # Already checked against the exact archived successor.\n'),
  ("        if name not in TRANSITIONS and (name.startswith(('nif/','tutorial/','dev_tests/','tests/')) or name=='.gitattributes'):",
   "        if name not in TRANSITIONS and name not in portable and (name.startswith(('nif/','tutorial/','dev_tests/','tests/')) or name=='.gitattributes'):"),
  ('    return set(TRANSITIONS)|set(phase_v3)',
   '    return set(TRANSITIONS)|set(phase_v3)|portable')],
 'dev_tests/publication_projection.py':[
  ("source/'dev_tests/publication_v1_inventory.json'",
   "source/'dev_tests/publication_windows_v2_inventory.json'")]}


def expected_source(name,source):
    for old,new in TRANSITIONS[name]:
        assert source.count(old)==1,(name,old)
        source=source.replace(old,new,1)
    if name=='dev_tests/adabelief_moment_scope.py':
        source=expected_checkout_source(source)
    return source


def assert_publication_windows_scope_v2(root=ROOT):
    root=Path(root)
    archive=json.loads((root/ARCHIVE/'manifest.json').read_text(encoding='utf-8'))
    phase1=json.loads((root/'dev_tests/publication_v1_manifest.json').read_text(encoding='utf-8'))
    before1=json.loads((root/'dev_tests/publication_v1_before.json').read_text(encoding='utf-8'))
    for name in TRANSITIONS:
        entry=archive[name]
        original=(root/entry['archive']).read_bytes()
        assert hashlib.sha256(original).hexdigest()==entry['sha256'],name
        # Tie each prior source to unchanged v1 provenance, not refreshed hashes.
        original_hash=phase1['public_file_sha256'][name] if name in phase1['public_file_sha256'] else before1[name]
        assert entry['sha256']==original_hash,name
        assert (root/name).read_text(encoding='utf-8')==expected_source(name,original.decode('utf-8')),name
    red=json.loads((root/'dev_tests/publication_windows_v2_red_fingerprints.json').read_text(encoding='utf-8'))
    for name,digest in red.items():
        if name in TRANSITIONS:
            assert archive[name]['sha256']==digest,name
        else:assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    return set(TRANSITIONS)


def expected_checkout_source(source):
    """Checkout v3 permits only the exact attributes guard succession."""
    old='            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name\n'
    new=("            if name=='.gitattributes':\n"
         '                from dev_tests.publication_windows_scope import assert_checkout_attributes\n'
         '                assert_checkout_attributes(ROOT,digest)\n'
         '            else:\n'
         '                assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name\n')
    assert source.count(old)==1
    return source.replace(old,new,1)


def assert_checkout_attributes(root,digest):
    """Keep the historical bytes/hash; allow only four exact checkout rules."""
    suffix=(b'/tutorial/README.md text eol=lf\n'
            b'/dev_tests/adabelief_moment_scope.diff -text -whitespace\n'
            b'/dev_tests/publication_v1_scope.diff -text -whitespace\n'
            b'/dev_tests/publication_windows_v2_scope.diff -text -whitespace\n')
    current=(Path(root)/'.gitattributes').read_bytes()
    assert current.endswith(suffix),'.gitattributes'
    assert hashlib.sha256(current[:-len(suffix)]).hexdigest()==digest,'.gitattributes'
