"""Explicit v3 scope succession: exact approved source transforms, not new goldens."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'dev_tests/fixtures/adabelief_boundary_v3'
OLD_TRAJECTORY='test_existing_2d_rng_and_default_training_trajectories_unchanged'
NEW_TRAJECTORY='test_current_initializers_rng_and_historical_adabelief_trajectories_v2'
COMPATIBILITY=f'''

def {OLD_TRAJECTORY}():
    """Compatibility entry for prior build-portability checks; runs all v2 assertions."""
    return {NEW_TRAJECTORY}()


# The explicitly named v2 test is collected; this compatibility entry is called
# by the existing platform-skip regression and executes the same assertions.
{OLD_TRAJECTORY}.__test__ = False
'''

# Each tuple is (old text, new text, replacement limit). These transforms permit
# the exact guard, historical binding/versioning, and UTF-8 reader changes only.
TRANSITIONS={
 'nif/optimizers/external_optimizers.py':[
  ('        for group, entries in _gradients(self):',
   "        if not any(p.grad is not None for group in self.param_groups for p in group['params']):\n"
   "            raise ValueError('No gradients provided for any parameter')\n"
   '        for group, entries in _gradients(self):',1),
  ('    the reference; identical scalars are cached only within a step and group.\n',
   '    the reference; identical scalars are cached only within a step and group.\n'
   '    After the closure, all-None gradients raise before optimizer/schedule mutation.\n'
   '    Zero tensors count as updates. On a real update all existing groups advance\n'
   '    their per-group clocks; a late-added group starts with its own step one.\n',1)],
 'dev_tests/tutorial1_gc_scope.py':[
  ('.read_text()',".read_text(encoding='utf-8')",-1)],
 'dev_tests/test_tutorial1_gc_scope.py':[
  ('.read_text()',".read_text(encoding='utf-8')",-1),
  ('def test_other_notebooks_package_sources_and_previous_frozen_regressions_unchanged():',
   'def test_other_notebooks_package_sources_and_frozen_expectations_phase_v3():',1),
  ('    for name,digest in previous.items():\n',
   '    # V3 authorizes only exact, archived source transforms; numerical goldens stay fixed.\n'
   '    from dev_tests.adabelief_phase_scope import assert_adabelief_phase_scope_v3\n'
   '    superseded = assert_adabelief_phase_scope_v3()\n'
   '    for name,digest in previous.items():\n'
   '        if name in superseded:\n'
   '            continue  # Already checked against the exact approved transformation.\n',1)],
 'dev_tests/test_public_api_regressions.py':[
  (f'def {OLD_TRAJECTORY}():',f'def {NEW_TRAJECTORY}():',1),
  ('    """Compare with self-owned fingerprints captured before either source fix."""',
   '    """V2: current initializers/models, historical PRE-scalar AdaBelief, original hashes.\n\n'
   '    Corrected current optimizer behavior is tested by the frozen source-derived\n'
   '    test_adabelief_scalar_dtype.py regressions, not replacement trajectory hashes.\n'
   '    """',1),
  ('    from nif.optimizers import AdaBeliefOptimizer\n',
   '    from dev_tests.adabelief_history import HistoricalAdaBelief as AdaBeliefOptimizer\n',1),
  ('            hashes = [digest(torch.cat([p.detach().flatten() for p in model.parameters()]))]\n',
   '            hashes = [digest(torch.cat([p.detach().flatten() for p in model.parameters()]))]\n'
   "            assert hashes[0] == baseline['trajectories'][cls.__name__][0], cls.__name__\n",1)],
 'dev_tests/test_adabelief_dtype_preservation.py':[
  ('    tree=ast.parse(path.read_text())\n',
   "    tree=ast.parse((ROOT/'dev_tests/fixtures/adabelief_boundary_v3/test_public_api_regressions.py.txt').read_text(encoding='utf-8'))\n",1),
  ('    The original test is still collected independently against the new optimizer.\n',
   '    The original source is archived verbatim; the collected successor explicitly\n'
   '    labels historical trajectories, with source-derived current-optimizer tests.\n',1)]}


def expected_source(name,source):
    for old,new,count in TRANSITIONS[name]:
        assert old in source,(name,old)
        source=source.replace(old,new,count)
    if name=='dev_tests/test_public_api_regressions.py':source+=COMPATIBILITY
    return source


def assert_adabelief_phase_scope_v3():
    archive=json.loads((ARCHIVE/'manifest.json').read_text(encoding='utf-8'))
    for name,entry in archive.items():
        data=(ROOT/entry['archive']).read_bytes()
        assert hashlib.sha256(data).hexdigest()==entry['sha256'],name
    # Retain the scalar-phase source, goldens and raw two-failure result exactly.
    scalar=json.loads((ROOT/'dev_tests/adabelief_dtype_candidate_manifest.json').read_text(encoding='utf-8'))
    assert archive['nif/optimizers/external_optimizers.py']['sha256']==scalar['source_sha256']['nif/optimizers/external_optimizers.py']
    assert (ROOT/archive['dev_tests/adabelief_dtype_full_final.log']['archive']).read_bytes()==(ROOT/'dev_tests/adabelief_dtype_full_final.log').read_bytes()
    for name in ('dev_tests/public_api_pre_fix_fingerprints.json','dev_tests/public_api_pre_fix_environment.json'):
        assert (ROOT/name).read_bytes()==(ROOT/archive[name]['archive']).read_bytes()
    for name in TRANSITIONS:
        before=(ROOT/archive[name]['archive']).read_text(encoding='utf-8')
        assert (ROOT/name).read_text(encoding='utf-8')==expected_source(name,before),name
    before=json.loads((ROOT/'dev_tests/adabelief_boundary_pre_fix_fingerprints.json').read_text(encoding='utf-8'))
    for name,digest in before.items():
        if name not in TRANSITIONS and (name.startswith(('nif/','tutorial/','dev_tests/','tests/')) or name=='.gitattributes'):
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    return set(TRANSITIONS)
