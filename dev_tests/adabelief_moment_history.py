"""Hashed PRE-moment-correction optimizer for explicitly historical tests only."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE='dev_tests/fixtures/adabelief_moment_v4/external_optimizers.py.txt'
previous=json.loads((ROOT/'dev_tests/adabelief_boundary_candidate_manifest.json').read_text(encoding='utf-8'))
expected=previous['source_sha256']['nif/optimizers/external_optimizers.py']
source=(ROOT/SOURCE).read_bytes()
assert hashlib.sha256(source).hexdigest()==expected
namespace={'__name__':'public_historical_adabelief_pre_moment'}
exec(compile(source,str(ROOT/SOURCE),'exec'),namespace)
PreMomentAdaBelief=namespace['AdaBeliefOptimizer']
