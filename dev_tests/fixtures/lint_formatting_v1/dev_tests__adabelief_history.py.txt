"""Hashed historical optimizer binding for initializer/model fingerprint tests.

This never changes nif exports. Current optimizer mathematics remain covered by
frozen source-derived scalar regressions and the empty-gradient boundary tests.
"""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE='dev_tests/fixtures/adabelief_dtype/native_before.py.txt'
expected=json.loads((ROOT/'dev_tests/adabelief_dtype_red_fingerprints.json').read_text(encoding='utf-8'))[SOURCE]
source=(ROOT/SOURCE).read_bytes()
assert hashlib.sha256(source).hexdigest()==expected
namespace={'__name__':'public_historical_adabelief_pre_scalar'}
exec(compile(source,str(ROOT/SOURCE),'exec'),namespace)
HistoricalAdaBelief=namespace['AdaBeliefOptimizer']
