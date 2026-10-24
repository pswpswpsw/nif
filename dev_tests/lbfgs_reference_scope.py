"""Exact public ReferenceLBFGS transition and inverse historical-source view.

Only the named optimizer sources and two T1 code cells have successors. Historical
archives, manifests and numerical assertions continue to check their old bytes.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = 'dev_tests/lbfgs_reference_v1.json'
RECORD_SHA256 = 'a2a1f310cd58ed1cfa0f2721929a8d366793ec2f88229afbdd24af178096e656'
SUCCESSORS = frozenset({
    'MANIFEST.in',
    'dev_tests/test_test_portability.py',
    '.gitattributes',
    'dev_tests/test_publication_checkout_eol.py',
    'dev_tests/test_publication_checkout_successor.py',
    'dev_tests/test_adabelief_final_update_successor.py',
    'tutorial/1_simple_1d_wave_pytorch.ipynb',
    'dev_tests/generate_native_notebooks.py',
    'dev_tests/test_tutorial1_lbfgs.py',
    'dev_tests/tutorial1_gc_scope.py',
    'dev_tests/publication_windows_scope.py',
    'dev_tests/adabelief_moment_scope.py',
    'dev_tests/publication_projection.py',
    'dev_tests/test_teaching_markdown_scope.py',
    'dev_tests/test_tutorial1_gc_scope.py',
    'nif/optimizers/lbfgs.py',
    'nif/optimizers/lbfgs_V2.py',
    'nif/optimizers/__init__.py',
})


def transition_record(root=ROOT):
    raw = (Path(root)/RECORD).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RECORD_SHA256, RECORD
    record = json.loads(raw)
    assert set(record['successors']) == SUCCESSORS, RECORD
    return record


def historical_bytes(root, name, current):
    """Undo only a frozen, exact named successor before an old guard compares it."""
    if name not in SUCCESSORS:
        return current
    root = Path(root)
    entry = transition_record(root)['successors'][name]
    assert hashlib.sha256(current).hexdigest() == entry['after_sha256'], name
    original = (root/entry['archive']).read_bytes()
    assert hashlib.sha256(original).hexdigest() == entry['before_sha256'], entry['archive']
    if 'notebook_patches' in entry:
        restored = json.loads(current)
        for patch in entry['notebook_patches']:
            cell = restored['cells'][patch['cell']]
            assert cell['cell_type'] == 'code' and cell['source'] == patch['after'], name
            cell['source'] = patch['before']
        if 'notebook_markdown_patch' in entry:
            patch = entry['notebook_markdown_patch']
            assert patch['cell'] == 32, name
            cell = restored['cells'][32]
            assert cell['cell_type'] == 'markdown' and cell['source'] == patch['after'], name
            cell['source'] = patch['before']
        assert restored == json.loads(original), name
    elif 'text_hunks' in entry:
        text = current.decode('utf-8')
        # Offsets refer to the AFTER source. Descending inverse application keeps
        # every other character fixed, including original line endings.
        for hunk in reversed(entry['text_hunks']):
            start, end = hunk['after_start'], hunk['after_end']
            assert text[start:end] == hunk['after'], name
            text = text[:start] + hunk['before'] + text[end:]
        assert text.encode('utf-8') == original, name
    else:
        # The three rewritten optimizer entry points have exact full-source
        # successors, each frozen independently from the historical archive.
        assert name in {'nif/optimizers/lbfgs.py', 'nif/optimizers/lbfgs_V2.py',
                        'nif/optimizers/__init__.py'}, name
    return original


def assert_reference_lbfgs_scope(root=ROOT):
    root = Path(root)
    record = transition_record(root)
    source = (root/'dev_tests/lbfgs_reference_scope.py').read_bytes()
    slot = ("RECORD_SHA256 = '" + RECORD_SHA256 + "'").encode('utf-8')
    assert source.count(slot) == 1, 'dev_tests/lbfgs_reference_scope.py'
    canonical = source.replace(slot, b"RECORD_SHA256 = '<transition-record-sha256>'", 1)
    assert hashlib.sha256(canonical).hexdigest() == record['scope_source_template_sha256'], 'dev_tests/lbfgs_reference_scope.py'
    for name in SUCCESSORS:
        historical_bytes(root, name, (root/name).read_bytes())
    for name, digest in record['required_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
    inventory = json.loads((root/'dev_tests/lbfgs_reference_v1_inventory.json').read_bytes())
    previous = json.loads((root/'dev_tests/teaching_markdown_v1_inventory.json').read_bytes())
    assert inventory[:len(previous)] == previous
    assert len(inventory) == len(set(inventory))
    assert inventory[len(previous):] == record['added_public_files']
    for name in inventory:
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts, name
        assert '.git' not in relative.parts and 'artifacts' not in relative.parts, name
        (root/name).stat()  # Preserve structured FileNotFoundError identity.
        assert (root/name).is_file(), name
    return set(SUCCESSORS)
