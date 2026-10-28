"""Exact T7 output/export successor; older scientific guards retain their bytes.

Only the recorded LF and CRLF notebook containers are interchangeable. JSON
content, Python source, archives, and all inverse patches remain exact.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = 'dev_tests/tutorial7_reference_v1.json'
RECORD_SHA256 = 'f1f086135d137e756b38bd4405f587401fc4699f2098f07610519f4d2b34e8ae'
SCOPE = 'dev_tests/tutorial7_reference_scope.py'
INVENTORY = 'dev_tests/tutorial7_reference_v1_inventory.json'
PREVIOUS_INVENTORY = 'dev_tests/lbfgs_reference_v1_inventory.json'
TRANSITIONS = frozenset({
    'dev_tests/generate_native_notebooks.py',
    'tutorial/7_model_pruning_and_quantization_pytorch.ipynb',
    'tutorial/7_model_pruning_and_quantization_colab.ipynb',
    'tests/build_tutorial_7.py',
    'tests/test_tutorial_7.py',
    'dev_tests/lbfgs_reference_scope.py',
    'dev_tests/publication_projection.py',
    'dev_tests/test_publication_checkout_eol.py',
})
NOTEBOOK_CELLS = {
    'tutorial/7_model_pruning_and_quantization_pytorch.ipynb': {0, 2, 8, 28, 31, 36, 37},
    'tutorial/7_model_pruning_and_quantization_colab.ipynb': {44, 47},
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def transition_record(root=ROOT):
    root = Path(root)
    raw = (root / RECORD).read_bytes()
    assert digest(raw) == RECORD_SHA256, RECORD
    record = json.loads(raw)
    assert set(record['transitions']) == TRANSITIONS, RECORD
    source = (root / SCOPE).read_bytes()
    slot = ("RECORD_SHA256 = '" + RECORD_SHA256 + "'").encode('utf-8')
    assert source.count(slot) == 1, SCOPE
    template = source.replace(slot, b"RECORD_SHA256 = '<tutorial7-record-sha256>'", 1)
    assert digest(template) == record['scope_source_template_sha256'], SCOPE
    return record


def inverse_bytes(root, name, current, record):
    entry = record['transitions'][name]
    forms = entry['after_sha256_by_eol']
    # Validate the complete raw container before the one allowed EOL conversion.
    assert digest(current) in forms.values(), name
    if digest(current) == forms.get('crlf'):
        assert b'\r' not in current.replace(b'\r\n', b''), name
        current = current.replace(b'\r\n', b'\n')
    assert digest(current) == forms['lf'] and b'\r' not in current, name
    archive = (Path(root) / entry['archive']).read_bytes()
    assert digest(archive) == entry['before_sha256'], entry['archive']
    if name in NOTEBOOK_CELLS:
        restored = json.loads(current)
        patches = entry['notebook_patches']
        assert {patch['cell'] for patch in patches} == NOTEBOOK_CELLS[name], name
        for patch in patches:
            target = restored['cells'][patch['cell']]
            assert target['cell_type'] == patch['cell_type'], name
            assert target['source'] == patch['after'], name
            target['source'] = patch['before']
        if name.endswith('_colab.ipynb'):
            target = restored['metadata']['nif_colab_compatibility']
            assert target['modified_code_cells'] == [0], name
            target['modified_code_cells'] = [0, 44, 47]
        assert restored == json.loads(archive), name
    else:
        text = current.decode('utf-8')
        for patch in reversed(entry['text_hunks']):
            start, end = patch['after_start'], patch['after_end']
            assert text[start:end] == patch['after'], name
            text = text[:start] + patch['before'] + text[end:]
        assert text.encode('utf-8') == archive, name
    return archive


def historical_tutorial7_bytes(root, name, current):
    """Validate and undo T7 first, then let the existing historical chain run."""
    if name not in TRANSITIONS:
        return current  # Unaffected minimal historical fixtures need no T7 files.
    return inverse_bytes(root, name, current, transition_record(root))


def assert_tutorial7_reference_scope(root=ROOT):
    root = Path(root)
    record = transition_record(root)
    for name, expected in record['required_sha256'].items():
        assert digest((root / name).read_bytes()) == expected, name
    original = record['original_tensorflow']
    assert digest((root / original['path']).read_bytes()) in original['sha256_by_eol'].values(), original['path']
    for name in sorted(TRANSITIONS):
        inverse_bytes(root, name, (root / name).read_bytes(), record)
    inventory = json.loads((root / INVENTORY).read_bytes())
    previous = json.loads((root / PREVIOUS_INVENTORY).read_bytes())
    assert inventory[:len(previous)] == previous, INVENTORY
    assert inventory[len(previous):] == record['added_public_files'], INVENTORY
    assert len(inventory) == len(set(inventory)), INVENTORY
    for name in inventory:
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts, name
        assert '.git' not in relative.parts and 'artifacts' not in relative.parts, name
        (root / name).stat()  # Keep structured missing-file identity.
        assert (root / name).is_file(), name
    return set(TRANSITIONS)
