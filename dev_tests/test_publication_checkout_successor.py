"""Checkout-only succession, with unchanged historical hashes and strict negatives."""
import hashlib
import json
from pathlib import Path

import pytest

from dev_tests.adabelief_moment_scope import assert_adabelief_phase_scope_v4
from dev_tests.lbfgs_reference_scope import historical_bytes
from dev_tests.publication_windows_scope import ROOT


def test_checkout_helper_is_the_recorded_exact_successor():
    # New successor hash; the original source is separately frozen by the RED test.
    current = historical_bytes(
        ROOT,
        "dev_tests/publication_windows_scope.py",
        (ROOT / "dev_tests/publication_windows_scope.py").read_bytes(),
    )
    teaching_changes = [
        (
            b"        source=source.replace(old,new,1)\n    return source\n",
            (
                b"        source=source.replace(old,new,1)\n    if name in MARK"
                b"DOWN_TRANSITIONS:\n        for old,new in MARKDOWN_TRANSITION"
                b"S[name]:\n            assert source.count(old)==1,(name,old)\n"
                b"            source=source.replace(old,new,1)\n    return sour"
                b"ce\n"
            ),
        ),
        (
            b"    root=Path(root)\n    archive=json.loads(",
            b"    root=Path(root)\n    assert_teaching_helper_source(root)\n    archive=json.loads(",
        ),
        (
            b"    assert hashlib.sha256(current[:-len(suffix)]).hexdigest()==digest,'.gitattributes'\n",
            (
                b"    assert hashlib.sha256(current[:-len(suffix)]).hexdigest("
                b")==digest,'.gitattributes'\n\n\nMARKDOWN_TRANSITIONS={'dev_test"
                b"s/adabelief_moment_scope.py': [('def assert_adabelief_phase_"
                b"scope_v4():\\n', \"def assert_adabelief_phase_scope_v4():\\n   "
                b" from dev_tests.publication_windows_scope import assert_teac"
                b"hing_helper_source\\n    assert_teaching_helper_source(ROOT)\\"
                b"n    from dev_tests.tutorial1_gc_scope import assert_teachin"
                b"g_markdown_scope\\n    teaching=assert_teaching_markdown_scop"
                b"e(ROOT)|{'dev_tests/tutorial1_gc_scope.py'}\\n\"), (\"        i"
                b"f name not in TRANSITIONS and name not in portable and (name"
                b".startswith(('nif/','tutorial/','dev_tests/','tests/')) or n"
                b"ame=='.gitattributes'):\", \"        if name not in TRANSITION"
                b"S and name not in portable and name not in teaching and (nam"
                b"e.startswith(('nif/','tutorial/','dev_tests/','tests/')) or "
                b"name=='.gitattributes'):\")], 'dev_tests/test_tutorial1_gc_sc"
                b"ope.py': [(\"                'tutorial/1_simple_1d_wave_pytor"
                b"ch.ipynb','docs/PYTORCH_CANDIDATE_REPORT.md'}\", \"           "
                b"     'tutorial/1_simple_1d_wave_pytorch.ipynb','docs/PYTORCH"
                b"_CANDIDATE_REPORT.md',\\n                'tutorial/5_large_sc"
                b"ale_training_on_tensorflow_record_data_pytorch.ipynb',\\n    "
                b"            'tutorial/8_revisit_multi_scale_NIF_with_sobolov"
                b"_training_pytorch.ipynb'}\")], 'dev_tests/publication_project"
                b"ion.py': [(\"source/'dev_tests/publication_windows_v2_invento"
                b"ry.json'\", \"source/'dev_tests/teaching_markdown_v1_inventory"
                b".json'\")]}\n\n\ndef assert_teaching_helper_source(root=ROOT):\n "
                b'   """Freeze the exact JSON validator and require its public'
                b' provenance."""\n    root=Path(root)\n    name=\'dev_tests/tuto'
                b"rial1_gc_scope.py'\n    assert hashlib.sha256((root/name).rea"
                b"d_bytes()).hexdigest()=='97b11dda02bcce8e088659bcedede1e66b7"
                b"c62ce8362a09b768a5c13f005fd32',name\n    archive='dev_tests/f"
                b"ixtures/teaching_markdown_v1/tutorial1_gc_scope.py.before.tx"
                b"t'\n    assert hashlib.sha256((root/archive).read_bytes()).he"
                b"xdigest()=='29fa494900a56d2a4875d3970fa262f2acddfccae0f0ebdc"
                b"538a6fd869c585bb',archive\n    successors=json.loads((root/'d"
                b"ev_tests/teaching_markdown_v1_sources.json').read_text(encod"
                b"ing='utf-8'))\n    assert successors['source_before_sha256']["
                b"name]=='29fa494900a56d2a4875d3970fa262f2acddfccae0f0ebdc538a"
                b"6fd869c585bb',name\n    assert successors['source_after_sha25"
                b"6'][name]=='97b11dda02bcce8e088659bcedede1e66b7c62ce8362a09b"
                b"768a5c13f005fd32',name\n    assert (root/'dev_tests/fixtures/"
                b"teaching_markdown_v1/scope.diff').read_bytes()\n"
            ),
        ),
    ]
    for old, new in reversed(teaching_changes):
        assert current.count(new) == 1
        current = current.replace(new, old, 1)
    added = (
        b"        # Final-update v5: authorize only the one staged par"
        b"ameter-update tuple.\n        old=\" 'nif/optimizers/external_"
        b"optimizers.py':[\\n\"\n        new=old+\"  ('                p.a"
        b"dd_(direction, alpha=-lr)\\\\n',\\n   '                p.sub_(d"
        b"irection * lr)\\\\n',1),\\n\"\n        assert source.count(old)=="
        b"1\n        source=source.replace(old,new,1)\n"
    )
    assert current.count(added) == 1
    historical = current.replace(added, b"", 1)
    assert hashlib.sha256(historical).hexdigest() == (
        "5e056ad1626384b96d9dca4ffa67dc1d5ed3a6807a52944f633b873ea5cff431"
    )


@pytest.mark.parametrize("mutation", ["original-rule", "extra-rule", "missing-rule"])
def test_checkout_attributes_reject_any_change_beyond_the_four_rules(
    monkeypatch, mutation
):
    original = Path.read_bytes

    def changed(path):
        data = original(path)
        if path == ROOT / ".gitattributes":
            if mutation == "original-rule":
                return data.replace(b"/tests/* text eol=lf", b"/tests/* -text")
            if mutation == "extra-rule":
                return data + b"/nif/** -text\n"
            return data.replace(b"/tutorial/README.md text eol=lf\n", b"")
        return data

    monkeypatch.setattr(Path, "read_bytes", changed)
    with pytest.raises(AssertionError, match=r"\.gitattributes"):
        assert_adabelief_phase_scope_v4()


@pytest.mark.parametrize(
    "name,manifest,key",
    [
        (
            "dev_tests/adabelief_moment_scope.diff",
            "dev_tests/adabelief_moment_candidate_manifest.json",
            "source_sha256",
        ),
        (
            "dev_tests/publication_v1_scope.diff",
            "dev_tests/publication_v1_manifest.json",
            "public_file_sha256",
        ),
        (
            "dev_tests/publication_windows_v2_scope.diff",
            "dev_tests/publication_windows_v2_manifest.json",
            "public_file_sha256",
        ),
    ],
)
def test_patch_records_retain_their_unchanged_historical_hash(name, manifest, key):
    recorded = json.loads((ROOT / manifest).read_text(encoding="utf-8"))
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == recorded[key][name]


def test_checkout_scope_rejects_substantive_readme_edit(monkeypatch):
    original = Path.read_bytes

    def changed(path):
        data = original(path)
        return data + b"changed\n" if path == ROOT / "tutorial/README.md" else data

    monkeypatch.setattr(Path, "read_bytes", changed)
    with pytest.raises(AssertionError, match="tutorial/README.md"):
        assert_adabelief_phase_scope_v4()
