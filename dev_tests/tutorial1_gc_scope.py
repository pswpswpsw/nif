"""Version-2 scope guard for the explicitly authorized tutorial-1 GC correction.

Version 1 and its historical passing suite are retained as non-collected bytes.
This guard permits exactly the GC selector/call and corresponding teaching note;
all notebook metadata, other cell sources, and generator source remain fixed.
"""
import copy
import hashlib
import json
from pathlib import Path

from dev_tests.lbfgs_reference_scope import historical_bytes

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "dev_tests/fixtures/tutorial1_gc_scope_v1"
OLD_CALL = "schedule=scheduler, centralization=True"
NEW_CALL = "schedule=scheduler, centralization=use_gradient_centralization"
NOTE = (
    "# Match actual current TensorFlow tutorial 1: its Colab fit does not activate GC.\n"
    "# Set True to opt in to native GC; this changes that primary reference behavior.\n"
    "use_gradient_centralization = False\n"
)
OLD_INTRO = "5,000 AdaBelief epochs with gradient centralization,"
NEW_INTRO = "5,000 AdaBelief epochs without gradient centralization by default,"
OPTIMIZER = "optimizer = AdaBeliefOptimizer(model_opt.parameters(), lr=lr)"


def assert_tutorial1_gc_scope_v2():
    archive_manifest = json.loads(
        (ARCHIVE / "manifest.json").read_text(encoding="utf-8")
    )
    for name, entry in archive_manifest.items():
        assert (
            hashlib.sha256((ARCHIVE / name).read_bytes()).hexdigest() == entry["sha256"]
        )
    original_fingerprints = json.loads(
        (ROOT / ("dev_tests/pruning_adam_pre_fix_fingerprints.json")).read_text(
            encoding=("utf-8")
        )
    )
    before_bytes = (ARCHIVE / "tutorial1_before.ipynb.txt").read_bytes()
    assert (
        hashlib.sha256(before_bytes).hexdigest()
        == original_fingerprints[("tutorial/1_simple_1d_wave_pytorch.ipynb")]
    )
    historical_test_hash = json.loads(
        (
            ROOT / ("dev_tests/pruning_adam_rounding_pre_fix_fingerprints.json")
        ).read_text(encoding=("utf-8"))
    )
    assert (
        archive_manifest[("test_pruning_adam_state.py.txt")][("sha256")]
        == historical_test_hash[("dev_tests/test_pruning_adam_state.py")]
    )
    assert "239 passed, 2 skipped" in (
        ARCHIVE / "historical_public_suite.log"
    ).read_text(encoding="utf-8")

    before = json.loads(before_bytes)
    expected = copy.deepcopy(before)
    changed = []
    for i, cell in enumerate(expected["cells"]):
        source = "".join(cell["source"])
        revised = source
        if cell["cell_type"] == "code" and OPTIMIZER in source:
            assert source.count(OLD_CALL) == 1
            revised = source.replace(OPTIMIZER, NOTE + OPTIMIZER).replace(
                OLD_CALL, NEW_CALL
            )
        elif i == 0:
            assert OLD_INTRO in source
            revised = source.replace(OLD_INTRO, NEW_INTRO)
        if revised != source:
            cell["source"] = revised.splitlines(keepends=True)
            changed.append(i)
    assert len(changed) == 2
    # Teaching v1 permits only the three independently validated T1 cells.
    assert_teaching_markdown_scope(ROOT)
    teaching = json.loads(
        (ROOT / "dev_tests/teaching_markdown_v1.json").read_text(encoding="utf-8")
    )
    for patch in teaching["notebooks"]["tutorial/1_simple_1d_wave_pytorch.ipynb"][
        "changes"
    ]:
        cell = expected["cells"][patch["cell"]]
        assert cell["cell_type"] == "markdown" and cell["source"] == patch["before"]
        cell["source"] = patch["after"]
    current = json.loads(
        historical_bytes(
            ROOT,
            ("tutorial/1_simple_1d_wave_pytorch.ipynb"),
            (ROOT / ("tutorial/1_simple_1d_wave_pytorch.ipynb")).read_bytes(),
        )
    )
    assert (
        current == expected
    ), "Only tutorial1 GC selector/call and teaching note are authorized"

    generator = (ARCHIVE / "generator_before.py.txt").read_text(encoding="utf-8")
    assert generator.count(OPTIMIZER) == 1 and generator.count(OLD_INTRO) == 1
    expected_generator = (
        generator.replace(OPTIMIZER, NOTE + OPTIMIZER)
        .replace(OLD_CALL, NEW_CALL, 1)
        .replace(OLD_INTRO, NEW_INTRO)
    )
    assert (
        historical_bytes(
            ROOT,
            ("dev_tests/generate_native_notebooks.py"),
            (ROOT / ("dev_tests/generate_native_notebooks.py")).read_bytes(),
        ).decode(("utf-8"))
        == expected_generator
    )


def assert_teaching_markdown_scope(root=ROOT):
    """Validate precisely the eight published Markdown replacements as JSON."""
    root = Path(root)
    name = "dev_tests/teaching_markdown_v1.json"
    raw = (root / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "aad2865228b9f93d9d47cd56c720f6c3fed56abb7ae1d430311970f0f428adfa"
    ), name
    teaching = json.loads(raw)
    for archive, digest in teaching["required_archives"].items():
        assert (
            hashlib.sha256((root / archive).read_bytes()).hexdigest() == digest
        ), archive
    assert len(teaching["notebooks"]) == 3
    assert sum(len(entry["changes"]) for entry in teaching["notebooks"].values()) == 8
    for name, entry in teaching["notebooks"].items():
        original = (root / entry["archive"]).read_bytes()
        assert hashlib.sha256(original).hexdigest() == entry["before_sha256"], name
        expected = json.loads(original)
        for patch in entry["changes"]:
            cell = expected["cells"][patch["cell"]]
            assert cell["cell_type"] == "markdown", name
            assert cell["source"] == patch["before"], name
            cell["source"] = patch["after"]
        current = historical_bytes(root, name, (root / name).read_bytes())
        assert json.loads(current) == expected, name
        assert hashlib.sha256(current).hexdigest() == entry["after_sha256"], name
    return set(teaching["notebooks"])
