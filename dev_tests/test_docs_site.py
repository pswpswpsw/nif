"""Documentation inputs preserve executable examples and original site assets."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def helper():
    path = ROOT / "scripts/build_docs.py"
    spec = importlib.util.spec_from_file_location("nif_docs_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def put(root, name, data):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "checkout"
    files = {
        "nif/__about__.py": b'__version__ = "2.0.0"\n',
        "docs/source/conf.py": b'project = "NIF"\n',
        "docs/source/index.rst": b"NIF documentation\n=================\n",
        "docs/source/_static/compare.png": b"original-png",
        "docs/index.html": b"<html>\nArchived TensorFlow documentation</html>\n",
        "docs/api_nif_model.html": b"<html>Legacy API</html>\n",
        "docs/_static/theme.js": b"// legacy theme\n",
        "docs/_sources/index.rst.txt": b"legacy source\n",
        "docs/PYTORCH_API.md": b"# Native API\n\n[training](PYTORCH_TRAINING_EXAMPLE.md)\n",
        "docs/PYTORCH_VALIDATION.md": b"# Recorded validation\n",
        "docs/PYTORCH_TENSORFLOW_PARITY.md": b"# Recorded comparisons\n",
        "docs/PYTORCH_TRAINING_EXAMPLE.md": b"# Training\n",
        "docs/SELF_SCALED_OPTIMIZERS.md": (
            b"# Experimental self-scaled optimizers\n\n"
            b"[source](../examples/self_scaled/benchmark.py)\n"
            b"[notebook](../tutorial/9_self_scaled_optimizers_pytorch.ipynb)\n"
            b"![Historical convergence](source/_static/self_scaled/convergence.png)\n"
            b"```bash\npython examples/self_scaled/benchmark.py --help\n```\n"
        ),
        "docs/source/_static/self_scaled/convergence.png": b"historical-ss-figure",
        "docs/private_analysis.md": b"must not be published implicitly\n",
        "docs/build/old_output.html": b"do not copy prior build output\n",
        "misc/myimage.gif": b"original-flow-gif",
        "misc/compare.gif": b"original-comparison-gif",
        "README.md": (
            b"# NIF\n\nPyTorch examples\n\n"
            b'<p align="center">\n  <img src="./misc/myimage.gif" alt="Flow" />\n</p>\n'
            b'<p align="center">\n  <img src="./misc/compare.gif" alt="Compare" />\n</p>\n'
            b"[API](docs/PYTORCH_API.md)\n[example](examples/train_pytorch.py)\n"
            b"```python\n# Preserve [literal](docs/PYTORCH_API.md)\nprint('exact example')\n```\n"
        ),
    }
    for name, data in files.items():
        put(root, name, data)
    legacy = (
        "index.html",
        "api_nif_model.html",
        "_static/theme.js",
        "_sources/index.rst.txt",
        "source/_static/compare.png",
    )
    manifest = {
        "schema_version": 1,
        "source_commit": "73b2ffcf71d9f0d019bd86d62cbf80989e9f7eac",
        "files": {
            name: {
                "sha256": hashlib.sha256(files["docs/" + name]).hexdigest(),
                "bytes": len(files["docs/" + name]),
                "text_lf": not name.endswith(".png"),
            }
            for name in legacy
        },
    }
    put(root, "docs/legacy-assets.json", json.dumps(manifest).encode("utf-8"))
    return root, files


def test_prepared_site_has_current_examples_and_original_images(checkout, tmp_path):
    root, originals = checkout
    source = tmp_path / "prepared"
    helper().prepare_source(root, source)
    quickstart = (source / "quickstart.md").read_bytes()
    exact_code = originals["README.md"].split(b"```python", 1)[1]
    assert b"```python" + exact_code in quickstart
    assert b"[API](api_guide.md)" in quickstart
    assert (
        b"https://github.com/pswpswpsw/nif/blob/v2.0.0/examples/train_pytorch.py"
        in quickstart
    )
    assert b"![Flow](_static/readme/myimage.gif)" in quickstart
    for name in ("myimage.gif", "compare.gif"):
        assert (source / "_static/readme" / name).read_bytes() == originals[
            "misc/" + name
        ]
    assert "[training](training_example.md)" in (source / "api_guide.md").read_text()
    assert not (source / "private_analysis.md").exists()
    for name, original in originals.items():
        assert (root / name).read_bytes() == original


def test_legacy_site_is_copied_verbatim_without_build_outputs(checkout, tmp_path):
    root, originals = checkout
    output = tmp_path / "html"
    helper().copy_legacy_site(root, output)
    for name in (
        "docs/index.html",
        "docs/api_nif_model.html",
        "docs/_static/theme.js",
        "docs/_sources/index.rst.txt",
        "docs/source/_static/compare.png",
    ):
        assert (
            output / "legacy/tensorflow" / name.removeprefix("docs/")
        ).read_bytes() == originals[name]
    assert not (output / "legacy/tensorflow/build").exists()
    assert not (output / "legacy/tensorflow/private_analysis.md").exists()


def test_legacy_copy_excludes_current_and_unrecorded_assets(checkout, tmp_path):
    root, _ = checkout
    put(root, "docs/future_page.html", b"new page outside the historical manifest")
    put(root, "docs/_static/future.js", b"// new theme resource")
    legacy = helper().copy_legacy_site(root, tmp_path / "html")
    assert not (legacy / "source/_static/self_scaled").exists()
    assert not (legacy / "future_page.html").exists()
    assert not (legacy / "_static/future.js").exists()


def test_rebuilding_legacy_removes_only_stale_generated_archive_files(
    checkout, tmp_path
):
    root, _ = checkout
    output = tmp_path / "html"
    put(output, "legacy/tensorflow/source/_static/self_scaled/stale.png", b"old copy")
    put(output, "outside-legacy.txt", b"unrelated output")
    legacy = helper().copy_legacy_site(root, output)
    assert not (legacy / "source/_static/self_scaled").exists()
    assert (output / "outside-legacy.txt").read_bytes() == b"unrelated output"


def test_legacy_copy_rejects_output_inside_original_docs(checkout):
    root, _ = checkout
    with pytest.raises(ValueError, match="original documentation"):
        helper().copy_legacy_site(root, root / "docs")


def test_legacy_copy_requires_recorded_bytes_before_writing(checkout, tmp_path):
    root, _ = checkout
    put(root, "docs/source/_static/compare.png", b"changed original image")
    output = tmp_path / "html"
    with pytest.raises(ValueError, match="source/_static/compare.png"):
        helper().copy_legacy_site(root, output)
    assert not (output / "legacy/tensorflow").exists()


def test_legacy_copy_preserves_uniform_crlf_checkout_bytes(checkout, tmp_path):
    root, originals = checkout
    current = originals["docs/index.html"].replace(b"\n", b"\r\n")
    put(root, "docs/index.html", current)
    legacy = helper().copy_legacy_site(root, tmp_path / "html")
    assert (legacy / "index.html").read_bytes() == current


def test_legacy_copy_rejects_mixed_checkout_endings(checkout, tmp_path):
    root, originals = checkout
    current = originals["docs/index.html"].replace(b"\n", b"\r\n", 1)
    put(root, "docs/index.html", current)
    with pytest.raises(ValueError, match="index.html"):
        helper().copy_legacy_site(root, tmp_path / "html")


@pytest.mark.parametrize("name", ["../outside.txt", "/absolute.txt", "a/../b.txt"])
def test_legacy_copy_rejects_unsafe_manifest_paths(checkout, tmp_path, name):
    root, _ = checkout
    manifest = {"schema_version": 1, "files": {name: {"sha256": "0" * 64}}}
    put(root, "docs/legacy-assets.json", json.dumps(manifest).encode("utf-8"))
    with pytest.raises(ValueError, match="legacy asset path"):
        helper().copy_legacy_site(root, tmp_path / "html")


def test_rejects_output_inside_original_documentation(checkout):
    root, _ = checkout
    with pytest.raises(ValueError, match="original documentation"):
        helper().validate_output(root, root / "docs/new-output")


def test_repository_link_rewrite_keeps_external_and_anchor_links():
    text = (
        "[external](https://example.org/a.md) [local](#section) [source](../nif/model.py) "
        "[tutorial](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/one.ipynb)"
    )
    rewritten = helper().rewrite_markdown(text, "docs/PYTORCH_API.md", "v2.0.0")
    assert "[external](https://example.org/a.md)" in rewritten
    assert "[local](#section)" in rewritten
    assert "nif/blob/v2.0.0/tutorial/one.ipynb" in rewritten
    assert (
        "[source](https://github.com/pswpswpsw/nif/blob/v2.0.0/nif/model.py)"
        in rewritten
    )


def test_post_release_experiment_links_use_current_source_not_old_release():
    notebook = "tutorial/9_self_scaled_optimizers_pytorch.ipynb"
    text = (
        "[example](../examples/self_scaled/benchmark.py) "
        "[notebook](../" + notebook + ") "
        "[Colab](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/"
        + notebook
        + ") "
        "[source](https://github.com/pswpswpsw/nif/tree/main/examples/self_scaled) "
        "[original](../tutorial/1_simple_1d_wave_pytorch.ipynb)"
    )
    rewritten = helper().rewrite_markdown(
        text, "docs/SELF_SCALED_OPTIMIZERS.md", "v2.0.0"
    )
    assert "nif/blob/main/examples/self_scaled/benchmark.py" in rewritten
    assert "nif/blob/main/" + notebook in rewritten
    assert "nif/tree/main/examples/self_scaled" in rewritten
    assert "nif/blob/v2.0.0/tutorial/1_simple_1d_wave_pytorch.ipynb" in rewritten
    assert "nif/blob/v2.0.0/" + notebook not in rewritten


def test_prepared_experiment_guide_retains_local_historical_assets(checkout, tmp_path):
    root, originals = checkout
    source = tmp_path / "prepared"
    helper().prepare_source(root, source)
    guide = (source / "self_scaled_optimizers.md").read_text()
    assert "![Historical convergence](_static/self_scaled/convergence.png)" in guide
    assert "nif/blob/main/examples/self_scaled/benchmark.py" in guide
    assert "```bash\npython examples/self_scaled/benchmark.py --help\n```" in guide
    assert (source / "_static/self_scaled/convergence.png").read_bytes() == originals[
        "docs/source/_static/self_scaled/convergence.png"
    ]


def add_snapshot(root):
    location = "docs/archive/2026-11-15/"
    snapshots = {
        "docs/PYTORCH_API.md": (
            "PYTORCH_API.md",
            b"# Historical API\n\n"
            b"![Original figure](source/_static/compare.png)\n"
            b"[validation](PYTORCH_VALIDATION.md#recorded-validation)\n"
            b"[source](../nif/model.py)\n",
        ),
        "docs/PYTORCH_VALIDATION.md": (
            "PYTORCH_VALIDATION.md",
            b"# Recorded validation\n",
        ),
    }
    records = {}
    for original, (name, data) in snapshots.items():
        put(root, location + name, data)
        records[original] = {
            "archive": name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    manifest = {
        "source_commit": "73b2ffcf71d9f0d019bd86d62cbf80989e9f7eac",
        "snapshot_date": "2026-11-15",
        "records": records,
    }
    put(root, location + "manifest.json", json.dumps(manifest).encode("utf-8"))
    put(root, location + "README.md", b"# Snapshot index\n\n[API](PYTORCH_API.md)\n")
    return snapshots


def test_archived_pages_keep_source_context_and_local_figure_links(checkout, tmp_path):
    root, _ = checkout
    snapshots = add_snapshot(root)
    source = tmp_path / "prepared"
    helper().prepare_source(root, source)
    page = (source / "archive/2026-11-15/PYTORCH_API.md").read_text()
    assert "orphan: true" in page
    assert "2026-11-15" in page
    assert "![Original figure](../../_static/compare.png)" in page
    assert "[validation](PYTORCH_VALIDATION.md#recorded-validation)" in page
    assert "nif/blob/73b2ffcf71d9f0d019bd86d62cbf80989e9f7eac/nif/model.py" in page
    for _, (name, original) in snapshots.items():
        assert (root / "docs/archive/2026-11-15" / name).read_bytes() == original


def test_archive_links_render_locally_and_manifest_links_use_main():
    text = (
        "[history](archive/2026-11-15/PYTORCH_API.md#historical-api) "
        "[manifest](archive/2026-11-15/manifest.json)"
    )
    rewritten = helper().rewrite_markdown(text, "docs/PYTORCH_VALIDATION.md", "v2.0.0")
    assert "[history](archive/2026-11-15/PYTORCH_API.md#historical-api)" in rewritten
    assert "nif/blob/main/docs/archive/2026-11-15/manifest.json" in rewritten


def test_changed_archive_is_rejected_before_rendering(checkout, tmp_path):
    root, _ = checkout
    add_snapshot(root)
    put(root, "docs/archive/2026-11-15/PYTORCH_API.md", b"changed historical report")
    with pytest.raises(ValueError, match="PYTORCH_API.md"):
        helper().prepare_source(root, tmp_path / "prepared")


def test_legacy_readme_snapshot_can_keep_its_original_root_link_context(
    checkout, tmp_path
):
    root, _ = checkout
    add_snapshot(root)
    location = root / "docs/archive/2026-11-15"
    data = b"# Legacy README\n\n![Flow](misc/myimage.gif)\n"
    (location / "legacy.md").write_bytes(data)
    manifest = json.loads((location / "manifest.json").read_text())
    manifest["records"]["docs/README_TENSORFLOW_LEGACY.md"] = {
        "archive": "legacy.md",
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "link_source": "README.md",
    }
    (location / "manifest.json").write_text(json.dumps(manifest))
    source = tmp_path / "prepared"
    helper().prepare_source(root, source)
    page = (source / "archive/2026-11-15/legacy.md").read_text()
    assert "![Flow](../../_static/readme/myimage.gif)" in page


def test_current_validation_does_not_gain_a_runtime_only_historical_banner(
    checkout, tmp_path
):
    root, originals = checkout
    source = tmp_path / "prepared"
    helper().prepare_source(root, source)
    assert (source / "validation.md").read_bytes() == originals[
        "docs/PYTORCH_VALIDATION.md"
    ]
