"""Documentation inputs preserve executable examples and original site assets."""
import importlib.util
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
        "docs/index.html": b"<html>Archived TensorFlow documentation</html>\n",
        "docs/api_nif_model.html": b"<html>Legacy API</html>\n",
        "docs/_static/theme.js": b"// legacy theme\n",
        "docs/_sources/index.rst.txt": b"legacy source\n",
        "docs/PYTORCH_API.md": b"# Native API\n\n[training](PYTORCH_TRAINING_EXAMPLE.md)\n",
        "docs/PYTORCH_VALIDATION.md": b"# Recorded validation\n",
        "docs/PYTORCH_TENSORFLOW_PARITY.md": b"# Recorded comparisons\n",
        "docs/PYTORCH_TRAINING_EXAMPLE.md": b"# Training\n",
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
