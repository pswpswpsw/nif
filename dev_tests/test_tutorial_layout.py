"""Published tutorials distinguish the framework from their execution platform."""
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY = Path("tutorial/legacy/tensorflow")


def catalog():
    return json.loads((ROOT / "tutorial/catalog.json").read_text(encoding="utf-8"))


def test_legacy_notebooks_and_support_have_a_framework_named_archive():
    entries = catalog()["tutorials"]
    native = [entry["notebook"] for entry in entries]
    assert len(native) == 9
    assert all(Path(name).parent == Path("tutorial") for name in native)
    for entry in entries:
        assert (ROOT / entry["notebook"]).is_file()
        if entry["status"] == "released":
            for key in ("original", "compatibility"):
                name = Path(entry[key])
                assert name.parent == LEGACY
                assert (ROOT / name).is_file()
                assert not (ROOT / "tutorial" / name.name).exists()
    assert (ROOT / LEGACY / "colab_support/bootstrap_tutorials.py").is_file()
    assert not (ROOT / "tutorial/colab_support").exists()


def test_historical_links_keep_the_original_published_paths():
    data = catalog()
    assert data["links"]["historical_docs_ref"] == "v2.0.0"
    text = (ROOT / "docs/source/tutorials.rst").read_text(encoding="utf-8")
    for entry in data["tutorials"]:
        if entry["status"] == "released":
            for key in ("original", "compatibility"):
                published = entry["historical_" + key]
                assert Path(published).parent == Path("tutorial")
                assert Path(published).name == Path(entry[key]).name
                assert "/blob/v2.0.0/" + published in text
    assert "/blob/v2.0.0/tutorial/legacy/" not in text


def test_current_colab_links_open_the_same_pytorch_notebooks():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    links = re.findall(
        r"https://colab\.research\.google\.com/github/pswpswpsw/nif/blob/[^)\s]+",
        text,
    )
    assert len(links) == 9
    assert all(link.endswith("_pytorch.ipynb") for link in links)
    guide = (ROOT / "tutorial/README.md").read_text(encoding="utf-8")
    assert "Colab is a notebook execution platform" in guide
    assert "legacy/tensorflow/README.md" in guide


def test_legacy_index_explains_both_tensorflow_sets_and_has_local_targets():
    path = ROOT / LEGACY / "README.md"
    text = path.read_text(encoding="utf-8")
    assert "TensorFlow" in text and "compatibility" in text
    links = re.findall(r"\]\(([^)]+)\)", text)
    for link in links:
        if "://" not in link and not link.startswith("#"):
            assert (path.parent / link.split("#", 1)[0]).resolve().is_file(), link
    for entry in catalog()["tutorials"]:
        if entry["status"] == "released":
            for key in ("original", "compatibility"):
                assert "](" + Path(entry[key]).name + ")" in text
