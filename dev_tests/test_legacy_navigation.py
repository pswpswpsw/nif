"""Relocated compatibility guides retain working local navigation."""
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "tutorial/legacy/tensorflow/colab_support"


@pytest.mark.parametrize(
    "name", ["README.md", "README-tutorials-2-8.md", "VALIDATION.md"]
)
def test_relocated_guides_link_to_existing_local_files(name):
    guide = SUPPORT / name
    text = guide.read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)]+)\)", text):
        if "://" in target or target.startswith("#"):
            continue
        path = (guide.parent / target.split("#", 1)[0]).resolve()
        assert path.is_relative_to(ROOT)
        assert path.is_file(), target
    assert "[tutorial index](../../../README.md)" in text
    assert "/blob/main/tutorial/1_simple_1d_wave_colab.ipynb" not in text
