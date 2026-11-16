"""Current documentation agrees with the tutorial and keeps dated evidence."""
import hashlib
import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/archive/2026-11-15"
COLAB_GUIDE_HASH = "efe37f928ae6f3ea2d1c60f7a16f909b1e3ccab33bc471d9f0e4b95f8247ac6d"
BASELINE = {
    "PYTORCH_TENSORFLOW_PARITY.md": "b1830dab2a72faa0f9e12059975f8381db27281fed95873efb8301ed9b56c091",
    "PYTORCH_CANDIDATE_REPORT.md": "0d76aa6c9bcb8ce40be2e4365d40c965ca06930a0d2bf7b6bb67c699c40c9b95",
    "PYTORCH_VALIDATION.md": "16529119e1641af9889bf229c6ea5418b2c309fa268251a3c296b91d360876a8",
    "PYTORCH_API.md": "6665a515594ccaeec2d1fb73a0453a99814ed98c3b967eee282e721142a456b8",
    "README.rst": "be839f65b7d4eed6bdf6feaa7b84840d6e623ef5749358ef22d4cef4d6754989",
    "README_TENSORFLOW_LEGACY.md": "b32e5b43de2f4d4c04272e912da20aee9a038ee2cd83c4fe451114972baba69b",
    "colab-support/README-tutorials-2-8.md": COLAB_GUIDE_HASH,
    "colab-support/README.md": "81a037b04810ab7d49c765a5b151d5106379076f695890a1677d1157b889bdd9",
    "colab-support/VALIDATION.md": "06380077d4f4308b2ce1e658d09ea0aed1962e261653587ea0290df6dc5cb773",
}


@pytest.mark.parametrize("name,digest", BASELINE.items())
def test_dated_records_retain_the_original_git_bytes(name, digest):
    archived = ARCHIVE / name
    assert hashlib.sha256(archived.read_bytes()).hexdigest() == digest
    manifest = json.loads((ARCHIVE / "manifest.json").read_text(encoding="utf-8"))
    record = next(
        item for item in manifest["records"].values() if item["archive"] == name
    )
    assert record["sha256"] == digest
    assert record["bytes"] == archived.stat().st_size
    assert manifest["source_commit"] == "73b2ffcf71d9f0d019bd86d62cbf80989e9f7eac"


def test_current_t7_metric_matches_the_original_tutorial_definition():
    api = (ROOT / "docs/PYTORCH_API.md").read_text(encoding="utf-8")
    assert "norm(error, ord=2)**2 / norm(unpruned, ord=2)**2" in api
    assert "matrix spectral" in api
    assert "native metric\ncorrects the earlier MSE label" not in api
    guide = (
        ROOT / "tutorial/legacy/tensorflow/colab_support/README-tutorials-2-8.md"
    ).read_text(encoding="utf-8")
    assert "ord=2" in guide
    assert "replacing the matrix spectral norm" not in guide


def test_current_validation_links_to_one_comparison_summary():
    text = (ROOT / "docs/PYTORCH_VALIDATION.md").read_text(encoding="utf-8")
    assert "training comparison in progress" not in text
    assert "Implementation branch: `pytorch`" not in text
    assert "PYTORCH_TENSORFLOW_PARITY.md" in text
    assert "archive/2026-11-15/PYTORCH_VALIDATION.md" in text


def test_current_comparison_retains_all_eight_reviews_and_failed_criteria():
    text = (ROOT / "docs/PYTORCH_TENSORFLOW_PARITY.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\|\s*([1-8])\s*\|", text, flags=re.MULTILINE)
    assert rows == [str(value) for value in range(1, 9)]
    for result in ("19/27", "21/27", "24/30", "57/81", "36/60", "9,792"):
        assert result in text
    assert "archive/2026-11-15/PYTORCH_TENSORFLOW_PARITY.md#" in text
    assert "### Candidate 016" not in text


def test_experiment_guide_keeps_figures_and_distinct_execution_receipts():
    text = (ROOT / "docs/SELF_SCALED_OPTIMIZERS.md").read_text(encoding="utf-8")
    for name in (
        "lbfgs_original_style.png",
        "ssbfgs_original_style.png",
        "ssbroyden_original_style.png",
        "training_physical_comparison.png",
        "dense_absolute_errors.png",
        "convergence.png",
        "historical_results.json",
        "tutorial9_validation.json",
        "tutorial9_colab_validation.json",
        "tutorial9_colab_smoke_validation.json",
    ):
        assert "source/_static/self_scaled/" + name in text
    for value in ("6.06450e-5", "1.25679e-6", "3.76647e-7", "5.92239e-5"):
        assert value in text
