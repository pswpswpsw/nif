"""Regression checks for the public README's original visual and tutorial entry points."""
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_retains_original_visuals_and_native_tutorials():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for relative in ("misc/myimage.gif", "misc/compare.gif"):
        assert (ROOT / relative).is_file(), f"Missing original image: {relative}"
        assert f"./{relative}" in readme, f"Original image is not displayed: {relative}"

    for stem in (
        "1_simple_1d_wave",
        "2_multi_scale_NIF",
        "3_multi_scale_linear_NIF",
        "4_get_gradients_by_wrapping_model_with_layer",
        "5_large_scale_training_on_tensorflow_record_data",
        "6_revisit_multi_scale_NIF_with_L1_L2_regularization",
        "7_model_pruning_and_quantization",
        "8_revisit_multi_scale_NIF_with_sobolov_training",
    ):
        notebook = f"{stem}_pytorch.ipynb"
        assert (ROOT / "tutorial" / notebook).is_file(), notebook
        assert f"/nif/blob/main/tutorial/{notebook}" in readme, notebook

    assert "## Highlights" in readme
    assert "PointWiseData.minmax_normalize" in readme
    assert "TFRDataset" in readme
    assert readme.count("https://colab.research.google.com/assets/colab-badge.svg") == 8
    assert "PyTorch" in readme
    # The tutorial 5 filename contains a historical format name. It is part
    # of an actual URL, not text rendered in the public README.
    visible = re.sub(r"\]\([^)]*\)", "]()", readme)
    assert not re.search(r"\b(?:tensorflow|keras)\b", visible, re.IGNORECASE)


def test_readme_documents_distributed_training_and_logging_examples():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "torchrun" in readme
    assert "TensorBoard" in readme
    assert (ROOT / "examples" / "train_pytorch.py").is_file()
    assert "MULTIGPU_SECTION_PENDING" not in readme
    assert "MONITORING_SECTION_PENDING" not in readme
