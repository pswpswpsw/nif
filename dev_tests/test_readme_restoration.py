"""Regression checks for the public README's original visual and tutorial entry points."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_retains_original_visuals_and_both_tutorial_families():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for relative in ("misc/myimage.gif", "misc/compare.gif"):
        assert (ROOT / relative).is_file(), f"Missing original image: {relative}"
        assert f'./{relative}' in readme, f"Original image is not displayed: {relative}"

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
        for suffix in ("_colab.ipynb", "_pytorch.ipynb"):
            notebook = f"{stem}{suffix}"
            assert (ROOT / "tutorial" / notebook).is_file(), notebook
            assert notebook in readme, f"Missing tutorial entry point: {notebook}"

    assert "## Highlights" in readme
    assert "PointWiseData.minmax_normalize" in readme
    assert "TFRecord" in readme or "tfrecord" in readme
    assert "PyTorch" in readme and "TensorFlow" in readme
