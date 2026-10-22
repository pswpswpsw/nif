# Tutorials 2–8: Colab compatibility copies

Run the seven `tutorial/*_colab.ipynb` copies from their first cell in a fresh runtime. The original notebooks are preserved. Every original Markdown cell and code comment is retained exactly; compatibility explanations live in this guide rather than additional notebook cells. The copies retain the original data sizes, model settings, learning rates, batch sizes, training lengths, and pruning schedule.

| Copy | Full workload |
| --- | --- |
| `2_multi_scale_NIF_colab.ipynb` | Two 5,000-epoch fits |
| `3_multi_scale_linear_NIF_colab.ipynb` | 600 epochs on all 191,775 cylinder-flow samples |
| `4_get_gradients_by_wrapping_model_with_layer_colab.ipynb` | Jacobian and Hessian examples; no training |
| `5_large_scale_training_on_tensorflow_record_data_colab.ipynb` | 1,000,000 rows, 100 TFRecord shards, two complete passes |
| `6_revisit_multi_scale_NIF_with_L1_L2_regularization_colab.ipynb` | Nine 3,000-epoch fits: baseline, four L2 settings, four L1 settings |
| `7_model_pruning_and_quantization_colab.ipynb` | 5,000 baseline epochs, 50,000 pruning epochs, export and TFLite inference |
| `8_revisit_multi_scale_NIF_with_sobolov_training_colab.ipynb` | Two 50,000-epoch fits, derivative evaluation and superresolution |

The cylinder-flow and traveling-wave datasets come with the pinned NIF package. Tutorial 5 generates its own data. Use a separate working directory for each notebook because checkpoints, logs, and image filenames overlap.

Open the GitHub copies directly in Colab:
[2](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/2_multi_scale_NIF_colab.ipynb),
[3](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/3_multi_scale_linear_NIF_colab.ipynb),
[4](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/4_get_gradients_by_wrapping_model_with_layer_colab.ipynb),
[5](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/5_large_scale_training_on_tensorflow_record_data_colab.ipynb),
[6](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/6_revisit_multi_scale_NIF_with_L1_L2_regularization_colab.ipynb),
[7](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/7_model_pruning_and_quantization_colab.ipynb), and
[8](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/8_revisit_multi_scale_NIF_with_sobolov_training_colab.ipynb).
Downloaded full-run snapshots matched the pre-publication copies, allowing Colab's trailing-whitespace normalization. Publication changes only bootstrap formatting and independent standard-library import order; normalized executable ASTs match and every other source cell is unchanged. See [publication validation](VALIDATION.md).

## Runtime and fixes

Each copy embeds `bootstrap_tutorials.py` as its first cell. It sets `TF_USE_LEGACY_KERAS=1` before importing TensorFlow, checks the installed versions, and installs NIF from commit `ad9113df14590189b12068b6b3fa7c8c8d9187d4`. NIF compatibility patches are applied to the temporary `nif-tutorials-v2-ad9113df1459` runtime directory. Repository package sources remain unchanged. Python 3.11–3.13 is supported by this bootstrap.

| Package | Pinned version |
| --- | --- |
| TensorFlow / tf-keras / TensorBoard | 2.21.0 / 2.21.0 / 2.21.0 |
| Keras | 3.13.2 |
| TensorFlow Probability | 0.25.0 |
| TensorFlow Model Optimization | 0.8.1 |
| NumPy / SciPy | 2.1.3 / 1.16.3 |
| Matplotlib / absl-py | 3.10.0 / 1.4.0 |

The shared runtime patches update the distribution-context import, put AdaBelief/L4Adam on the legacy optimizer base, replace removed `np.product` calls, and address two execution issues:

- **TFRecord targets:** remove the shard-batch singleton dimension before transposing, so tutorial 5 receives targets shaped `(number_of_points, number_of_targets)`.
- **Gradient centralization:** attach the transformation to the gradient-tape optimizer path used by current `model.fit`; tutorial 3's original `get_gradients` assignment alone does not activate it.

Three small notebook changes accompany the bootstrap:

- **Tutorial 5:** use a portable `pathlib` file listing in place of `%ls TFR_dir -lrth`.
- **Tutorial 7:** use the Frobenius norm for the reported relative mean-square error, replacing the matrix spectral norm.
- **Tutorial 8:** convert superresolution predictions back to physical units before comparing them with the analytic wave. Its original loss still penalizes the field and spatial derivative; the temporal derivative remains a reported metric.

If TensorFlow or a package that needs changing has already been imported, restart the runtime and begin at the first cell. The bootstrap reports package versions and the loaded NIF source location.

## Full local execution

The shared runner is `tests/execute_tutorial1.py`; despite its filename, it executes any supplied notebook cell by cell, saves the executed notebook after each completed code cell, and writes a companion `.events.json`. Full runs omit `--smoke`, whose reductions are specific to tutorial 1. Its timeout is 7,200 seconds per cell; a timeout is an incomplete run.

The validated Windows CPU environment uses Python 3.13.11. With `uv` and Git installed, run from the cloned repository root in PowerShell 7. Create the environment once, then execute the desired notebook:

```powershell
$nifEnvironment = Join-Path $env:USERPROFILE '.cache\nif-colab-qa\colab-matched'
uv venv --python 3.13.11 $nifEnvironment
$nifPython = Join-Path $nifEnvironment 'Scripts\python.exe'
uv pip install --python $nifPython -r 'tutorial\colab_support\requirements-qa.txt'
& $nifPython 'tests\execute_tutorial1.py' `
  --notebook 'tutorial\2_multi_scale_NIF_colab.ipynb' `
  --output '.codex\validation\tutorials-2-3\tutorial2.reproduction.executed.ipynb' `
  --workdir (Join-Path $env:USERPROFILE '.cache\nif-colab-qa\reproduction-2')
```

Native Windows TensorFlow in this environment uses CPU. The separate validated WSL Ubuntu environment used Python 3.13.11 and an NVIDIA GeForce RTX 3080 Ti Laptop GPU. For local GPU reproduction, prepare a Linux/WSL environment with the shared requirements and CUDA dependencies following [TensorFlow's pip installation instructions](https://www.tensorflow.org/install/pip). From the repository root inside WSL, point `NIF_ENV` to that environment. The example collects its NVIDIA wheel library paths so the notebook kernel can find them:

```bash
NIF_ENV="$HOME/.cache/nif-colab-qa/colab-matched"
NIF_CUDA_LIBS=$("$NIF_ENV/bin/python" -c 'import pathlib, sysconfig; p=pathlib.Path(sysconfig.get_paths()["purelib"]); print(":".join(str(x) for x in sorted((p/"nvidia").glob("*/lib"))))')
LD_LIBRARY_PATH="$NIF_CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
TF_FORCE_GPU_ALLOW_GROWTH=true \
"$NIF_ENV/bin/python" tests/execute_tutorial1.py \
  --notebook tutorial/7_model_pruning_and_quantization_colab.ipynb \
  --output .codex/validation/tutorial-7/reproduction.executed.ipynb \
  --workdir "$HOME/.cache/nif-colab-qa/reproduction-7"
```

Use an unused output name and working directory for each reproduction. The maintainer's GPU setup and device-calculation records are retained locally; the [public validation summary](VALIDATION.md) records the tested environments and results.

## Validation status

Verified on 2026-10-21: all seven copies completed full execution in hosted Colab T4 runtimes, with **112 executable code cells passed** at the original training settings. The original Markdown, comments, and scientific configurations remain preserved.

The final focused regression suite passed **25/25 tests** in eight isolated Python processes. It covers target/weight shapes, actual gradient centralization in eager and graph training, analytic derivatives, regularization penalties, pruning and TFLite inference, physical output scaling, preservation, bootstrap version checks, empty cells, and cloud payload fidelity. That full-run bootstrap SHA256 was `c3f73140a34c182172701498e01791dfb7c257603f692930b5128b09db1a3a1d`. After publication formatting, all 29 tests passed with the published bootstrap; both hashes are recorded in [validation-summary.json](validation-summary.json).

| Tutorials | Full execution status |
| --- | --- |
| 2 | Windows CPU full run passed all 9 code cells in 876.54 seconds; hosted T4 full run passed in 370.41 seconds |
| 3 | Windows CPU full run passed all 14 code cells in 1,172.89 seconds; hosted T4 full run passed in 1,337.56 seconds |
| 4 | Hosted T4 full run passed all 10 code cells with the final helper in 10.02 seconds; earlier full Windows CPU run also passed |
| 5 | Hosted T4 full run passed all 9 code cells with the final helper in 56.97 seconds, retaining one million rows and both complete passes; earlier full Windows CPU run also passed |
| 6 | Hosted T4 full run passed all 21 executable code cells in 1,194.30 seconds. WSL GPU also passed in 1,076.69 seconds: all nine 3,000-epoch fits, 27,000 finite losses, nine restored checkpoints, and finite training and 137,641-point predictions; both empty cells are preserved |
| 7 | Hosted T4 full run passed all 28 code cells in 2,237.20 seconds. WSL GPU also passed in 2,724.81 seconds: 5,000 baseline epochs and 50,000 pruning epochs, finite losses, 90.005% exported-kernel sparsity, genuine int8 weights, and 2,000 finite TFLite predictions |
| 8 | Hosted T4 Run all completed both 50,000-epoch fits and all final evaluations. The downloaded 40-cell notebook has 21 executable code cells numbered sequentially 1–21, no Python errors, and 44 embedded PNG outputs; its sources match the local copy |

The completed hosted validation used a harness for tutorials 2–7 and direct Run all for tutorial 8. The harness passed 91 executable cells using the complete delivered sources in fresh Jupyter kernels, separate working directories, and unchanged training settings. Its payload fidelity check verified every source cell and the input notebook hashes. Tutorial 8 passed its 21 executable cells through direct Run all. Local WSL GPU execution and hosted Colab execution are recorded as separate validation environments in the [public validation summary](VALIDATION.md).

For tutorial 8's completed run, physical superresolution MSE was `1.5412078e-4` with spatial-gradient training and `3.1253930e-3` without it, a ratio of approximately **20.28**. Weighted normalized spatial-derivative MSE was `0.001637614` versus `0.735814538`. The corresponding temporal-derivative metric remained approximately `0.300` for both models; that derivative is unsupervised in the original loss. These measurements describe this single full run; saved predictions and weights are finite.

Chrome continued to show a rich-output JavaScript loading warning. Downloaded notebooks contain the image outputs, and exported PNGs and numerical arrays were reviewed; execution completed without Python errors. Browser rendering itself has not been repaired. Tutorial 8's original `vis_multiscale.png` has overlapping axes; its independently checked predictions are also shown in the readable `tutorial8.full.comparison.png` QA figure.

Detailed executed notebooks, images, checkpoints, and audit files are retained locally by the maintainer under `.codex/validation`; they are excluded from the Git repository and are not included in a fresh clone. The following paths identify those local records. Primary records are `tutorial-suite/cloud.full.final-audit.json`, `tutorial-suite/regression-final-summary.json`, `tutorial-suite/final-preservation.json`, and `tutorial-8/cloud.full.source-verification.json`. The executed tutorials 2–7 and plots are in `tutorial-suite/cloud-full`; 172 downloaded model artifacts have verified SHA256 hashes in `tutorial-suite/cloud-models`. Independent cloud artifact audits are `tutorials-2-3/cloud-full/audit.json`, `tutorials-4-5/cloud-final/audit.json`, `tutorial-7/cloud-full/audit.json`, and `tutorial-8/cloud-full/artifact-qa.json`.

The tutorial 7 cloud exports retain 90.005% kernel sparsity and genuine int8 weights; all 2,000 predictions from each checked format are finite, and the relative MSE calculations agree with the notebook. Tutorial 8 scientific metrics, original final plots, and the readable comparison are in `tutorial-8/cloud-full`. Additional local-run evidence remains in `tutorials-2-3`, `tutorials-4-5`, `tutorial-6`, and `tutorial-7`.
