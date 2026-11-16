# Tutorial 1: Colab compatibility copy

For the remaining notebooks, see [Tutorials 2–8: Colab compatibility copies](README-tutorials-2-8.md).

Use [`../1_simple_1d_wave_colab.ipynb`](../1_simple_1d_wave_colab.ipynb) in a fresh Colab session and run from the first cell. The original `1_simple_1d_wave.ipynb` is preserved. The copy retains all 27 code cells, the original 5,000 training epochs, and L-BFGS settings (`rounds=200, max_iter=1000`).

The compatibility copy defaults to CPU, matching the hosted runtime used for validation.

The original 51-cell order, all 24 Markdown cells, and all 11 original code comments are preserved verbatim. Added introductory prose and code comments were removed from the notebook; compatibility explanations remain in this file.

[Open tutorial 1 in Colab](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/1_simple_1d_wave_colab.ipynb). Downloaded validation snapshots matched the same 27 code cells, allowing Colab's trailing-whitespace normalization.

Publication formatting changes only the bootstrap layout and independent standard-library import order. Its normalized executable AST matches the validated source; every other cell is unchanged. See [publication validation](VALIDATION.md).

## Environment

A live Colab **CPU** runtime was inspected on 2026-10-21: Ubuntu 24.04.5 LTS, Python **3.13.16**. The relevant package versions below agree with the official CPU snapshot at [`a0660eef62a8bb0d96480e10b2389ffd75be012b`](https://github.com/googlecolab/backend-info/tree/a0660eef62a8bb0d96480e10b2389ffd75be012b).

| Package | Live Colab CPU | Pinned NIF environment |
| --- | --- | --- |
| TensorFlow / tf-keras | 2.21.0 / 2.21.0 | same |
| Keras | 3.13.2 | same; TensorFlow uses legacy tf-keras |
| NumPy / SciPy | 2.1.3 / 1.16.3 | same |
| TensorFlow Probability | 0.25.0 | same |
| TensorBoard / Matplotlib | 2.21.0 / 3.10.0 | same |
| TensorFlow Model Optimization | absent | 0.8.1 |
| absl-py | 2.0.0 | 1.4.0 |

NIF imports TensorFlow Model Optimization (TFMOT). Version 0.8.1 supports NumPy 2 and declares `absl-py~=1.2` (below 2), so adding it requires the explicit absl-py change. See [`requirements-colab.txt`](requirements-colab.txt) for the complete relevant package pins and [`requirements-qa.txt`](requirements-qa.txt) for local notebook execution tools.

The local debugging environment uses **Windows CPU and Python 3.13.11**, with the pinned package versions above. It is a matching package baseline, not an identical Linux image or Python patch release. Native Windows TensorFlow 2.21 does not use CUDA; Linux/WSL2 is needed for a supported local GPU route. PyTorch is not used by this tutorial.

## Compatibility changes

The first cell embeds [`bootstrap_tutorial1.py`](bootstrap_tutorial1.py). It sets `TF_USE_LEGACY_KERAS=1` before importing TensorFlow, installs missing or different pinned packages, and installs NIF from commit [`ad9113df14590189b12068b6b3fa7c8c8d9187d4`](https://github.com/pswpswpsw/nif/tree/ad9113df14590189b12068b6b3fa7c8c8d9187d4) into a temporary runtime directory. Only that temporary NIF installation receives these compatibility edits:

- Use the current TensorFlow distribution-context import.
- Keep custom optimizers on the legacy Keras optimizer API.
- Replace removed NumPy `np.product` calls with `np.prod`.

The notebook also handles CPU-only runtimes and uses a legacy dummy Adam optimizer when restoring the checkpoint. A prediction equality check verifies checkpoint restoration. The training optimizer remains AdaBelief. If an already imported package needs a version change, restart the session; preloaded NumPy with the correct version is accepted.

## Run locally

Prerequisites: PowerShell 7, `uv`, Git, and network access. From the cloned repository root in PowerShell:

```powershell
& '.\tests\run_tutorial1_local.ps1'
```

This creates or uses `%USERPROFILE%\.cache\nif-colab-qa\colab-matched`, installs the pinned dependencies, then executes **every cell with full training** in an isolated working directory. The executed notebook and cell event log are saved under `.codex/validation/tutorial1-colab` in your checkout.

For interactive local debugging, select `%USERPROFILE%\.cache\nif-colab-qa\colab-matched\Scripts\python.exe` as the notebook interpreter in your editor, then run the compatibility copy from its first cell.

For a quick execution check:

```powershell
& '.\tests\run_tutorial1_local.ps1' -Smoke
```

`-Smoke` changes only the in-memory QA execution copy to two epochs and one short L-BFGS round. It does not shorten the delivered notebook or edit the original. Rebuild the compatibility copy after changing its bootstrap using `python tests/build_tutorial1_colab_copy.py` from the repository root.

## Validation status

The [public validation summary](VALIDATION.md) records the completed runs. Detailed executed notebooks, images, logs, and environment manifests cited below are retained locally by the maintainer under `.codex/validation/tutorial1-colab`; they are excluded from the Git repository and are not included in a fresh clone.

The local environment with the pins above completed all **27 code cells with full training** in **879.19 seconds** (`full.colab-matched.executed.ipynb`): the original 5,000 epochs and L-BFGS settings were retained, the checkpoint prediction equality assertion passed, and model export was produced. The generated prediction and error plots were visually inspected. The PowerShell entry point also passed its smoke run in 17.60 seconds. An earlier compatibility environment passed all 27 cells in 518.83 seconds.

The hosted latest **Colab CPU runtime also completed all 27 code cells**, with live execution numbers 1–27 and no Python error outputs. Its source matches the locally tested copy, checkpoint 4999 was restored with matching predictions, and the final prediction PNG was exported and visually checked. Evidence: `cloud.full.executed.download.ipynb`, `cloud.full.executed.live-counts.json`, `cloud.full.executed.ax.txt`, and `cloud.vis_fine_tuned.png`. The downloaded file retained null execution counters for some cells; the live UI counters are recorded separately rather than altering the downloaded evidence.

The subsequent comment restoration changed only comments, a module description string, and the extra introductory Markdown cell. All 27 code cells retain the same executable AST as the completed full run; the three compatibility checks passed again. Full training was not repeated for this presentation change. The comparison is recorded in `comment-restoration.json`, with the previous copy retained under `before-comment-restoration`.

The live Chrome page repeatedly reported that it could not load the JavaScript needed to display rich output, leaving blank output frames. The generated PNGs are present in the downloaded notebook. **Browser display remains unresolved**; the notebook's computational run passed. See the [Colab FAQ on output frames and third-party cookies](https://research.google.com/colaboratory/intl/en-GB/faq.html).

Locally retained environment records include `colab-live-baseline.json`, `colab-official-baseline.json`, `local-colab-matched-environment.json`, and `local-colab-matched.freeze.txt`. Colab images change, so inspect the first-cell version output on future runs.

Additional primary references: [official CPU package snapshot](https://github.com/googlecolab/backend-info/blob/a0660eef62a8bb0d96480e10b2389ffd75be012b/pip-freeze.txt), [TFMOT 0.8.1 release](https://github.com/tensorflow/model-optimization/releases/tag/v0.8.1), [legacy Keras configuration](https://keras.io/getting_started/#tensorflow--keras-2-backwards-compatibility), and [TensorFlow platform support](https://www.tensorflow.org/install/pip#windows-native).
