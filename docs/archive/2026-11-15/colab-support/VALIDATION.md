# Colab tutorial validation

Validated on **2026-10-21** with full original workloads. The original notebooks, Markdown, code comments, data sizes, model settings and training schedules are preserved; the repaired notebooks are separate `*_colab.ipynb` copies.

| Tutorial | Colab cells passed | Original workload |
| --- | ---: | --- |
| 1 | 27 / 27 | 5,000 epochs and L-BFGS rounds=200, max_iter=1000 |
| 2 | 9 / 9 | Two 5,000-epoch fits |
| 3 | 14 / 14 | 600 epochs, 191,775 cylinder-flow samples |
| 4 | 10 / 10 | Jacobian and Hessian examples |
| 5 | 9 / 9 | One million rows, 100 TFRecord shards, two complete passes |
| 6 | 21 / 21 | Nine 3,000-epoch fits; two empty cells retained |
| 7 | 28 / 28 | 5,000 baseline and 50,000 pruning epochs, export and TFLite inference |
| 8 | 21 / 21 | Two 50,000-epoch fits, derivatives and superresolution |

Tutorial 1 ran directly in a hosted Colab CPU runtime, tutorial 8 directly on a hosted T4 with Run all, and tutorials 2–7 in fresh Jupyter kernels on Colab T4 using the complete delivered sources. Tutorials 2–8 passed **112 executable cells**; tutorial 1 contributes another 27. No Python error outputs were reported. Saved figures and scientific outputs were reviewed independently.

Tutorial 7's cloud exports retain **90.005% kernel sparsity** and genuine int8 weights. All 2,000 predictions from each checked format are finite. Tutorial 8's physical superresolution MSE was `1.5412078e-4` with spatial-gradient training versus `3.1253930e-3` without it in this full run; its temporal derivative remains unsupervised, as in the original experiment. All 172 collected cloud model artifact hashes were verified.

## Publication checks

The original focused suite passed 25 regression tests. Before publication, all **29 tests in eleven isolated processes passed**, including tutorial 1 checks and a cloud-payload test that generates its own inputs in a temporary directory. The new Python files also pass the repository's pre-commit hooks.

Inherited execution metadata and cached TensorBoard resources were removed from the copies. LF line endings preserve source hashes across platforms. Publication formatting changes only the bootstrap's layout and the order of independent standard-library imports. Its normalized AST matches the validated bootstrap; every other cell source is unchanged. Full training was not repeated for formatting. The JSON summary records both the pre-publication and published notebook hashes, the full-run and published bootstrap hashes, pinned packages, per-tutorial timing and validation mode. Tutorial 1's earlier comment restoration also retained the executable AST of its completed full run.

[Machine-readable results and hashes](validation-summary.json) · [Tutorial 1 guide](README.md) · [Tutorials 2–8 guide](README-tutorials-2-8.md)

Detailed executed notebooks, model files, logs and screenshots remain in the maintainer's ignored `.codex/validation` directory. They are not part of a fresh clone. The checked-in builders, runner and regression tests support reproducing the results; Colab runtimes can change after the validation date.

## Known display limits

Chrome reported a rich-output JavaScript loading warning even though computations finished. Saved PNGs and numerical outputs passed separate inspection; browser rendering remains unresolved. Tutorial 8's callback-generated `vis_multiscale.png` has overlapping axes, while its final derivative and superresolution plots are readable.
