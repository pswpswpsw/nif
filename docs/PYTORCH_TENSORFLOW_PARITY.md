# TensorFlow/PyTorch tutorial comparison

**Current review status:** the user accepted all eight released PyTorch tutorials
on 2026-10-25–27 after inspecting their results, figures and absolute errors.
The table below is the current summary of those reviews. It preserves the
automatic numerical criteria that remained unmet.

The original TensorFlow notebooks define the scientific cases. Comparisons use
actual shared weights, declared batch sequences, normalization and precision
where specified by each protocol. Original-style figures preserve coordinate
order, evaluation grids, plotting methods and error definitions. The complete
[dated comparison record](archive/2026-11-15/PYTORCH_TENSORFLOW_PARITY.md#controls-and-progression)
retains candidate histories, measurement tables, source hashes, figure paths and
the exact conditions of every experiment.

## Released tutorial results

| Tutorial | User review | Recorded numerical result and remaining differences |
|---|---|---|
| 1 | Accepted 2026-10-25 | AdaBelief 19/27 and repaired L-BFGS 21/27 automatic checks. L-BFGS paired-field checks pass; six L-BFGS MSE failures, eight AdaBelief MSE failures and initial-gradient/convergence limits remain. [Complete matrix](archive/2026-11-15/PYTORCH_TENSORFLOW_PARITY.md#tutorial-1-repaired-optimizer-completes-the-matrix-with-failures-retained). |
| 2 | Accepted 2026-10-25 | Main quality checks 24/30, retaining six SIREN paired-field failures and nine separate final-weight check-group failures. Fourteen declared-state disk-continuation cases and their observer controls pass. [Paired runs and controls](archive/2026-11-15/PYTORCH_TENSORFLOW_PARITY.md#tutorial-2-all-three-complete-paired-initializations). |
| 3 | Accepted candidate-016, 2026-10-25 | Nine MSE checks pass; twelve paired-field/Jacobian checks fail. The loss-curve steps follow the declared learning-rate schedule. [User review and candidate evidence](archive/2026-11-15/PYTORCH_TENSORFLOW_PARITY.md#tutorial-3-user-acceptance). |
| 4 | Accepted 2026-10-25 | Saved derivative tensors, figures and absolute errors were reviewed. The source tutorial has no training workload; six supplemental float64 optimizer-update differences are recorded separately. [Derivative deliverables](archive/2026-11-15/PYTORCH_TENSORFLOW_PARITY.md#tutorial-4-derivative-deliverables). |
| 5 | Accepted 2026-10-26 | Three complete paired streaming runs, the declared-state TensorFlow repeat and original end-cell checks pass the local protocol. [Full streamed-data comparisons](archive/2026-11-15/PYTORCH_TENSORFLOW_PARITY.md#tutorial-5-three-complete-controlled-pairs-and-reference-repeat). |
| 6 | Accepted 2026-10-27 | All 63 full fits, 63 disk controls and 63 saved-final evaluations completed. Quality checks pass 57/81 and retain 24 failures; earlier seven CPU and 24 GPU prefix failures remain. TensorFlow repeat checks cover 42,322 retained arrays, with 63 common-field figures inspected. [Regularization controls](archive/2026-11-15/PYTORCH_TENSORFLOW_PARITY.md#tutorial-6-gpu-controls-retain-cross-framework-failures). |
| 7 | Accepted 2026-10-27 | Seven baselines, eight complete diagnostic pruning runs, five comparisons and 16 PNG-path inspections completed. Primary-pruning baseline criteria remain unmet; three own-baseline pairs and a common-start supplement retain numerical failures. TensorFlow repeat checks cover 4,770 retained arrays. [Pruning comparisons](archive/2026-11-15/PYTORCH_TENSORFLOW_PARITY.md#tutorial-7-complete-baselines-fail-before-primary-pruning). |
| 8 | Accepted 2026-10-27 | Seven full training lanes completed. Three complete pairs pass 36/60 quality checks and retain 24 failures; all 9,792 prefix checks pass. Strict TensorFlow-repeat checks pass, while the original aggregate and scoped CPU final-function diagnostic retain failures. [Sobolev comparisons](archive/2026-11-15/PYTORCH_TENSORFLOW_PARITY.md#tutorial-8-controls-and-complete-matched-training). |

Acceptance records the user's judgment of the displayed cases. The numerical
results above describe the measured agreement between frameworks under each
comparison's conditions. Full workload execution, numerical agreement and user
acceptance are separate parts of the evidence.

## Current scientific definitions

The [native API guide](PYTORCH_API.md) describes the implemented definitions.
The [tutorial semantics review](TUTORIAL_SEMANTICS_REVIEW.md) records repairs to
the ports. In particular, T7 now retains the original squared matrix spectral
norm ratio (`ord=2`), original six-panel comparison and inference-artifact size
measurement. Earlier Frobenius-based reports remain identifiable in the dated
history; the [T7 restoration record](TUTORIAL_SEMANTICS_REVIEW.md#f-t7-三项原定义恢复及独立验收)
describes the accepted correction.

## Execution and hosted Colab evidence

The full workload and checkpoint/export histories are retained in the
[original execution report](archive/2026-11-15/PYTORCH_VALIDATION.md#full-tutorial-execution)
and the [hosted Colab records](archive/2026-11-15/PYTORCH_TENSORFLOW_PARITY.md#hosted-colab-tutorial-1-full-execution-separate-from-parity).
Hosted execution is recorded for T1–T8. T6 and T7 retain original notebook
outputs but their raw runtime archives were not restored; output-JavaScript
loading problems and how metrics/figures were inspected are documented with
those runs. These records retain their original source commits and environments.

[Tutorial 9 and the self-scaled experiments](SELF_SCALED_OPTIMIZERS.md) have
separate Linux FULL, Colab FULL and shortened execution receipts. Their optimizer
comparison is an experimental extension of T1.

## Historical records

- [Complete comparison history](archive/2026-11-15/PYTORCH_TENSORFLOW_PARITY.md): all candidate results, failed criteria, controls, source hashes and figure paths.
- [Development log](archive/2026-11-15/PYTORCH_CANDIDATE_REPORT.md): implementation iterations and RED/GREEN records.
- [Execution snapshot](archive/2026-11-15/PYTORCH_VALIDATION.md): original package and full-workload execution acceptance.
- [Archive index and byte-level provenance](archive/2026-11-15/README.md): source revision and checksums of the unchanged records.
