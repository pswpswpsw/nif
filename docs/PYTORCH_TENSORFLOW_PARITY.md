# Controlled TensorFlow/PyTorch tutorial comparison

Status: in progress, 2026-10-23. Start with tutorial 3, then cover tutorials
1, 2, 4, 5, 6, 7 and 8. The existing full-execution acceptance remains useful
but is not evidence of matched cross-framework training.

## Independence and source preservation

Agent A owns implementation fixes and public development regressions in an
isolated container. Agent B owns separate private tests, fixtures and result
artifacts outside the repository and container mounts. B freezes each relevant
protocol and regression before A changes its corresponding implementation.
Only public requirements and sanitized failures reach A. The coordinator
reviews evaluator changes and retains their provenance. The original
TensorFlow notebooks, data and reference implementation remain preserved.

The TensorFlow reference is main at
`f8f9d31f4aa331b6e7dbc48758302c801441bcce`, with its documented compatibility
runtime based on `ad9113df14590189b12068b6b3fa7c8c8d9187d4`. Tutorial 1 uses
its own preserved `bootstrap_tutorial1.py` runtime; tutorials 2–8 use the
separate suite bootstrap. These are not interchangeable references. Baseline native
source is `220771c6fb5e567fb4e38fa5021253c3682b1cca`. Corrected comparisons
must identify both immutable source snapshots and environment versions.

## Controls and progression

1. Transfer actual tensors according to their mathematical roles, checking
   every model parameter and relevant buffer. A shared seed is insufficient.
   Transpose ordinary Dense/Linear kernels only where their layouts require it;
   retain generated ShapeNet packing and all biases.
2. Share normalized features, targets, area weights and exact batch row IDs,
   including remainder batches. Verify the actual consumed IDs. Match loss
   reductions, regularization, learning rates, optimizer state/epsilon,
   gradient centralization, dtype, device and determinism settings.
3. Compare the entire model's intermediate values, outputs, losses, input
   derivatives and every trainable parameter gradient. Compare centralized
   gradients, optimizer slots and parameter updates through short trajectories,
   including optimizer branch transitions. Use float64 CPU diagnostics and
   float32 checks under the actual training configuration.
4. Freeze meaningful full-training acceptance criteria before patched results.
   Account for the TensorFlow reference's numerical repeatability when choosing
   tolerances. Run original full workloads using actual notebook scientific
   code, with independently observed steps, rows, batch order and schedules.
   Evaluate both checkpoints with identical full-data and physical-field metrics.
   Short smoke runs do not replace this stage.
5. Reproduce each discovered implementation bug with a failing automated
   regression, record RED, apply the smallest fix, and record GREEN. Do not
   change architecture, epochs, objectives or learning rates merely to obtain
   passing comparisons. Revalidate affected earlier tutorials after shared
   implementation changes.

## Tutorial-specific checks

| Tutorial | Required comparison |
|---|---|
| 1 | AdaBelief trajectory and complete L-BFGS refinement; document distinct native/TFP line searches and evaluate common fields and convergence |
| 2 | Both tanh/Swish and SIREN complete fits with actual shared initial tensors |
| 3 | Full last-layer-parameterized model, weighted cylinder loss, actual GC/AdaBelief, 600 epochs and all 191,775 rows |
| 4 | Selected full Jacobian/Hessian tensors and their parameter differentiation, including zero/linear cases |
| 5 | Actual common TFRecord shards and pass/shard/row order, target shapes, loss broadcasting, all one million rows and two passes |
| 6 | All nine regularized fits; shared physical superresolution grid and correct parameter/coordinate order |
| 7 | Baseline fit, exact pruning schedule/mask-update semantics and complete sparse fit; compare quantized quality and export/restoration separately |
| 8 | Both complete fits, field/spatial/temporal derivative metrics, Sobolev objective, correct physical inverse scaling and superresolution |

Cross-framework floating-point training need not follow bitwise-identical
long trajectories. Numerical differences must be measured and explained;
end-to-end quality acceptance requires a predeclared TensorFlow-anchored
criterion, rather than a post-hoc threshold selected to accept a candidate.

## Current evidence

Historical runs used different actual weights, batch sequences and devices.
Tutorial 3's recomputed weighted normalized MSE was approximately 0.00184422
for native PyTorch, 0.00110383 for TensorFlow CPU and 0.000570633 for TensorFlow
Colab. These values expose a quality gap to investigate; they do not isolate
its cause. No controlled-training parity pass is claimed yet.

The first controlled diagnostic now passes 101 float64 checks covering all
19 trainable tensors, both output channels' input Jacobians, losses, raw and
centralized gradients, and ten updates with all AdaBelief moment slots.
An additional 24 checks compare ParameterNet coefficients, spatial bases,
complete input Hessians and their parameter gradients. The float64 reference
requires a labeled diagnostic-only initializer dtype correction and promotion
of one incorrectly float32 HyperLinear variable; the original reference is
preserved. Float32 training uses the unmodified compatibility reference.
Three of 100 initial float32 trace checks fail on input Jacobians after
rectification, with maximum absolute difference about 3.6e-5. Equal-gradient
probes localize update differences near 1e-7; this does not yet establish an
algorithm defect or explain the full-training quality gap.

Two independent source-backed discrepancies have separate frozen failing
regressions and native fixes, with independent short-trajectory validation:

- The migrated Adam fits use PyTorch's epsilon after bias correction, whereas
  the TensorFlow reference uses epsilon after the square root of raw variance.
  Equal epsilon values do not produce the same update. Matched float64 moment
  slots agree while parameter updates differ, isolating the algebraic issue.
- Tutorial 7 applies its first mask too early and uses a differentiable mask
  parameterization that blocks masked gradients. The reference instead masks
  the variable before forward execution, permits optimizer regrowth, refreshes
  at optimizer-step boundaries, and remasks at epoch end.

The native Adam tutorials now use `ReferenceAdam` with the reference epsilon
convention. Tutorial 7 uses a native assignment-based pruning controller with
the original step schedule. Complete learning and quantized-quality acceptance
for those tutorials remains pending.

Tutorial 1 has a separate reference-behavior issue: its current Colab bootstrap
does not activate the modern private gradient hook. Assigning the old public
`get_gradients` method therefore leaves GC inactive in actual TensorFlow fit.
The suite bootstrap activates GC, which tutorial 3 requires. Tutorial 1's
primary comparison uses its actual current Colab behavior. An independently
observed fit/update regression failed before the native default was corrected
to inactive GC; the corrected eight-update CPU pilot passes all 248 checks.
An active-GC reference is retained only as a separately labeled intended-GC
diagnostic. Reference code will not be changed to make a candidate pass.

### First complete controlled tutorial 3 pair

Both frameworks completed 600 epochs, 112,800 updates and 115,065,000 actual
row visits on the same GPU, starting from all 19 identical parameter tensors.
The independently observed consumption digest was identical. Native candidate
011 has manifest
`6eeef3ba84f4d55fa8f4c296a687c666bd29ef8874efb8dd698bdec31ce258c3`.

| Final metric | TensorFlow | Native PyTorch | Fixed criterion | Result |
|---|---:|---:|---|---|
| Weighted normalized MSE | 0.0014905620 | 0.0014670580 | Native <= 1.10 TF + 1e-6 | Pass |
| Physical u MSE | 0.0356365000 | 0.0353955722 | Native <= 1.10 TF + variance floor | Pass |
| Physical v MSE | 0.0758665733 | 0.0755009848 | Native <= 1.10 TF + variance floor | Pass |
| Paired u field NRMSE | — | 0.10445044 | <= 0.05 | Fail |
| Paired v field NRMSE | — | 0.08127170 | <= 0.05 | Fail |
| Paired u input-Jacobian relative RMS | — | 0.23453748 | <= 0.10 | Fail |
| Paired v input-Jacobian relative RMS | — | 0.17074924 | <= 0.10 | Fail |

This candidate fails full scientific acceptance despite its similar aggregate
MSE. Metrics use every original mesh row after inverse normalization into the
source data's units; their SI interpretation is not established here. The field
denominator is the TensorFlow field energy centered at the target mean; the
Jacobian comparison includes all three input coordinates per output. This
combined source-scaled norm depends on the chosen coordinate units; it is a
fixed array-comparison criterion, not a unit-invariant physical distance.

A separately frozen shared-gradient regression localizes a float32 arithmetic
discrepancy: the TensorFlow AdaBelief reference evaluates bias powers and
rectification scalars in parameter dtype, whereas the native implementation
evaluates them in Python double precision. At step 6 this changes the
rectification coefficient by about 0.185%. Moment slots match bitwise in that
diagnostic, and all float64 controls pass. The native scalar correction now
passes all 4,608 independently frozen checks against the unchanged TensorFlow
trace. In that controlled-hyperparameter fixture, all float32 parameter and
moment-slot comparisons are bitwise equal. Native candidate 012 has manifest
`cd0cd078e9fc147d6d9627ad702bcc74bfef15b0551a3f41f2d12e42bd86de62`.
This local result does not establish that correcting scalar arithmetic will
eliminate the full-training differences. Default-constructor hyperparameter
representation is also checked separately from effective-hyperparameter
alignment.

That separate nominal-default control found a second arithmetic discrepancy:
the native moment coefficients formed Python `1 - beta` before casting, while
the legacy TensorFlow reference subtracts after casting beta to parameter dtype.
The corrected native implementation now passes all 2,304 original default-path
numerical checks, with all 1,280 float32 moment comparisons bitwise equal.
Nominal and explicitly materialized float32-beta native controls are also
bitwise equal. The original 4,608 effective-hyperparameter checks remain green.
Candidate 014 has manifest
`8dfe4545c7f706d5f10233266239883a6f5f3ac54acf9b625c89d2b1124c3143`.
The native float64 extension retains caller beta precision; TensorFlow's stored
hyperparameter representation is explicitly matched in its separate controls.

An all-None-gradient counter bug was separately reproduced and fixed. All 92
independent behavior checks and 56 TensorFlow numerical comparisons now pass;
real zero gradients still advance training, and partial missing gradients remain
supported. These local fixes do not replace full-training acceptance.

The failed candidate and thresholds remain preserved. TensorFlow seed 42's
exact-configuration repeat has completed all 600 epochs: all 6,722 numerical
comparisons are bitwise equal across checkpoints, optimizer state, histories,
full-domain fields and Jacobians. All 10,168 repeat-audit checks pass, including
the actual batch sequence and learning rates. This verifies repeatability for
this configuration and seed, without establishing statistical equivalence.
Candidate 014's nominal-default native run has also completed the original
600 epochs, 112,800 updates and 115,065,000 row visits on the same GPU. All
19 initial parameter tensors, the actual consumed row-ID digest and the
learning-rate schedule match the TensorFlow reference. The first 32 updates
pass all 1,824 frozen trace checks. The full result still fails four of the
seven original quality criteria:

| Final metric | TensorFlow | Corrected native 014 | Fixed criterion | Result |
|---|---:|---:|---|---|
| Weighted normalized MSE | 0.0014905620 | 0.0015494086 | Native <= 1.10 TF + 1e-6 | Pass |
| Physical u MSE | 0.0356365000 | 0.0370344560 | Native <= 1.10 TF + variance floor | Pass |
| Physical v MSE | 0.0758665733 | 0.0799414370 | Native <= 1.10 TF + variance floor | Pass |
| Paired u field NRMSE | — | 0.10093719 | <= 0.05 | Fail |
| Paired v field NRMSE | — | 0.08777805 | <= 0.05 | Fail |
| Paired u input-Jacobian relative RMS | — | 0.24066280 | <= 0.10 | Fail |
| Paired v input-Jacobian relative RMS | — | 0.20095327 | <= 0.10 | Fail |

The weighted MSE is approximately 3.95% higher than TensorFlow's, within its
predeclared limit. That aggregate result does not establish agreement of the
learned fields or their derivatives. The optimizer fixes pass their separate
regressions, but have not eliminated long-training divergence. The cause of
the remaining divergence is under investigation; numerical sensitivity alone
has not been established as its explanation. Both full candidates' failures
and the original thresholds remain retained.

The corrected run's per-coordinate Jacobian relative RMS, listed in order
`(time, x, y)`, is approximately `(0.2803, 0.3377, 0.1787)` for u and
`(0.1843, 0.2105, 0.3276)` for v. These dimensionless ratios report each
coordinate separately; no new acceptance thresholds are introduced.

A second paired initialization, seed 1729, has completed the same 600 epochs,
112,800 updates and 115,065,000 row visits in both frameworks. Its first
32-update comparison passes all 1,824 checks. Final results again pass all
three MSE criteria and fail all four original field/Jacobian criteria:

| Metric | TensorFlow | Native 014 | Result |
|---|---:|---:|---|
| Weighted normalized MSE | 0.0006445052 | 0.0006483124 | Pass, approximately 0.59% higher |
| Physical u MSE | 0.0241680964 | 0.0230065779 | Pass |
| Physical v MSE | 0.0262528010 | 0.0263212329 | Pass |
| Paired field NRMSE (u, v) | — | (0.11195700, 0.08216504) | Both fail |
| Paired input-Jacobian relative RMS (u, v) | — | (0.27191703, 0.17880680) | Both fail |

The third paired initialization, seed 2718, has also completed the original
full workload in both frameworks:

| Metric | TensorFlow | Native 014 | Result |
|---|---:|---:|---|
| Weighted normalized MSE | 0.0014851422 | 0.0014407756 | Pass, approximately 2.99% lower |
| Physical u MSE | 0.0396937056 | 0.0381359566 | Pass |
| Physical v MSE | 0.0697421596 | 0.0689458198 | Pass |
| Paired field NRMSE (u, v) | — | (0.12295959, 0.09412079) | Both fail |
| Paired input-Jacobian relative RMS (u, v) | — | (0.28976927, 0.20127615) | Both fail |

All three original paired initializations are now complete: six runs, each
with 600 epochs and 112,800 updates. All 5,472 short-prefix tensor checks
pass. The full results pass nine MSE records and fail twelve field/derivative
records out of 21. The exact initial tensors, actual batch consumption and
actual learning rates are verified. Close aggregate MSE does not replace the
failed criteria or establish statistical equivalence. Common-state backward
and optimizer localization continues; the remaining learning tutorials are
not yet fully accepted.

An additional zero-update diagnostic transfers all 19 final tensors in BOTH
directions and evaluates all 191,775 rows on the original GPU. Same-framework
restore/clone controls, exact tensor readback and source/input guards are
included. For identical weights, cross-framework field relative RMS is about
6.8e-7 to 7.0e-7 and complete-Jacobian relative RMS about 1.1e-6. All 12
aggregate diagnostic comparisons pass. The older elementwise forward gate
still fails for 13 unique elements out of 383,550 in the TF-weight-to-native
direction, with maximum normalized forward difference approximately 3.76e-6;
three failed comparisons refer to that same array. These failures and small
native same-framework Jacobian rerun differences remain recorded. This
supports close same-weight function agreement, not bitwise equivalence or
an explanation of the training divergence. The original full-training failures
stand.

A separate CPU diagnostic restores both frameworks' states at epochs 1, 25,
200 and 400 and tests both a full batch and the original remainder batch.
All 16 cases pass 1,600 original forward/backward/update checks. Replaying the
actual TensorFlow gradients and pre-update state through the native optimizer
passes all 912 numerical checks; all 608 moment arrays are byte-identical.
Of 304 parameter arrays, 92 differ in storage bytes, with maximum difference
2.98e-8, within the original numerical limits. Same-framework restore and
observer controls pass. These CPU results do not establish GPU optimizer
agreement or identify the cause of long-training divergence; the corresponding
GPU diagnostic remains pending.

Full paired-training acceptance for the other learning tutorials, including
their remaining initializations, is pending. Tutorial 4's derivative-only
result is recorded below.

### Tutorial 4 derivative deliverables

The archived tutorial 4 has no optimizer training workload. Its actual
derivative deliverables pass CPU float64 and same-GPU float32 comparisons with
identical 25 initial parameter scalars and ten four-coordinate input rows.
All 52 float32 checks pass, including complete output Jacobians and Hessians,
derivative-loss parameter/input gradients, the delivered scientific cells,
checkpoint reload and figure inspection. GPU maximum absolute differences for
output/Jacobian/Hessian are approximately 5.96e-8/8.94e-8/5.96e-8.

The native notebook's additional Adam update is a supplemental demonstration;
the archived notebook supplies no TensorFlow optimizer trajectory for it.
Six supplemental float64 optimizer representation conflicts remain recorded
separately and are not represented as derivative failures or as optimizer
parity passes.

### Remaining learning tutorials: readiness, not full acceptance

The following are finite controls. They do not stand in for the original
full workloads or the remaining paired initializations.

| Tutorial | Current finite evidence | Still required |
|---|---|---|
| 1 | Corrected nominal-default 32-update comparison passes 1,312 checks; the full-entry two-epoch coupling passes 248 checks and actual postprocessing cells | Full AdaBelief fit, each framework's own L-BFGS refinement, final common metrics and figures |
| 2 | Both network variants pass 3,328 short-trace checks; fresh disk restoration followed by two actual updates passes 400 strict storage-byte checks | Same-GPU checks, both full fits and remaining initializations |
| 5 | Actual million-row TFRecord corpus is fixed; the 158-update pilot passes 416 comparisons; current-Colab target shapes/listing and disk restoration followed by two actual next-shard updates are verified | Same-GPU checks, both complete corpus passes and final cells |
| 6 | All nine regularization cases' actual raw/used gradients pass; fresh disk restoration and two updates pass 1,782 strict storage-byte checks | Four small post-momentum discrepancies remain recorded, plus three inherited pre-momentum differences; full nine-case fits and final aggregate/gallery cells remain pending |
| 7 | Full-architecture baseline and native pruning boundary controls exist | TF pruning observer neutrality currently fails and blocks full release; pruned disk continuation, own-baseline/full-pruning fits and real export/quantization quality remain pending |
| 8 | Both 32-update fits pass 3,584 numerical checks; disk continuation and observer controls pass; actual Colab postprocessing order reproduces all six prior common-metric arrays byte-for-byte | Same-GPU checks, both full fits, remaining initializations and final figures |

The tutorial 7 observer failure is under investigation and is not an
established product defect. Tutorial 8's replay difference is localized by a
controlled same-TensorFlow experiment: calling the common direct-inference
helper before the original eight Colab prediction calls changes cached
Jacobian execution paths. The differences occur only in the spatial
derivative; field and time-derivative channels match. Performing the actual
Colab postprocessing calls first reproduces the prior arrays exactly. The
full runner already uses that source-faithful order, so no model, full-runner
or tolerance change was needed. The specific kernel cause remains unknown.
All earlier direct-first and cold-replay failures remain retained as
execution-path controls. No full learning tutorial is marked accepted from
the finite controls in this table.

### Public regression portability

The scientific package and notebooks are unchanged by the separate public-test
publication fixes. A clean public-only Linux CPU projection passes 416 tests
with two unavailable-CUDA skips. An independent native Windows Python 3.13.11 /
PyTorch 2.14.1 CPU projection passes 414 tests with four explicit skips: two
unavailable-CUDA checks and two exact RNG fingerprint checks restricted to
their recorded Linux build. Missing public fixtures/provenance and modified
models, numeric goldens or archives are still rejected. The original failures
and bounded source successors are preserved in the public development records.
These regression results do not constitute full scientific parity acceptance.
