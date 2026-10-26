# Controlled TensorFlow/PyTorch tutorial comparison

Status: execution queue closed, 2026-10-25; recorded numerical failures remain.
The comparison started with tutorial 3, then covered tutorials 1, 2, 4, 5, 6,
7 and 8. Hosted full execution and controlled numerical acceptance are
separate evidence.

Scope terminology: in the controlled-comparison sections, "original/frozen
criteria" means the comparison protocol set during this migration, not tests
shipped with the original TensorFlow notebooks. Refined grids, additional
derivative checks and repeat/restore controls may extend the original teaching
demonstration. The archived original, the repaired Colab reference and the
native implementation are distinct baselines. Their confirmed semantic and
presentation differences are listed in the independent
[tutorial semantics review](TUTORIAL_SEMANTICS_REVIEW.md). Matching the repaired
reference alone does not establish preservation of every original definition.

The three tutorial-7 output/export differences identified by that review have
now been restored to the archived TensorFlow definitions, with implementation
and independent acceptance separated. See the repair record in
[section F](TUTORIAL_SEMANTICS_REVIEW.md#f-t7-三项原定义恢复及独立验收).
This scoped repair does not relabel the historical full-training failures.

| Tutorial | Current local controlled-comparison result |
|---|---|
| 1 | User accepted the current L-BFGS test on 2026-10-25. Recorded automatic results remain AdaBelief 19/27 and repaired L-BFGS 21/27, with all paired-field checks passing and six L-BFGS MSE failures retained. Complete own-state and shared-start comparisons are recorded below. |
| 2 | User accepted the current results on 2026-10-25 after reviewing absolute errors. Recorded automatic results remain 24/30 main quality checks, with six SIREN paired-field failures; the separate final-weight diagnostic retains nine failed check groups. All fourteen declared-state disk-continuation cases and their observer controls pass. |
| 3 | Three corrected complete pairs pass all nine MSE gates and fail all twelve paired-field/Jacobian gates. |
| 4 | Original derivative deliverables pass; the reference has no training workload. |
| 5 | Three complete pairs, exact declared-state reference repeat and original end-cell checks pass the local protocol. |
| 6 | All 63 full fits, 63 disk controls and 63 saved-final evaluations complete. The full quality matrix passes 57/81 flags and retains 24 failures; earlier CPU/GPU prefix failures remain. TF repeat retained-array checks and 63 common-field figure inspections complete. |
| 7 | Baseline quality remains insufficient for primary pruning acceptance. All eight separately labeled full diagnostic pruning runs and five comparisons complete; three own-baseline pairs and the common-start supplement retain numerical failures. Export execution and TF repeat do not grant whole-tutorial acceptance. |
| 8 | All seven original full lanes complete. Three complete pairs pass 36/60 quality criteria and fail 24; all 9,792 prefix checks pass. Strict TF-repeat retained-state/stream checks pass. The original aggregate and the scoped CPU final-function diagnostic retain failures. |

These local results use preprovisioned environments. Separate hosted Colab
execution records for tutorials 1–8 appear below, including each notebook's
artifact and rendering limits. Those hosted runs use their own initialization
and batches; controlled cross-framework acceptance remains as reported above.

Tutorials 1–8 now have separate hosted-Colab code-execution records,
described below. The records for
tutorials 6 and 7 retain original notebook outputs; their complete raw runtime
archives are unavailable. Their evidence limits are stated separately.

## Independence and source preservation

Agent A owns implementation fixes and public development regressions. The
initial work used an isolated container; after Docker IPC failures, the user
authorized separate agent contexts without Docker. A is instructed to access
only the public repository and tools, and not to discover or read B's private
materials. This later separation relies on those instructions rather than a
container access boundary. Agent B owns private tests, fixtures and result
artifacts outside the repository. B freezes each relevant protocol and
regression before A changes its corresponding implementation. Only public
requirements and sanitized failures reach A. The coordinator reviews evaluator
changes and retains their provenance. The original TensorFlow notebooks, data
and reference implementation remain preserved.

The earlier documentation phase corrected eight Markdown cells in native
tutorials 1, 5 and 8. At that phase, all 44 code cells, outputs, notebook
metadata and cell order stayed unchanged. The later tutorial-1 optimizer
repair below changes only its two L-BFGS code cells and the necessary optimizer
name in one teaching note. Original archives remain preserved; scientific
snapshots retain their own whole-file hashes and identify their source phase.

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
| 2 | Both Swish and SIREN complete fits with actual shared initial tensors |
| 3 | Full last-layer-parameterized model, weighted cylinder loss, actual GC/AdaBelief, 600 epochs and all 191,775 rows |
| 4 | Selected full Jacobian/Hessian tensors and their parameter differentiation, including zero/linear cases |
| 5 | Actual common TFRecord shards and pass/shard/row order, target shapes, loss broadcasting, all one million rows and two passes |
| 6 | All nine regularized fits; shared physical superresolution grid and correct parameter/coordinate order |
| 7 | Baseline fit, exact pruning schedule/mask-update semantics and complete sparse fit; compare quantized quality and export/restoration separately |
| 8 | Both complete fits; field plus spatial-derivative supervision, temporal derivative as monitor only; correct physical inverse scaling and superresolution |

Cross-framework floating-point training need not follow bitwise-identical
long trajectories. Numerical differences must be measured and explained;
end-to-end quality acceptance requires a predeclared TensorFlow-anchored
criterion, rather than a post-hoc threshold selected to accept a candidate.

## Current evidence

Historical runs used different actual weights, batch sequences and devices.
Tutorial 3's recomputed weighted normalized MSE was approximately 0.00184422
for native PyTorch, 0.00110383 for TensorFlow CPU and 0.000570633 for TensorFlow
Colab. These historical values expose a quality gap to investigate; they do
not isolate its cause or establish controlled-training agreement.

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
remains pending for tutorials 2, 6, 7 and 8; tutorial 5's completed local
matched-training evidence is reported below.

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
rectification coefficient by about 0.185%. Moment slots are numerically equal in that
diagnostic, and all float64 controls pass. The native scalar correction now
passes all 4,608 independently frozen checks against the unchanged TensorFlow
trace. In that controlled-hyperparameter fixture, all float32 parameter and
moment-slot comparisons are numerically equal. Native candidate 012 has manifest
`cd0cd078e9fc147d6d9627ad702bcc74bfef15b0551a3f41f2d12e42bd86de62`.
This local result does not establish that correcting scalar arithmetic will
eliminate the full-training differences. Default-constructor hyperparameter
representation is also checked separately from effective-hyperparameter
alignment.

That separate nominal-default control found a second arithmetic discrepancy:
the native moment coefficients formed Python `1 - beta` before casting, while
the legacy TensorFlow reference subtracts after casting beta to parameter dtype.
The corrected native implementation now passes all 2,304 original default-path
numerical checks, with all 1,280 float32 moment comparisons numerically equal.
Nominal and explicitly materialized float32-beta native controls are also
numerically equal. The original 4,608 effective-hyperparameter checks remain green.
Candidate 014 has manifest
`8dfe4545c7f706d5f10233266239883a6f5f3ac54acf9b625c89d2b1124c3143`.
The native float64 extension retains caller beta precision; TensorFlow's stored
hyperparameter representation is explicitly matched in its separate controls.
Offline reconstruction from the retained JSON traces confirms matching dtype,
shape and C-order bytes for these equal arrays. The original runtime buffers
were not archived; the original equality flags use `np.array_equal`.

An all-None-gradient counter bug was separately reproduced and fixed. All 92
independent behavior checks and 56 TensorFlow numerical comparisons now pass;
real zero gradients still advance training, and partial missing gradients remain
supported. These local fixes do not replace full-training acceptance.

A further final-update discrepancy was reproduced independently: the native
optimizer combined scaling and subtraction with `add_(direction, alpha=-lr)`,
whereas the TensorFlow reference first forms the scaled update and then
subtracts it. The minimal correction uses `sub_(direction * lr)`. The same
public float32/float64 regression changes from two failures to two passes;
the separately frozen independent regression changes from 37 failures among
40 cases to 40 passes. Candidate 016 also passes the original 2,304 nominal
float32 and 4,608 aligned-hyperparameter numerical checks. This establishes
the local correction only. Its seed-42 full revalidation is reported below;
the candidate 014 failures remain preserved.
Candidate 016's frozen scientific snapshot has manifest
`ef3a4f1a1b997c2a016f43a83f36e01fe66277244f611bfbe7c5f05a784ec8eb`;
its public test records and Markdown successors are recorded separately.

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
observer controls pass. These are CPU-only results.

The corresponding same-GPU diagnostic has now completed all 16 cases. All
32 restore/observer control sets, 1,600 actual forward/backward/update checks
and 912 shared-TensorFlow-gradient native-update checks pass their original
numerical gates. In the shared-gradient replay, all 304 first-moment and 304
second-moment arrays are byte-identical; 215 of 304 parameter arrays are
byte-identical. The other 89 parameter arrays differ by at most 1.49e-8, within
the original numerical limits. No case has a wholly byte-identical poststate.
One representative canary also observes original forward, backpropagation,
GC and update execution on the GPU; all 164 captured arrays per framework
match between profiled and unprofiled observer-enabled controls. This is not
exhaustive kernel attribution for every operation or case. These local results
do not establish identical updates, explain the long-training divergence or
replace the twelve failed full-training field/Jacobian criteria.

A subsequent fixed three-arm diagnostic starts from the original TensorFlow
seed-42 epoch-25 checkpoint. Each arm executes the same next 25 epochs,
4,700 updates and 4,794,375 row visits, ending at global optimizer counter
9,400. The reference arm assigns its own exact first-update poststate through
the same import machinery. It reproduces the original TensorFlow epoch-50 complete
57-array parameter/moment state, ten raw optimizer controls, and all four
full-domain field/Jacobian arrays in typed storage bytes. After the first
actual update, the intervention arm directly imports the measured native
57-array poststate into the same TensorFlow variables, then continues with
the same TensorFlow updates and batches. The native arm independently reaches
that identical poststate with its own first update, then continues natively;
later backend evolution is recorded separately.

| Reference versus intervention | Immediately after import | After 4,699 further TF updates | Ratio |
|---|---:|---:|---:|
| Normalized output RMS difference | 3.91504e-8 | 8.20652e-6 | 209.6 |
| Normalized input-Jacobian RMS difference | 3.32497e-7 | 1.17495e-4 | 353.4 |

An independent CPU readback recomputes all 366 raw-array comparisons. This
demonstrates amplification of this one measured combined parameter/moment
perturbation during this interval. Its endpoint differences remain small;
it does not explain all 600-epoch divergence, identify a particular operation
as its cause, or exclude another port defect. The nonlinear contrasts between
the three arms are not an additive decomposition. The twelve original final
field/Jacobian failures remain unchanged. The retained diagnostic metrics'
legacy `completed_epochs` label contains global counters 4,701 and 9,400;
the actual interval is the first update after epoch 25 through epoch 50.

A separate descriptive reuse compares the six candidate-014 final checkpoints across the
same rows and physical scaling. The following distances divide field RMS
differences by one common target standard deviation per output. This differs
from the original gate's TensorFlow-field-energy denominator.

| Common-scale field distance | Three observed TF seed pairs | Three matched TF/PT pairs |
|---|---:|---:|
| u | 0.12912–0.13335 | 0.09942–0.12055 |
| v | 0.13145–0.18006 | 0.08070–0.09067 |

Each matched field distance is below the observed TF seed-pair minimum.
Jacobian comparisons have exceptions: seed 2718's u_x and u_y distances
exceed some TF seed-pair distances, and u_x also exceeds their median. Every
matched Jacobian axis remains below its observed TF seed-pair maximum.
These shared-model comparisons are descriptive, with only three seeds;
they establish neither statistical equivalence nor a cause of divergence.
All twelve original full-training field/Jacobian failures remain unchanged.

### Candidate 016: complete tutorial 3 revalidation still fails

After the staged AdaBelief update correction, native seeds 42, 1729 and 2718
each complete the original 600 epochs, 112,800 updates and 115,065,000 row
visits, compared with their frozen TensorFlow references. Actual consumed
row-ID sequences match the frozen orders, and all 5,472 prefix checks pass.
All nine original MSE gates pass, including the weighted normalized MSE and
both physical component MSEs for each pair.

| Seed | TensorFlow weighted MSE | PyTorch weighted MSE | Paired u/v field NRMSE | u/v Jacobian relative RMS |
|---|---:|---:|---|---|
| 42 | 0.00149056202 | 0.00144493684 | 0.10424254 / 0.08071300 | 0.24229314 / 0.16823285 |
| 1729 | 0.00064450516 | 0.00063270773 | 0.11041576 / 0.08414305 | 0.26798610 / 0.17995844 |
| 2718 | 0.00148514221 | 0.00152167278 | 0.12537044 / 0.10453710 | 0.29897455 / 0.22360369 |

All twelve original paired-field/derivative gates still fail: each field
NRMSE exceeds 0.05, and each input-Jacobian relative RMS exceeds 0.10.
Independent saved-array calculations reproduce these distances for all three
pairs. The local optimizer regression passes, but complete tutorial 3 training
still fails the original acceptance criteria. These results do not establish
the cause of the remaining divergence. The earlier candidate 014 failures
remain recorded separately above.

Array-only descriptive recomputation for these six current candidate-016
endpoints uses the same common physical target-standard-deviation scales as
the earlier context, without evaluating a model or changing acceptance:

| Common-scale field distance | Three observed TF seed pairs | Three matched TF/PT-016 pairs |
|---|---:|---:|
| u | 0.12912–0.13335 | 0.10268–0.12291 |
| v | 0.13145–0.18006 | 0.07796–0.10071 |

Each matched field distance is below the observed TF seed-pair minimum.
For Jacobians, seed 2718's u_x, u_y and v_y distances exceed some observed
TF seed-pair distances; u_x also exceeds their median. Every matched axis
remains below its observed TF seed-pair maximum. These are three dependent
seed-pair comparisons, not a population distribution, an equivalence test or
a calibrated roundoff experiment. The original twelve failures remain;
changing initialization does not isolate their cause or exclude a port defect.

### Tutorial 1: first paired initialization through both optimizers

Native candidate 016 and TensorFlow seed 42 complete the original 5,000 epochs,
20,000 updates and ten million row visits each. All ten actual initial tensors
match in dtype, shape and C-order storage bytes. Actual consumed row IDs and
effective float32 learning rates match; raw nominal learning-rate numbers are
recorded separately. All 992 original prefix checks and all nine original
quality gates pass across the three grids.

| Grid | TensorFlow physical MSE | PyTorch physical MSE | Paired field RMS / truth std |
|---|---:|---:|---:|
| Training, 2,000 points | 2.75163092e-9 | 3.02419512e-9 | 0.00246331 |
| Original-time refined, 8,000 points | 2.75196248e-9 | 3.02490714e-9 | 0.00246952 |
| Dense space/time, 36,000 points | 7.58283352e-8 | 3.66338164e-8 | 0.03100185 |

Each grid passes the original physical-MSE, normalized-MSE and paired-field
criteria. Jacobians remain unsupervised diagnostics. Independent saved-array
calculations reproduce the metrics and verify all 20,000 actual batch and
effective-learning-rate records. Original Ada-stage postprocessing, checkpoint
reload and continuation checks pass; the actual native figures are inspected.

Both frameworks then execute the original 200 fresh L-BFGS calls from their
own AdaBelief endpoints, with at most 1,000 iterations per call. All nine
original final-quality gates pass; there are no L-BFGS prefix gates.

| Grid after L-BFGS | TensorFlow physical MSE | PyTorch physical MSE | Paired field RMS / truth std |
|---|---:|---:|---:|
| Training, 2,000 points | 2.30915509e-9 | 2.47991383e-9 | 0.00236302 |
| Original-time refined, 8,000 points | 2.30868356e-9 | 2.48073410e-9 | 0.00236681 |
| Dense space/time, 36,000 points | 7.50207275e-8 | 3.57095241e-8 | 0.03095228 |

Independent saved-array calculations reproduce the paired field and diagnostic
Jacobian distances. The complete TensorFlow reference repeat, including both
optimizer stages and their postprocessing, passes all four declared roles:
5,085 exported array pairs in 325 NPZ files match in dtype, shape and C-order
storage bytes. Nine scientific JSON records agree after typed parsing; their
floating-point byte checks reconstruct binary64 values from JSON and do not
establish identity of original runtime buffers. This does not assert identical
state after every update or checkpoint-container byte identity.

Execution and final-quality passes do not establish optimizer convergence.
In TensorFlow's primary and repeat runs, TFP reports `failed=true` in all 200 calls and
`converged=true` in none, with 226 total reported iterations and 6,740 function
evaluations per repeat. Native L-BFGS reports 241 iterations and 1,056 function
evaluations; its public state does not distinguish the stopping condition, so
neither convergence nor failure is inferred. The original budgets are limits,
not a requirement to perform 200,000 iterations. These reference limitations
remain recorded. These seed-42 results form part of the three-initialization
primary matrix below; they are not whole-tutorial acceptance.

The seed-42 native solver diagnostic starting from the shared TensorFlow Ada
endpoint passes all nine final-quality gates but fails one of eleven initial
comparisons: the gradient's maximum absolute difference is 1.01448968e-5.
The original failed criterion remains recorded; its cause is not established.
The opposite diagnostic, using TFP from the shared native Ada endpoint, passes
all eleven initial comparisons and all nine final-quality gates. These are
supplemental crossed-start comparisons, separate from the primary own-stage
results.

### Tutorial 1: three original own-stage pairs

Seeds 42, 142 and 242 each complete the original AdaBelief workload in both
frameworks: 5,000 epochs, 20,000 updates and ten million row visits. All 2,976
prefix comparisons pass. Actual initial tensors, consumed batch IDs and
effective float32 learning rates match. Each framework then executes 200 fresh
L-BFGS calls from its own AdaBelief endpoint, with at most 1,000 iterations per
call. There are no L-BFGS prefix gates.

At each stage, 19 of 27 original quality checks pass. All nine paired-field
checks pass; eight MSE checks fail. Physical and normalized MSE are two units
for the same underlying grid error, so these are not eight independent causes.
The original criteria and failures remain unchanged.

| Seed | Stage | Quality checks passed | Dense space/time TensorFlow MSE | Dense space/time PyTorch MSE | Original physical-MSE limit |
|---|---|---:|---:|---:|---:|
| 142 | AdaBelief | 7/9 | 9.64753381e-9 | 1.67622105e-8 | 1.07701878e-8 |
| 142 | Own-stage L-BFGS | 3/9 | 6.63905973e-10 | 1.79475033e-8 | 8.88197217e-10 |
| 242 | AdaBelief | 3/9 | 1.24593845e-8 | 1.86036171e-8 | 1.38632236e-8 |
| 242 | Own-stage L-BFGS | 7/9 | 1.72363263e-9 | 3.00787527e-9 | 2.05389654e-9 |

The seed-42 results are shown in the preceding tables. Seed 142 fails MSE on
the dense space/time grid after AdaBelief and on all three grids after L-BFGS.
Seed 242 fails MSE on all three grids after AdaBelief and on the dense
space/time grid after L-BFGS. The L-BFGS contract investigation cannot erase
or explain these earlier AdaBelief-stage failures.

In the original TensorFlow own-stage L-BFGS runs, seeds 142 and 242 each report
198 failed calls, two calls reaching the iteration limit, and no converged
calls. Their totals are respectively 2,411 iterations / 12,633 evaluations
and 2,232 / 12,511. Native reports 2,253 / 3,248 and 1,740 / 2,673, respectively;
its remaining stop reasons are not distinguished by the recorded native
state. These counters describe execution, not successful convergence.

The seed-142 native diagnostic from the shared TensorFlow Ada endpoint passes
all eleven initial comparisons but only three of nine final-quality checks;
all six MSE checks fail. The opposite diagnostic passes all nine final-quality
checks but fails two initial gradient comparisons. The seed-242 native
diagnostic from the shared TensorFlow endpoint likewise passes all eleven
initial comparisons and three of nine final-quality checks. The opposite
seed-242 diagnostic, using TFP from the native endpoint, passes all eleven
initial comparisons and seven of nine final-quality checks; dense space/time
physical and normalized MSE fail. It reports 1,460 iterations and 9,643
evaluations across 200 calls: 198 failed, one converged and one reaching the
1,000-iteration limit. All six original crossed runs are now complete.

The initial comparisons here evaluate eager normalized-MSE objectives and
canonical trainable gradients. They are separate diagnostics and have not
been proved identical to the solvers' internal evaluation graphs. Their
failures do not establish a cause of the final-field differences. Crossed
runs are supplemental; their original scope does not include final notebook
postprocessing. Original own-stage postprocessing is recorded separately from
these diagnostics. For each of seeds 142 and 242, all fifteen saved PNG paths
are accounted for and eleven distinct images are actually inspected. The four
primary postprocessing roles complete; the crossed runs supply only their
two actual loss plots. Existing clipping and color-scale limits remain
recorded, and figure inspection does not convert failed numerical checks
into passes. The original comparison matrix is complete; tutorial 1 has
not passed its full quality criteria.

### Tutorial 1: reference L-BFGS contract repair, CPU acceptance

A public analytic regression demonstrates an optimizer-contract difference.
On a small-scale quadratic with gradient above the original threshold,
`torch.optim.LBFGS` stops before moving, while the original TFP solver makes
progress. The same frozen actual-notebook-constructor test changes from one
failure and one pass to two passes after the repair.

The native notebook and generic `lbfgs_minimize` now share a pure-PyTorch
`ReferenceLBFGS` core. It uses Hager-Zhang line search, independent gradient,
actual-displacement, signed relative-objective and absolute-objective stopping
conditions, and the reference history-update rules. The original 200 fresh
calls, 1,000-iteration limit, twenty correction pairs and stated tolerances
remain. Class `max_iter` is an additional per-call budget; the generic helper
retains cumulative `max_iterations`. Stored state supports the notebook's
`weights_only=True` loader and rejects incompatible legacy optimizer state.
The two adapted modules retain their third-party attribution and license in
both source and built distributions.

Agent B's twelve valid, previously frozen CPU cases pass through both the
class and generic APIs: each passes twelve numerical and twelve API-contract
checks. The six old stopping failures close under both entry points, with
actual flags/counters, disk state restoration and cumulative resume checked.
The fixtures, tolerances and comparison predicates remain unchanged. This
numeric-only acceptance does not include NIF model training or establish
general finite-left-endpoint failure coverage. Invalid oracle-isolation
attempts remain recorded separately and are not counted as valid cases.

The complete public CPU suite passes 530 tests with four original skips,
including the real Git/autocrlf checkout test. A read-only source review finds
no material issue in the accepted numeric core. These checks establish the
repair's tested CPU behavior. The complete tutorial-1 GPU revalidation below
retains the original quality gates and candidate-016 failures. No hosted-Colab
completion follows from these CPU results.

### Tutorial 1: repaired optimizer completes the matrix, with failures retained

Candidate 017 passes all 1,312 original short-run numerical comparisons,
2,981 declared-state dtype/shape/storage-byte checks and 32 counter checks
across four fresh processes. Four fresh disk-continuation routes and the
actual notebook's zero-update constructor/closure check also pass. These
restore checks cover the declared weights, moments, hyperparameters, clock
and mirror state; they do not establish preservation of every callback or RNG.

Three native own-state runs then complete the original 200 fresh L-BFGS
calls. Their unchanged AdaBelief checkpoints and the frozen TensorFlow
reference results are reused with exact source/state binding. No original
5,000-epoch AdaBelief training is silently replaced or rerun.

| Seed | Training MSE, TF / native | Dense-45 MSE, TF / native | Original quality checks |
|---|---|---|---:|
| 42 | 2.30915509e-9 / 2.47897747e-9 | 7.50207275e-8 / 3.56939541e-8 | 9/9 |
| 142 | 4.23194535e-10 / 1.07651044e-9 | 6.63905973e-10 / 1.60285844e-8 | 3/9 |
| 242 | 6.82955692e-10 / 4.29789676e-10 | 1.72363263e-9 / 1.31109825e-9 | 9/9 |

The repaired L-BFGS stage passes 21/27 checks, compared with candidate 016's
19/27. Seed 242's two MSE failures close; seed 142 still fails the physical
and normalized MSE criteria on all three grids. These six records represent
the same three grid errors expressed in two units. All nine paired-field
checks pass. Unchanged AdaBelief still passes 19/27 with eight MSE failures
and all 2,976 prefix checks passing. The full tutorial remains unaccepted.

The shared-start comparisons hand off **AdaBelief-end** weights. They do not
start from refined L-BFGS endpoints. Both solvers then use the same handed
weights and retain their original 200 fresh calls and limits:

| Seed | Start from TF AdaBelief end: quality | Start from native AdaBelief end: quality |
|---|---:|---:|
| 42 | 7/9 | 9/9 |
| 142 | 9/9 | 9/9 |
| 242 | 3/9 | 9/9 |

The actual handoff tensors match in their shared canonical layout, including
dtype, shape and storage bytes. The initial comparisons retain one gradient failure for
seed 42's TF start and two for seed 142's native start. These are not failed
weight transfers; the gradient readbacks do not establish a discrepancy in
TFP's internal traced solver gradient or identify the cause of the divergence.
The shared-start outcomes do not replace the own-state primary failures.
The reverse TF cross-runs remain immutable reused references because their
native AdaBelief-end inputs and TF implementation are unchanged; they are
compared with the new native L-BFGS outputs.

Actual `ReferenceLBFGS.result` records report 231/1,259/2,318 iterations and
6,780/9,557/12,509 objective evaluations for seeds 42/142/242. Each executes
200 fresh calls; none reports convergence. Failed calls number 200/199/198;
the remaining 0/1/2 calls reach the 1,000-iteration limit. The original TF
flags and all candidate-016 records remain preserved. Completion of the
training loop is distinct from reported convergence and quality acceptance.

All twelve original comparison records complete: six pass and six fail when
their initial and final checks are combined. Visual inspection individually
covers twelve actual PNG paths, nine distinct images. Own-state reconstruction
plots show normalized u on the 10-by-200 training grid with separate color
scales and coherent residual definitions; this is not dense-grid visual
coverage. Crossed-start inspection covers only the three actual loss PNGs.
Readable, unclipped canvases do not replace the retained numerical failures.

### Tutorial 1: official PyTorch L-BFGS quality comparison

Practical acceptance, 2026-10-25: the user accepts this tutorial-1 L-BFGS
test and comparison. Keep the current `ReferenceLBFGS` default. This records
the user's acceptance decision for this test; all original numerical results,
failed automatic criteria and verification limits below remain recorded.

A separate practical comparison evaluates official `torch.optim.LBFGS`
as an alternative when its reconstruction is similar or better. This does
not change the original TensorFlow parity gates.
The current custom solver's twelve valid independent CPU/TFP cases above
still apply: all five integrated optimizer modules have the same raw bytes
as that tested source. This reuses existing evidence; it is not a fresh
analytic test run or a general correctness proof. Read-only review has not
identified a finite-path port defect explaining tutorial 1's remaining errors.
Nonfinite-objective and interface-lifecycle boundaries remain outside that
conclusion.

The following new comparison runs on the same CPU with one thread and
float32 parameters. For each seed, both solvers receive exactly the same
stored native AdaBelief-end weights: ten tensors and 5,853 scalars, with
dtype, shape and C-order bytes checked. Neither AdaBelief nor the reference
TensorFlow training is rerun. Both use the original full 2,000-row normalized
MSE and normalization, with the original zero regularization term retained.

Both solvers execute 200 fresh calls, each with `max_iter=1000`. Custom
Hager-Zhang retains its original twenty correction pairs and independent
stopping conditions. Official strong Wolfe uses `lr=1`, `history_size=20`,
`tolerance_grad=tolerance_change=1e-15`, and `max_eval=1250` per call.
These are explicit comparison settings; official stopping and line-search
semantics remain different from TFP. See the
[official PyTorch L-BFGS API](https://docs.pytorch.org/docs/2.14/generated/torch.optim.LBFGS.html).
The earlier 50-call, 20-iteration screening runs cannot establish official
precision limits: every seed's first call reached that smaller iteration cap.

All six complete runs finish at returned optimizer boundaries. Final physical
MSE is recomputed on the actual final parameters; the official `step()` return
is the entry loss and is not used as the terminal loss.

| Seed | Solver | Training MSE | Dense-10 MSE | Dense-45 MSE | Optimizer closure calls |
|---|---|---:|---:|---:|---:|
| 42 | Custom Hager-Zhang | 1.13484588e-9 | 1.13974530e-9 | 3.53085175e-8 | 11,914 |
| 42 | Official strong Wolfe | 2.47829490e-9 | 2.47907297e-9 | 3.56737854e-8 | 1,079 |
| 142 | Custom Hager-Zhang | 7.02273073e-10 | 7.02277318e-10 | 1.48484918e-8 | 13,026 |
| 142 | Official strong Wolfe | 3.25480265e-9 | 3.25373811e-9 | 1.62317698e-8 | 1,027 |
| 242 | Custom Hager-Zhang | 6.24668650e-10 | 6.25445897e-10 | 1.95675253e-9 | 9,929 |
| 242 | Official strong Wolfe | 2.55827026e-9 | 2.56008230e-9 | 1.72092477e-8 | 1,046 |

Official strong Wolfe uses fewer closures but has higher MSE on all nine
whole-grid comparisons. Instrumented elapsed times for seeds 42/142/242 are
588.83/565.31/475.25 seconds for custom and 112.32/108.02/107.50 for official.
These include imports, guards, 201 full-grid evaluations, endpoint gradients,
checkpoint writes and I/O; they are not isolated production optimizer timings.
Neither solver is asserted to have converged merely because the calls finish.
Custom reports `failed=True` on 198/200, 198/200 and 199/200 calls for seeds
42/142/242, and reports zero converged calls. Those native flags remain
recorded. Official exposes no corresponding convergence/failure result flag;
no such status is inferred from its completed calls. These observations
distinguish tested reference behavior from successful full-model convergence.

The coordinator independently reads the saved artifacts without importing an
ML framework or executing a model. All three initial canonical states and
truth/coordinate arrays match exactly; all eighteen physical MSE values are
reproduced exactly from saved arrays. Thirty input hashes remain unchanged.
The seed-142 common-scale field/residual figure shows visually similar waves
and residual differences, particularly beyond the last training time, t=90.
That visual observation does not replace the numerical results.

Seven official screening recipes cover fresh/persistent strong Wolfe,
outer change tolerance zero, whole-objective scaling by 1e3/1e6, official
defaults without line search, and a persistent no-line-search variant with
`lr=0.1`, history twenty, change tolerance zero and objective scale 1e3.
Only optimization values and gradients are scaled; all reported MSE and
selection scores retain original units. Torch source is never patched.
The last variant lowers seed-142 Dense-45 MSE to 1.36917072e-8 after 50 calls,
but its training MSE is 2.31236852e-9. Its equal-weight three-grid normalized
score is worse than complete custom, so the original frozen screening
confirmation trigger remains false. Every one of those fifty calls used its
twenty-iteration allowance. A separately frozen extension keeps the same
recipe and script and increases only the maximum completed calls from fifty
to 200 and the cumulative closure allowance from 2,000 to 4,000. Both time
safeguards remain unchanged. It starts from the same AdaBelief-end weights
and actually completes all 200 calls, with 4,000 optimizer closures and no
budget overshoot or timeout; instrumented elapsed time is 252.43 seconds.

| Official persistent no-line-search run, seed 142 | Training MSE | Dense-10 MSE | Dense-45 MSE |
|---|---:|---:|---:|
| 50 calls, 1,000 closures | 2.31236852e-9 | 2.31041506e-9 | 1.36917072e-8 |
| 200 calls, 4,000 closures | 9.03217390e-10 | 9.02762060e-10 | 1.75601423e-8 |

The longer run improves the training fit while worsening Dense-45, including
its t>90 partition. At the declared terminal state, all three whole-grid MSE
values exceed complete custom for seed 142. Its normalized equal-grid score
is 4.08825306e-5 versus custom's 3.43107155e-5, so the separately declared
extension confirmation trigger is also false. No additional target-seed
confirmation or parameter search is run. The two frozen false triggers and
the distinct fifty-call result are retained; a favorable intermediate point
is not relabeled as the 200-call terminal result. The coordinator also
checks exact starting weights, truth/coordinates and all final physical and
normalized MSE from saved arrays for this extension, without a model call.

The existing notebook and optimizer default remain unchanged. These CPU
experiments do not replace the earlier GPU results, erase the six original
L-BFGS MSE failures, prove a roundoff cause, or establish fresh Colab execution
of an official-optimizer notebook. The nineteen CPU experiment jobs are
complete; the tested official settings do not establish a consistent quality
improvement over complete custom. Tutorials 6 and 7 have since completed their
separate execution queues with recorded numerical failures retained.

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

### Tutorial 5: three complete controlled pairs and reference repeat

The first paired initialization, seed 42, passes the original full-workload
criteria. Each framework consumes the same 100 TFRecord shards, containing
one million unique rows, over two passes: 200 shard visits, 15,800 updates
and two million actual row visits. Every shard includes the original
16-row remainder batch. Actual consumed row IDs have the same digest,
`976ee40987acd59e4acf7ecda340faf96f797220bcfdc81e0a1bed933d602860`.
The 15 actual parameter scalars, inputs, targets, order and training settings
are matched. Both runs use the same RTX 3080 Ti Laptop GPU, with deterministic
execution requested and TF32 disabled. The recorded runtimes are TensorFlow
2.21.0 and PyTorch 2.14.1+cu130, both with NumPy 2.1.3.

| Final metric | TensorFlow | Native PyTorch | Original criterion | Result |
|---|---:|---:|---|---|
| Full-domain MSE | 0.0833108572930703 | 0.0833108573612921 | Native <= 1.10 TF + variance floor | Pass |
| Paired field RMS / target RMS | — | 6.88626e-8 | <= 0.05 | Pass |

All three component-MSE gates and all 352 full-run prefix comparisons also
pass. Jacobian relative RMS is descriptive under the original tutorial 5
protocol; no new Jacobian acceptance threshold is introduced. This source
tutorial trains a single fully connected layer with tanh activation on random
inputs and targets.
The close paired results verify the stream-training comparison; they do not
establish predictive accuracy on a physical problem.

The original ordinary end cells, checkpoint reload and fresh optimizer
continuation pass. Each clone advances from update 15,800 to 15,801. Its
parameter and moment arrays agree with its same-framework in-memory
continuation under the original `np.array_equal` checks. These equality flags
alone do not establish dtype or storage-byte identity. The original model's
numeric state remains unchanged under the original comparisons. The actual
native loss figure was opened and
inspected: its 200-visit curve and axis labels are readable. The notebook's
streamed training-aggregate MSE, approximately 0.09213144, is distinct from
the final checkpoint MSE in the table.

Seed 142 also completes the original 15,800 updates and two million row
visits per framework and passes the same full-workload criteria. Its
TensorFlow/native full-domain MSE is 0.083331931307235 /
0.0833319312767014, and paired field RMS / target RMS is 7.19945e-8.
All three component-MSE gates, all 352 prefix comparisons and exact paired
row-consumption checks pass. Independent NumPy recomputation from all one
million raw predictions reproduces these metrics. The ordinary end cells,
checkpoint reload and continuation checks pass within the numerical-equality
scope described above; the actual native 200-visit figure is also inspected.

Seed 242 completes the same original workload and passes all three component-
MSE gates, all 352 prefix comparisons and the exact paired row-consumption
checks. Its TensorFlow/native full-domain MSE is 0.0833249163339565 /
0.0833249163102524, and paired field RMS / target RMS is 6.17476e-8.
Independent NumPy recomputation again uses all one million predictions.
The original end cells, checkpoint reload and continuation pass within the
numerical-equality scope described above. The actual native 200-visit figure
is inspected; its accumulated training MSE of approximately 0.08696509 is
distinct from the final checkpoint MSE.

The native evaluator's filename-resolution metadata cache was changed after
the untouched TensorFlow run. Independent controls establish the old/new
native observer's exact state, loss, input and counter agreement; the two full
runs are not represented as using an identical evaluator hash.

The TensorFlow seed-42 repeat now passes the original repeat checks. All
126,408 declared buffer pairs across 15,801 saved state observations match in
dtype, shape and C-order storage bytes. Another 492 arrays in 36 NPZ files
also match under those checks. Loss, counter and actual-consumption logs
agree in their selected parsed fields; this does not assert log-file or
TensorFlow checkpoint-container byte identity. The original eight end cells,
checkpoint reload, full-million-row reload evaluation and 4,096-row holdout
also pass. The TensorFlow source produces no PNGs in these lanes; the native
figures were inspected as described above. Clone-continuation equality retains
its numerical-only scope.

The three complete pairs, reference repeat and end-cell checks complete the
declared local matched-training protocol for tutorial 5. Preprovisioned WSL
GPU environments were used, so this is not a fresh installer or hosted-Colab
execution claim.

### Tutorial 2: user acceptance

On 2026-10-25, the user accepted the current tutorial-2 results after reviewing
the absolute TF/PyTorch prediction differences and instructed that these
differences need no further investigation. Tutorial 2 is therefore accepted
by the user. The numerical reports below retain their original results:
24/30 main quality checks pass, six SIREN paired-field checks fail, and the
separate same-final-weight diagnostic retains nine failed array/quantity
check groups. This acceptance decision does not change the recorded gates
or measurements.

### Tutorial 2: first complete pair fails the original field criterion

Both reference fits, Swish and NIFMultiScale, have completed the original
5,000 epochs each: 20,000 updates and 10 million row visits per fit. A complete
TensorFlow repeat matches the recorded batch-ID hashes and parsed numeric
learning-rate and loss records exactly. Recorded model/optimizer states and
final field arrays agree in dtype, shape and storage bytes. Six ordinary
notebook PNGs also match exactly. Both corresponding native fits and their
postprocessing have also completed.

The first complete paired initialization passes for Swish and fails for
NIFMultiScale. All 3,328 prefix checks, actual ID/LR records, training-MSE
criteria and both grids' field-MSE allowances pass. NIFMultiScale fails both
original paired-field criteria:

| NIFMultiScale measure | TensorFlow | PyTorch / paired difference | Original criterion | Result |
|---|---:|---:|---:|---|
| Training normalized MSE | 1.26055728e-6 | 1.25057260e-6 | <= 1.10 TF + 1e-6 | Pass |
| Original-time refined-grid field MSE | 0.0031578711 | 0.0030551253 | <= 1.10 TF + variance floor | Pass |
| Dense space/time field MSE | 0.0388786053 | 0.0388091097 | <= 1.10 TF + variance floor | Pass |
| Original-time paired field RMS / truth std | — | 0.17341189 | <= 0.05 | Fail |
| Dense space/time paired field RMS / truth std | — | 0.18983369 | <= 0.05 | Fail |

Spatial-derivative paired distances are 0.31355160 and 0.28607283 on these
grids; the temporal-derivative distances are 0.02640733 and 0.01952633.
The frozen protocol treats these unsupervised derivative comparisons as
diagnostics, so their 0.10 diagnostic reference is not introduced as a new
acceptance gate. Independent raw-array calculations reproduce all 40 paired
metric scalars and verify all 40,000 paired update records against the frozen
fit-specific batch orders.

Seed-42 disk continuation passes for six full-fit cases: TensorFlow
primary and repeat, and PyTorch primary, each for Swish and NIFMultiScale.
Twenty-four fresh workers each perform two updates, for 48 clone updates
and 24,576 row visits; the accepted parent runs receive no further updates.
All 660 original scientific checks and 1,200 observer/restore checks pass,
including dtype, shape and C-order storage-byte equality. This verifies
checkpoint continuation under the original controls. These first-seed checks
do not resolve the failed cross-framework field criteria.

Eight fresh final-weight restoration/transfer cases have completed with zero
optimizer updates. All four own-framework restores reproduce their 24 saved
field arrays exactly in dtype, shape and storage bytes. The recorded model/
optimizer arrays remain unchanged during evaluation. At identical final
weights, Swish passes all 18 original normalized pointwise checks; SIREN
passes nine and fails nine across the training and two refined grids. The
largest normalized field difference is 2.20537e-6 and the largest spatial
derivative difference is 5.03540e-4. All failed indices are retained.
SIREN's maximum whole-array relative RMS difference is 8.51456e-7; this is
descriptive and does not replace the failed pointwise criteria. The original
complete-training paired-field failures remain. These diagnostics do not
establish their cause or exclude every implementation defect.

The first evaluator process failed after training while serializing a
float32 learning-rate record. Its exit status remains failed. A narrowly
scoped JSON writer correction preserves the scientific code and criteria;
durable history and checkpoints were retained. A fresh process completed
postprocessing on the same GPU with zero optimizer updates and unchanged
restored states. The original process's live after-state check remains
unavailable. The complete repeat subsequently finished with a successful
exit and its actual postprocessing checks.

The source tutorial intentionally uses Swish's poor high-frequency fit as a
spectral-bias comparison. The original 2,000 training points, 8,000-point
spatially refined grid and 36,000-point space/time grid are evaluated
separately. Independent raw-array checks reproduce all 14 stored metric
scalars, verify finite field/derivative exports and confirm physical inverse
scaling. These reference and repeat results do not establish
TensorFlow/PyTorch parity.

### Tutorial 2: all three complete paired initializations

Seeds 142 and 242 also complete both original 5,000-epoch fits per framework,
including their actual postprocessing. Each fit performs 20,000 updates and
ten million row visits. All 9,984 prefix checks across the three paired
initializations pass. Actual consumed row-ID streams match and recorded
learning rates have zero difference in every pair.

Swish passes the original full-workload criteria in all three pairs. SIREN
passes the training-MSE and both grid-MSE allowances in all three pairs, but
all six original paired-field criteria fail:

| SIREN seed | TensorFlow training MSE | PyTorch training MSE | Original-time paired field RMS / truth std | Dense paired field RMS / truth std |
|---|---:|---:|---:|---:|
| 42 | 1.26055728e-6 | 1.25057260e-6 | 0.17341189 | 0.18983369 |
| 142 | 1.56999709e-6 | 1.54508141e-6 | 0.36631112 | 0.39252327 |
| 242 | 2.03971577e-6 | 2.05244025e-6 | 0.17733121 | 0.18551858 |

The original paired-field limit remains 0.05. Derivative comparisons remain
unsupervised diagnostics. Actual figures for both added initializations are
inspected, with exact-hash reuse of unchanged initialization images recorded
separately. The six original SIREN paired-field failures and nine SIREN
final-weight pointwise failures remain recorded separately from the completed
disk-continuation controls below.

### Tutorial 2: declared-state disk controls complete

All fourteen original disk cases and all fourteen observer-neutrality
comparisons pass. Their original predicates check dtype, shape and logical
C-order storage bytes, including signed zero: 1,536 scientific/state checks
and 2,792 neutrality checks pass. Fifty-six completed fresh workers report
two updates and 1,024 row visits each, totaling 112 clone updates and 57,344
row visits. The accepted parent runs receive no extra updates. The optimizer
counter advances from 20,000 to 20,002; TensorFlow's separate fit-local train
counter starts at zero and advances to two, rather than being claimed restored.

These are same-framework comparisons between independent reconstruction
from accepted final NPZ state and restoration of the actual epoch-4,999 disk
checkpoint. They establish continuation of the declared state, rather than
continuation of the original live process with every callback and RNG state.
Instrumented ON runs capture the actual input bytes; OFF controls retain no
independent actual-consumption claim. All 140 retained comparison, neutrality,
worker and lane JSON files have their current raw hashes verified.

Fourteen original full fits are recorded across seven parent workers, with
six genuine full-parent proofs. The original TF seed-42 primary parent retains
its failed exit and absent full-parent/live-after proof. Its durable fit and
checkpoint records, separately identified same-GPU zero-update postprocessing
and successful full repeat remain distinct evidence. Four previously qualified
process-lifetime AutoGraph cache files remain unavailable for rehashing. Repeat
array equality and parsed numerical-log equality retain their separate scopes.

The WSL interruption's start-without-end attempt remains unaccepted. Its
unread-back update count is not declared zero. Previously completed cases are
reused; only unfinished cases are recovered, with their successful attempt
records retained separately. These completion counts and passed restore
checks do not relabel any of the original cross-framework quality failures.

### Tutorial 2: descriptive variation across retained seeds

A separate NumPy-only analysis compares the saved endpoints on identical
coordinates and targets, using field RMS difference divided by each grid's
fixed target standard deviation. All 42 input-file hashes remain unchanged;
the saved physical fields and Jacobians reproduce the original inverse
scaling exactly. This analysis performs no model evaluation or training.

For SIREN, the three observed TensorFlow cross-seed pairs have larger field
distances than the matched TensorFlow/PyTorch pairs on both refined grids:

| Grid | Three TensorFlow cross-seed pairs: observed min–max | Matched TensorFlow/PyTorch seeds 42 / 142 / 242 |
|---|---|---|
| Original times, refined space | 0.502926–0.528424 | 0.173412 / 0.366311 / 0.177331 |
| Dense space/time | 0.576652–0.612783 | 0.189834 / 0.392523 / 0.185519 |

All six original paired-field failures remain failures under their frozen
0.05 gate. The three dependent seed pairs provide descriptive background,
not a population variability estimate or an equivalence test. Changing the
seed also changes initialization and controlled order; this comparison does
not isolate roundoff or exclude a port defect. Coordinate mapping is bound
to the retained canonical data, capture hashes and evaluator source, rather
than a new runtime trace. The original TF seed-42 proof limitations remain.

### Tutorial 7: complete baselines fail before primary pruning

Three paired initializations and an independent TensorFlow seed-42 repeat
complete all seven original baseline jobs: 5,000 epochs and 20,000 updates
per job, totaling 140,000 updates and seventy million row visits. Actual
initial tensors and consumed order match. All 5,091 paired prefix checks
pass, but all three complete pairs fail the original combined criteria.

| Seed | TensorFlow original-row MSE | PyTorch original-row MSE | Dense paired field RMS / target RMS |
|---|---:|---:|---:|
| 42 | 1.18771606e-6 | 1.34051812e-6 | 0.23943055 |
| 142 | 9.68691838e-7 | 1.05610853e-6 | 0.44951899 |
| 242 | 1.92301012e-5 | 3.07866421e-5 | 0.87533184 |

The independent TensorFlow repeat passes its original numerical comparison:
all 1,697 prefix comparisons have zero maximum absolute difference, the
recorded MSE values agree, and paired field/Jacobian distances are zero.
These are the original numerical comparisons, not a new dtype/storage-byte
identity claim. Figure inspection covers seven actual loss-plot paths and
six distinct images, with exact-file reuse for the unchanged repeat plot.
The baseline outputs do not supply field, residual, Jacobian or pruning PNGs.

No primary baseline-dependent 50,000-epoch pruning or export is released by
these failed pairs. A separate diagnostic continuation from the original own
baselines is authorized and pending; it retains the original workloads and
checks while explicitly recording the failed baseline dependency. It cannot
produce primary pruning acceptance or whole-tutorial parity. The original
baseline gates and blocked records remain unchanged.

### Tutorial 6: GPU controls retain cross-framework failures

All nine configurations pass the same-framework GPU observer controls in
both frameworks: eighteen OFF/ON pairs. All 36 fresh disk-next controls also
pass. The actual TensorFlow-GPU and PyTorch-GPU 32-update prefixes
pass 20,532 of 20,556 numerical checks: seven configurations pass completely,
while two retain 24 failures. These are separate from the seven earlier CPU
prefix failures; neither set has been removed or accepted by changing a gate.

A bounded diagnostic restores the saved pre-update state and actual used
gradient for each of the 32 steps in those two configurations. It performs
128 independent one-step clones, with no model forward calls or parent
training updates. Native self-replay passes all 1,920 numerical checks and
matches every compared array byte-for-byte. Starting instead from each saved
TensorFlow state and gradient also passes all 1,920 numerical checks; all
640 first-moment and 640 second-moment arrays match byte-for-byte. Parameters
are byte-identical in 292 of 640 comparisons, with maximum absolute
differences of 5.96e-8 and 1.49e-8 in the two configurations, within the
original parameter gates. This diagnostic does not reproduce a direct
moment-update defect. It does not prove the cause of the accumulated
trajectory differences or establish full quality acceptance.

The original-budget full matrix is authorized as a measurement that retains
both CPU and GPU prefix failures. Its first TensorFlow fit completes all
3,000 epochs and 12,000 updates, and its disk-next process completes. The
following zero-update saved-final evaluator stops on a proof-label mismatch:
the source guard checks the exact frozen bytes successfully, but its returned
record contains different descriptive labels before and after evaluation.
The versioned label-only evaluator repair passes a frozen source-only
regression and then completes the same original zero-update GPU request.
The restored counter remains 12,000, all eight profiled training/update call
counts are zero, 66 owned arrays stay unchanged and eight saved/live field
and Jacobian arrays match strictly. The original stop is retained. Completed
fit and disk evidence are reused while the remaining full matrix continues.
Shared-grid comparisons and figures are still required. No whole-tutorial
acceptance follows from this partial execution.

### Tutorial 8: controls and complete matched training

Four fresh TensorFlow/PyTorch GPU lanes, with observation on and off, complete
both variants' eight epochs and 32 updates. Observation-ON lanes record
16,000 actual consumed rows per variant. OFF lanes record controlled source
delivery and actual update counts, linked to ON by strict observer-neutrality
checks; direct consumption is not claimed for OFF. The original GPU numerical
comparison passes 20/20 and 3,264/3,264 checks. The mandatory control certificate
also passes with the original typed comparisons.

An evaluator-local loop variable had overwritten the expected row count with
the dense-grid size before checking the second variant. The actual retained
counts were correct. The original failure is preserved; a three-name local
variable rename passes the same actual-metadata regression, and the
artifact-only comparison succeeds using the four completed lanes. Additional
provenance metadata is kept in a separate sidecar so the canonical certificate
retains its original schema. No training, metric or tolerance was changed.
All seven original full training lanes subsequently finish: three TensorFlow/
PyTorch pairs and one TensorFlow seed-42 repeat. Each lane completes two
50,000-epoch variants, 400,000 primary updates and 200 million actual row
visits. Together they account for 2.8 million primary updates and 1.4 billion
row visits. Independent checkpoint clones add 16 updates per lane, without
updating the primary runs.

| Paired seed | Full quality checks passed | Prefix checks passed |
|---|---|---|
| 42 | 10/20 | 3,264/3,264 |
| 142 | 14/20 | 3,264/3,264 |
| 242 | 12/20 | 3,264/3,264 |
| Total | 36/60 | 9,792/9,792 |

All three original reports verify the full workload. The original parent
stops after the seed-42 quality failure; the remaining two original
comparators run in a recorded CPU-only continuation against the completed
lane artifacts. The original stop, sources and thresholds remain unchanged.
The 24 failed checks include paired field/spatial-derivative differences
and some terminal MSE/objective criteria. For seed 42, normalized paired
field RMS is 0.05748 on the training grid, 0.05750 on spatial superresolution
and 0.32896 on the dense grid. Dense paired spatial-derivative RMS is 0.26462.
The original field/derivative bounds remain 0.05/0.10. Time derivatives are
monitor-only. The original aggregate executes and returns `passed=false`
and `full_learning_acceptance=false`, with the original numeric-gate/workload
failure reason, `integrity_error=null` and `numeric_gates_changed=false`.
The CPU continuation ends with exit 1 after completing its independent checks.

The original strict TensorFlow seed-42 repeat passes: 9,162 retained state
and output arrays match by dtype, shape and logical C-order bytes, and
400,000 entries in each recorded host-metric and actual-consumption stream
match. Full checkpoint/postprocessing proofs are rechecked. Intermediate
per-update model/optimizer states were not retained, so that broader claim
is not made. This repeat result does not establish cross-framework parity.
Separate original-PNG inspection is complete for all seven lanes, using
strictly byte-identical prior-view representatives for duplicates. The
original postprocessing records are preserved; the additional image QA
does not replace their stored fields or establish dense-grid accuracy.

The spatial-superresolution grid has 800 spatial points at the same ten
training times, 0, 10, ..., 90. The dense grid has 800 points at 45 times
`100*k/45`: forty are new, and four exceed the largest training time.
These test different temporal coverage. The original spatial-superresolution
figures do not establish performance on that dense grid, and its failures
are retained.

A separately frozen, zero-update CPU diagnostic assigns identical final
weights to both backends: TensorFlow and PyTorch donors for each seed-42
variant, four sets and eight fresh backend executions. All ten canonical
tensors per set are assigned strictly; tracked owned scientific state and
input/source hashes stay unchanged. Every normalized three-column grid
matrix retains pointwise failures under the original prediction tolerance.
Across 552,000 point-components, 12,908 fail: 32 field values, 747 time
derivatives and 12,129 spatial derivatives. Maximum component relative RMS
differences are 1.35e-6, 1.77e-6 and 1.25e-6 respectively. These descriptive
RMS values do not replace the failed pointwise criteria. All four original
seven-row objectives and scalar-loss input gradients pass; 38 of 40
parameter-gradient arrays pass. The two failed arrays belong to the
TensorFlow-donor Sobolev set. The comparison returns exit 1.

This diagnostic uses ordered outer-direct CPU calls with the original
internal TensorFlow `tf.function`, original backend-specific batch shapes,
and a warm seven-row gradient probe after the grids. It does not reproduce
the original eight-call predict-first path, compare identical batch shapes,
or establish same-GPU training-gradient equivalence. It neither identifies
a semantic port defect nor establishes the cause of the complete training
differences. All original 24 quality failures remain unresolved.

A descriptive analysis reads fourteen complete retained epoch histories,
without a new model or framework evaluation. Input raw hashes stay unchanged.
The ratio of PyTorch to TensorFlow mean training objective over the fixed
last 2,000 epochs is:

| Seed | Sobolev objective ratio | Baseline objective ratio |
|---|---|---|
| 42 | 1.06646 | 0.93855 |
| 142 | 1.00613 | 1.09958 |
| 242 | 1.05205 | 1.00846 |

These are epoch-averaged normalized training quantities, not final-checkpoint
physical-grid MSEs or replacement acceptance gates. The TF seed-42 repeat's
parsed history scalars also match exactly, recorded separately from the
strict retained-state result above. Similar loss histories do not prove
field parity, a roundoff cause or absence of a port defect.

### Remaining learning tutorials: readiness, not full acceptance

The following distinguishes completed comparisons from remaining controls
and workloads. A completed comparison can retain failed acceptance criteria.

| Tutorial | Current finite evidence | Still required |
|---|---|---|
| 1 | Candidate-017 complete own-state and shared-start matrices, fresh GPU/restore controls, public suite and independent CPU cases complete; repaired L-BFGS passes 21/27, with original records retained | Six L-BFGS MSE failures, eight unchanged AdaBelief MSE failures and initial-gradient/convergence limitations remain unresolved |
| 2 | All full pairs, TF repeat, fourteen disk cases and fourteen observer controls complete with their stated evidence scopes | User accepted the current results on 2026-10-25; no further investigation requested. Six automatic paired-field failures and nine separate final-weight check-group failures remain recorded. |
| 6 | All 63 full fits, 63 disk controls, 63 saved-final evaluations, original CPU postprocessing and 63 common-field PNG inspections complete; TF repeat checks 42,322 retained arrays strictly equal | Quality retains 24/81 failures, all three paired reports fail, and earlier seven CPU plus 24 GPU prefix failures remain unresolved |
| 7 | All seven baselines, eight full diagnostic pruning runs, five comparisons and 16 PNG-path inspections complete; TF repeat checks 4,770 retained arrays strictly equal; real export/quantized execution is recorded | Primary acceptance remains blocked by baseline quality; three own-baseline pruning pairs and the common-start supplement fail their numerical criteria. T7's source-level metric, figure and artifact-size differences are separately documented in the semantic review |
| 8 | All seven full lanes, three comparisons, strict TF repeat, original aggregate attempt, original figure QA and scoped same-final-weight CPU diagnostic complete. Quality passes 36/60; all 9,792 prefix checks pass; aggregate and CPU pointwise contracts retain failures | The 24 quality failures and the cause of the complete training differences remain unresolved |

The earlier tutorial 7 CPU observer failure remains recorded and is not an
established product defect; its actual TensorFlow GPU controls now pass.
Tutorial 8's replay difference is localized by a
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

### Hosted Colab tutorial 1: full execution, separate from parity

A Drive copy of the public tutorial runs on Colab Python 3.13.16 / Tesla T4
with PyTorch 2.14.1+cu130. The installed Git revision is
`6a7f151b42bf5661e234ba57796a6716aadcb6cf`. Its 18 original code cells match
the repository sources exactly and finish without a saved Python error.
The runtime receipt and downloaded training history confirm 5,000 AdaBelief
epochs, 20,000 updates, 10,000,000 consumed rows and 200 L-BFGS outer rounds
with the original 1,000-iteration per-round cap. All eleven output-file hashes
match the downloaded archive. One additional metadata-only cell records
versions, counts and hashes; the original code cells remain unchanged.

This execution initially encounters Colab output-JavaScript errors. Rerunning
only the final prediction/plot cell displays its figure successfully, but
persisted output rendering after a reload remains unreliable. The failure
screenshots and both notebook exports are retained. Colab exports null
execution counts and no executionInfo here, so those metadata fields are not
represented as execution proof; actual UI run markers, outputs, history and
the separate runtime receipt supply the evidence.

The final normalized training MSE is 9.54576e-6. The last L-BFGS round records
`converged=false, failed=true`; no complete-round convergence summary was
instrumented in this hosted run. It uses its own native random initialization
and batch sequence, so it is an execution check rather than a matched-weight
TensorFlow comparison. It does not close the local parity failures above or
establish hosted execution of the other tutorials.

### Hosted Colab tutorial 2: both complete fits

A fresh Colab CPU session with Python 3.13.16 / PyTorch 2.14.1+cu130 installs
public revision `c8fe5e5bf8bb1de2b11d3cc825250a7934f1a5bd`. All seven original
code sources match the repository exactly. The downloaded notebook retains
execution counts 1–7 and no Python error outputs. Swish and SIREN each finish
5,000 epochs, 20,000 saved optimizer updates and ten million row visits.
Both complete histories retain epoch indices 0–4,999, and their latest disk
checkpoints record epoch 4,999. All eleven original output-file hashes match
the downloaded archive.

The original final normalized MSE on the 2,000-row training grid is
0.0251962 for Swish and 1.12762e-6 for SIREN. This demonstrates the intended
contrast between weak high-frequency fitting and the multiscale model in
this run. It uses its own native initialization and batch sequence, so it
does not replace the matched TensorFlow comparison, evaluate a new dense
grid, or close any of the six paired-field failures above. Original loop
counts are recorded without adding independent consumed-row-ID tracing.

The browser again reports output-JavaScript errors. One appended evidence
cell reads the saved state and redisplays the original Swish/SIREN PNGs;
both figures are inspected in Chrome with no additional model or optimizer
call. Both notebook exports and the original error screenshots are retained.

### Hosted Colab tutorial 3: complete cylinder-flow workload

A fresh Colab Tesla T4 runtime uses Python 3.13.16 / PyTorch 2.14.1+cu130
and the same public revision as the hosted tutorial-2 record. All twelve
original code cells execute with saved counts 1–12 and no Python errors.
The complete history records 600 epochs, all 191,775 AMR rows each epoch,
and 115,065,000 row visits. Saved optimizer group and parameter counters
both record 112,800 updates; the latest checkpoint records epoch 599.
All eight original output-file hashes match the downloaded archive.

The original final weighted normalized training MSE is 0.00184355; physical
u- and v-component training MSEs are 0.0346912 and 0.102301. These evaluate
the original training rows. The saved reconstruction shows the first stored
time slice; it does not supply a new dense-grid or Jacobian comparison.
This hosted run uses its own native initialization and batch sequence, and
does not close the twelve controlled paired-field/Jacobian failures above.

Eleven original code sources are byte-exact. The final plotting cell loses
eight trailing-whitespace bytes in the Colab export; its Python AST and
comment text remain equal. The failed byte-exact source audit and raw diff
are retained, with the execution and whitespace classification recorded
separately. Both exported notebooks preserve the same original code prefix.
No repository notebook is changed to remove the export difference.

The browser again reports the output-JavaScript error. A single evidence
cell reads saved checkpoints, counters and hashes, then redisplays the four
original PNGs without training, another model/optimizer call or generating
new plots. The original figures are inspected, including the true/predicted
cylinder fields; existing title and axis-label clipping is retained.

### Hosted Colab tutorial 4: derivative notebook execution

A separate fresh Colab CPU runtime installs public revision
`35bcb4c2aee42cf95bd65a778b79ceedfb5bac7f` with Python 3.13.16 and PyTorch
2.14.1+cu130. All ten original code sources match the repository exactly.
The downloaded notebook retains execution counts 1–10 and no Python error
outputs. Its analytic, selected-derivative, zero-higher-derivative,
parameter-differentiation and checkpoint-restoration assertions complete.
The maximum Jacobian and Hessian analytic errors are both 5.96046e-8.
The native example's single supplemental Adam update is recorded separately
from the TensorFlow reference's untrained derivative example.

The same Colab output-JavaScript error occurs here. An additional evidence
cell records versions and hashes and redisplays the original saved Jacobian
PNG successfully, with no extra model or optimizer call. The original
notebook, checkpoint, metrics, three output hashes and error/display
screenshots are retained. This establishes the stated CPU execution and
analytic checks; reliable rendering of every persisted browser output remains
unverified.

### Hosted Colab tutorial 5: complete streamed workload

A separate fresh Colab CPU session uses the same Python/PyTorch versions and
public revision as the hosted tutorial-2 record. All nine original code
sources match exactly; its first export preserves execution counts 1–9 and
has no Python error outputs. The original loop generates one million rows,
writes 100 TFRecord files, records two million row visits across 200 shard
visits, and reaches optimizer counter 15,800. The original restoration and
row/shard-count assertions complete. The saved input NPZ is float64 with
shape 1,000,000 by 7. All 104 original output-file hashes match the archive,
which also retains all TFRecord shards and the runtime receipt.

The original printed `mse=0.0897564` averages loss over the streamed training
trajectory. It is not final-checkpoint full-data MSE and must not be compared
as that quantity against the controlled final metrics above. The original
loop counts supply execution evidence; no independent consumed-row-ID/byte
trace is added in this hosted run. The data are the original random-data
demonstration, so successful execution is not evidence of physical accuracy.

An additional evidence cell records versions, counters and file hashes and
redisplays the saved loss PNG in Chrome, without generating records again
or calling the model or optimizer. The same original output-JavaScript
error is retained separately. This hosted execution does not establish a
new matched-weight TensorFlow comparison.

### Hosted Colab tutorial 6: all nine regularization fits

The first downloaded notebook preserves all fifteen original code cells,
execution counts 1–15 and no Python error outputs. Its bootstrap prints
PyTorch 2.14.1+cu130, CPU and FULL mode. All code text and comments match
the repository exactly. The original final metrics report nine 3,000-epoch
fits and resolution 371. Each fit's sparse printed log reaches epoch 2,999;
the 31 printed records per fit are not represented as complete histories.
The original checkpoint-restoration assertions execute before the later
completed cells. All twenty unique embedded PNGs are extracted, hashed and
visually inspected without changing their bytes.

The baseline's original final normalized training MSE is 0.000164249,
and its physical superresolution MSE is 0.0328703. Strong L1 penalties
produce nearly constant predictions: their training MSE is about 0.025196
and superresolution MSE about 0.0198625. That lower superresolution MSE
accompanies loss of the moving-wave structure. The full nine-fit metrics
and figures are retained together, including weak fits and spurious waves.

When a later evidence cell is attempted in the reconnected kernel, its
first execution fails because the original training variables are absent.
A separate standard-library probe also finds the default output directory
absent. The first executed notebook and embedded outputs remain preserved,
but complete history files, raw checkpoints, optimizer counters and a
runtime receipt cannot be independently recovered from that directory.
The later diagnostic failure is recorded separately from the original
fifteen-cell execution. Only the added diagnostic cells are removed from
the working copy after preservation; its next export is byte-identical
to the first export. These execution records do not close the controlled
GPU-prefix or final-quality failures above.

### Hosted Colab tutorial 7: pruning and native int8 export

The first export preserves all twenty-one original code cells, counts 1–21
and no Python error outputs. The bootstrap prints the same CPU/FULL Torch
build as tutorial 6. Code and comments match exactly. Sparse training logs
reach baseline epoch 4,999 and pruning epoch 49,999, with 51 and 501 printed
records respectively. They are not complete per-epoch histories. The
original pruning-step/refresh-count, sparsity, checkpoint-restoration,
floating TorchScript and restored native-int8 assertions execute before
the completed final cell. All four unique embedded PNGs are extracted,
hashed and visually inspected.

The original result records 1,756 zero weights out of 1,951 in the selected
kernel, or sparsity 0.900051, and three packed qint8 Linear kernels.
Relative prediction MSE after pruning is 0.00100999; after pruning and
int8 quantization it is 0.0105517. Both use the unpruned prediction energy
as denominator, rather than independent ground truth. The original gzip
sizes of floating and int8 TorchScript files are 32,232 and 36,506 bytes;
the int8 export is larger in this small example. Figure residual panels
have separate color scales, so colors alone do not compare error magnitude.

A later standard-library recovery probe finds the original variables and
default output directory absent. The original notebook evidence remains;
raw history/checkpoint/archive hashes and independently recovered counters
are unavailable. The prepared state-reading evidence cell is not executed
in that empty kernel. After preserving the probe, only its added cell is
removed; the working-copy export is byte-identical to the first export.
This own-initialization hosted execution does not establish a new matched
TensorFlow comparison or acceptance of the private baseline-dependent
pruning protocol.

Across hosted copies 1, 2, 4, 5, 6 and 7, original cell types and order are preserved,
and all code text, including comments, matches exactly. Observed Markdown
export differences are confined to line-trailing whitespace. The raw
mismatches and their whitespace-only classification
are retained separately; repository notebook sources are not changed to
remove these differences. Tutorial 3's separate whitespace-only code-source
difference is recorded above. Tutorial 8's separate whitespace-only
code-source difference and complete execution are recorded below.

### Hosted Colab tutorial 8: both complete Sobolev and baseline fits

The reconnected Colab T4 session completes all seventeen original code
cells, with execution counts 1–17 and no Python error outputs. An appended
evidence cell completes as count 18. The actual eighteen-cell code export
is preserved; no earlier full export without that evidence cell is claimed.
All original Python ASTs and comment text match the repository. One analytic
derivative cell loses two line-trailing spaces on export, so raw code-source
equality remains false, with the exact difference preserved separately.

The runtime receipt records Python 3.13.16, PyTorch 2.14.1+cu130, NumPy
2.1.3 and Tesla T4, with installed public revision
`fa167ae59bdbb8bf5779eac24b5af32c8dfb998a`. Both original fits finish
50,000 epochs. Their complete histories contain exactly epochs 0–49,999,
each with 2,000 rows: 100 million recorded row visits per fit. The evidence
cell reads the actual saved checkpoints and asserts 200,000 optimizer
updates per fit and latest epoch 49,999. Together they record 400,000
updates and 200 million row visits. The local archive audit independently
checks history contents and raw file hashes; checkpoint counters come from
the hosted checkpoint-reading receipt. No independent consumed-row-ID
trace is added.

The original final physical metrics are:

| Quantity | Baseline | Spatial Sobolev |
|---|---|---|
| Training-grid field MSE | 7.84638e-6 | 1.16323e-4 |
| Training-grid time-derivative MSE | 0.117880 | 0.118920 |
| Training-grid spatial-derivative MSE | 2948.94 | 1.46833 |
| Spatial-superresolution field MSE | 0.00390429 | 9.79360e-5 |

The superresolution grid uses 800 spatial points at the same ten training
times. Spatial supervision has coefficient 1e-3 in the first fit only;
time derivatives remain monitor-only. These final physical-grid metrics
are separate from the printed last-epoch normalized training means.
All fifteen original embedded figures are inspected without replotting.
The baseline retains pronounced off-wave superresolution oscillations;
spatial supervision suppresses them while retaining small residual spikes.
Both time-derivative reconstructions remain poor, and the loss fluctuations
are preserved.

The original output-JavaScript error recurs. The evidence cell reads saved
state, hashes twenty original output files and redisplays thirteen saved
PNGs without another model, training or optimizer call. Its first archive
is downloaded through the Colab Files panel because the automatic browser
download does not arrive. The preserved ZIP contains twenty original
outputs and the receipt; its CRC is clean and all twenty output hashes
match. All thirteen saved PNGs are also inspected and match the evidence
cell's embedded image bytes exactly. This own-initialization hosted execution does not
close the matched-weight quality or CPU pointwise failures above, and
reliable rendering of every persisted browser output remains unverified.

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

The subsequent checkout-line-ending correction also passes actual fresh Git
checkouts of the published candidate on both platforms: Windows has 424 passes
and four skips, and Linux has 424 passes and four skips. The Linux run explicitly
disables GPU access; both suites skip two CUDA checks and two Linux-build-specific
RNG fingerprints when their exact recorded runtime is unavailable. The Windows
checkout uses the default `core.autocrlf=true`. The correction fixes explicit
LF treatment for the tutorial README and byte preservation for three archived
patches; model and notebook scientific source bytes are unchanged. Earlier
failed checkouts, the artifact-serialization failure, and their passing
successors are preserved separately.
