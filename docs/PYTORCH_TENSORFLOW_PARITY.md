# Controlled TensorFlow/PyTorch tutorial comparison

Status: in progress, 2026-10-23. Start with tutorial 3, then cover tutorials
1, 2, 4, 5, 6, 7 and 8. The existing full-execution acceptance remains useful
but is not evidence of matched cross-framework training.

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

Eight Markdown cells in native tutorials 1, 5 and 8 now correct inherited
API, epoch-counting, TFRecord and coordinate explanations. All 44 code cells,
their outputs, notebook metadata and cell order are unchanged. These are
documentation changes to the public copies; the frozen scientific snapshots
retain their original whole-file hashes.

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

A separate descriptive reuse compares the six final checkpoints across the
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
remain recorded. The remaining initializations and shared-start diagnostics
are still required; this is not whole-tutorial acceptance.

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

Actual disk continuation now passes for all six full-fit cases: TensorFlow
primary and repeat, and PyTorch primary, each for Swish and NIFMultiScale.
Twenty-four fresh workers each perform two updates, for 48 clone updates
and 24,576 row visits; the accepted parent runs receive no further updates.
All 660 original scientific checks and 1,200 observer/restore checks pass,
including dtype, shape and C-order storage-byte equality. This verifies
checkpoint continuation under the original controls. The remaining paired
initializations and the failed cross-framework field criteria remain open.

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

### Remaining learning tutorials: readiness, not full acceptance

The following distinguishes finite controls from completed reference work.
Neither completes the remaining paired initializations or full acceptance.

| Tutorial | Current finite evidence | Still required |
|---|---|---|
| 1 | First full candidate-016 AdaBelief pair passes all 992 prefix and nine quality checks; actual postprocessing and checkpoint checks pass; TensorFlow Ada-stage repeat agrees at 2,856 saved arrays | Each framework's own L-BFGS refinement, combined full-tutorial repeat and remaining initializations |
| 2 | First full pair completed: Swish passes, NIFMultiScale fails two original paired-field criteria; complete TF repeat agrees exactly at its recorded observations; all six disk-continuation cases pass; final-weight diagnostics retain nine SIREN pointwise failures | Remaining initializations and final acceptance |
| 6 | All nine regularization cases' actual raw/used gradients pass; fresh disk restoration and two updates pass 1,782 strict storage-byte checks | Four small post-momentum discrepancies remain recorded, plus three inherited pre-momentum differences; full nine-case fits and final aggregate/gallery cells remain pending |
| 7 | Actual TF/native baseline short comparisons and native OFF/ON 8,004-step boundary controls pass, including four actual two-update fresh-disk routes; prior TF-only 8,000-step controls also pass | Own-baseline/full-pruning fits, remaining initializations and real export/quantization quality remain pending |
| 8 | Both 32-update fits pass 3,584 numerical checks; disk continuation and observer controls pass; actual Colab postprocessing order reproduces all six prior common-metric arrays byte-for-byte | Same-GPU checks, both full fits, remaining initializations and final figures |

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
