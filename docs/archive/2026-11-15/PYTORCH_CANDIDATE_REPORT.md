# Agent A native PyTorch candidate

Implementation baseline: `f8f9d31f4aa331b6e7dbc48758302c801441bcce`.
Working branch remains `pytorch`; no commits, branch changes, or git-internal
writes were performed. The public contract was read completely before editing.
This report describes implementation-owned verification, not blind acceptance.

## Delivered implementation

- Native `nn.Module` implementations of all three NIF architectures, retaining
  configuration dictionaries, shape-kernel flattening order, activations,
  residual definitions, SIREN distributions, and differentiable hypernetwork
  outputs. `build()`/`model()` return the same module. Extracted subnets share
  tensors without recursive parent registration. Mixed-policy subnet arithmetic
  and dtype-preserving config/checkpoint restoration are tested.
- All exported layers and scientific helpers, including full-output/selected
  Jacobian and Hessian wrappers, zero higher derivatives, and parameter
  differentiation. Explicit current-batch regularization preserves kernel/bias
  sums, batch-scaled activity sums, and latent Jacobian mean squares.
- Actual native AdaBelief, Lion, and completed published L4-Adam, plus native
  gradient centralization and L-BFGS/packing APIs. L4 uses the contract's global
  inner product and serialized moving loss minimum; exact conventions are in
  `nif/optimizers/external_optimizers.py` and `PYTORCH_API.md`.
- Normalization validation and physical inverse scaling, preserved demo data,
  and genuine legacy TFRecord Example shards read/written without TensorFlow.
  Native streaming retains targets, area weights, remainders, and repetitions.
- Eight new standalone `tutorial/*_pytorch.ipynb` notebooks with pinned setup,
  explicit training loops, metrics, figures, checkpoint restoration, and native
  compression/export. Original notebooks and compatibility tooling are intact
  and clearly identified as legacy by the main and tutorial READMEs.
- Native requirements/setup metadata, installable wheel with all three bundled
  datasets, native API documentation, archived pre-port README, and separate
  development tests/runners under `dev_tests/`.

## Intentional corrections

The port corrects tutorial 6 coordinate order, tutorial 8 physical inverse
scaling, invalid last-layer subnet/bias references, shape-network regularizer
coefficients, the inverted SIREN final-layer configuration check, and legacy
TFRecord multi-column/weighted batching. It completes unfinished L4Adam rather
than reproducing `new_var = None`. Ordinary optimizer clipping preserves its
per-tensor `clipnorm`; the separate legacy centralization helper retains its
explicit global-norm reduction. These are distinct documented conventions.

## Local verification

Environment: Python 3.13, PyTorch 2.14.1+cpu, NumPy 2.1.3. CUDA is unavailable.
Initial and targeted RED checks were run before corresponding implementation
fixes. Tests exercise independent numerical formulas, finite differences,
initialization statistics, gradients and parameter updates, shared parameters,
serialization and dtype, finite training, physical scaling, L-BFGS convergence,
and TFRecord external-reader/writer interoperability and corruption checks.
Final unit tests: **201 passed, 1 skipped (CUDA unavailable)**. Prior notebook
execution hashes remain in `dev_tests/validation_summary.json`; current
display-only source hashes are recorded in the follow-up below.

All eight native notebooks passed sequential execution in separate clean
ipykernel sessions with `NIF_SMOKE=1`. Tutorials 4 and 5 also passed with smoke
disabled. Tutorial 5 generated 1,000,000 rows in 100 genuine TFRecord shards and
consumed 2,000,000 rows over 200 shard visits. Tutorial 7's smoke run verified
1,756 actual zero weights out of 1,951 (90.005% rounded sparsity), three native
packed qint8 Linear operators, and restored TorchScript prediction equality.
Its measured int8-vs-pruned MSE was approximately 2.86e-7; this is an artifact
check, not a claim of trained scientific accuracy.

Editable installation and wheel construction succeeded; wheel metadata lists
no TensorFlow-family dependencies and bundles all three demo NPZ datasets.
Native package imports do not load TensorFlow, Keras, TFP, or TFMOT. Original
legacy tests were preserved and were not rewritten to hide incompatibilities.

Execution notebooks, checkpoints, figures, TFRecord data, and metrics are under
`dev_tests/artifacts/{smoke,full}/`; generated artifacts are git-ignored. The
portable validation summary retains execution results and notebook hashes.

## Smoke/full distinction and remaining independent work

Full scientific defaults are unchanged: tutorial 1 has 5,000 AdaBelief epochs
plus 200 L-BFGS rounds with max_iter=1,000; tutorial 2 has two 5,000-epoch fits;
tutorial 3 has 600 epochs over all 191,775 weighted samples with actual gradient
centralization; tutorial 6 has nine 3,000-epoch fits; tutorial 7 has 5,000 baseline
and 50,000 pruning epochs; tutorial 8 has two 50,000-epoch fits and supervises
only spatial derivatives while measuring temporal derivatives.

The explicit smoke mode reduces training to two epochs per fit; tutorial 1 uses
one L-BFGS round/max_iter=2, tutorial 3 trains 4,096 rows, tutorial 5 generates
10,000 rows/10 shards, and tutorial 6 uses a 41-point superresolution axis. Full
mode remains the notebook default. See the actual notebook constants and
validation summary for authoritative settings.

Full training workloads 1, 2, 3, 6, 7, and 8, CUDA execution, fresh hosted Colab
installation, and Agent B's frozen independent acceptance suite were **not** run
locally. CPU smoke/full-data checks do not substitute for those checks. No
private tests or fixtures were accessed. The candidate is ready for the
coordinator to snapshot and submit to Agent B; full acceptance is not claimed.

## Public-API closure and initializer follow-up

Two implementation-owned regressions reproduced the sanitized public findings
before source edits: **24 failed, 46 passed**. The targeted corrected checks now
pass **73/73**; the complete native suite passes **146 tests with 1 CUDA skip**.
Exact commands, RED logs, final JUnit output, and hashes are in
`dev_tests/validation_summary.json`. No independent sources or fixtures were used.

`centralized_gradients_for_optimizer` now completes L4Adam's automatic backward
inside its closure wrapper when the closure returns only a loss. Centralization
therefore happens after gradients exist and before the optimizer updates moments.
A closure that calls backward still performs exactly one backward. The regression
checks both positional/keyword closure styles, analytical global L4 updates,
native/transposed kernel centralization axes, unaffected vector biases, and exact
serialized moments/minimum across three steps.

Native initialization now accepts all 15 standard names. The added defaults are
Constant(0), VarianceScaling(scale=1, fan_in, truncated_normal), Orthogonal(gain=1),
and Identity(gain=1). Rank-one Glorot biases use fan_in=fan_out=vector length.
Orthogonal/Identity retain their baseline rank restrictions; orthogonal QR runs
in float32 for half/bfloat16 storage and casts back. Distributions and fan rules
follow the [public Keras 2.11 implementation](https://github.com/keras-team/keras/blob/v2.11.0/keras/initializers/initializers_v2.py),
without importing or installing TensorFlow/Keras. A separate two-test RED
identified the native half-precision QR limitation before that path was fixed.

Twenty-six pre-fix 2D initializer value/RNG fingerprints and five representative
model/layer cases remain bit-for-bit identical, including three-step parameter
trajectories for all three NIF architectures. No tutorial source, training
setting, loss, epoch count, or data was changed. Existing notebook source hashes
were verified; their execution records are retained rather than reported as new
runs. Coordinator README/Colab links, discovery configuration, and unrelated
documentation were not edited. Long independent runs and new blind acceptance
remain owned by the coordinator and Agent B.

Final implementation SHA256:

- `nif/optimizers/gtcf.py`: `8698637fb975ff7df70a38a0503fa231f037855beeaf68efa0cb5b7ca37ac1b4`
- `nif/layers/_common.py`: `a52dc9c15e95dbbb311b83f9443dce6c32593dc6f8ad4885b9023380ee36e812`


## Public self-test portability follow-up

Reproduced both portability defects before fixes: **2 failed** in
`dev_tests/portability_red.log` (a simulated GBK source read raised
`UnicodeDecodeError`; a simulated Windows build sharing the Torch version
incorrectly reached the Linux-only fingerprint comparison).

Development source/config reads and runner error/summary writes now explicitly
use UTF-8. Exact fingerprint checks use a companion record of the original
Linux x86_64 capture environment: platform/architecture, byte order, pointer
width, Python/NumPy metadata, Torch version/git revision, and the SHA256 of
Torch's build configuration. A nonmatching or incomplete build identity gives
an explicit skip. The original fingerprint file remains byte-for-byte intact;
no expected tensor, RNG, or trajectory hash was regenerated. Portable
initializer, distribution, optimizer, and math checks run on every platform.
Fifteen new portability cases exercise GBK reads, matching metadata, Windows
with the same Torch version, each individual metadata mismatch, and incomplete
metadata.

Targeted checks: **92 passed**. The default discovery command
`python -m pytest -q --junitxml=dev_tests/artifacts/portability_final.junit.xml`
completed with **161 passed, 1 CUDA skip**, recorded in
`dev_tests/portability_final.log`. Existing multiprocessing/sparse warnings
remain. Native Windows execution was not repeated in this Linux container;
Windows/GBK conditions were simulated for these regressions.

All 53 protected file hashes matched before and after this follow-up, including
all native implementation modules, tutorial files, the generator, the frozen
fingerprint file, and coordinator README/config/API-doc files. Every previously
recorded implementation and notebook hash still matches. No model/optimizer
math, notebook source, or training setting changed. Prior notebook execution
records are retained; no notebook rerun is claimed for this test-only change.
Per-file source/evidence SHA256 values and exact counts are recorded under
`portability_followup` in `dev_tests/validation_summary.json`.

Original fingerprint file SHA256:
`91f2bff6b6d7b63c25bf7e20a3dcbcc30d4559ad38161294b1ce56979db5ea85`.


## Optimizer epsilon constructor follow-up

The public baseline uses `epsilon or backend_config.epsilon()`: an explicit
`None` or zero therefore resolves to `1e-7`. Before the fix, implementation-owned
regressions recorded **18 failed, 10 passed** in `dev_tests/epsilon_red.log`,
including constructor exceptions, nonfinite zero-gradient updates, and
fallback-equivalence mismatches.

Only `nif/optimizers/external_optimizers.py` changed in the native package.
AdaBeliefOptimizer and L4Adam now resolve that fallback before validation.
An omitted-alias sentinel distinguishes omitted `eps` from explicit `eps=None`;
an explicitly supplied native alias takes precedence and has the same fallback
as `epsilon`. Omitted defaults remain AdaBelief `1e-14` and L4Adam `1e-7`.
Positive values and negative-value rejection retain their prior behavior.
No step/update formula changed. These semantics are documented in the optimizer
class docstrings.

The 28 new cases in `dev_tests/test_optimizer_epsilon.py` exercise both argument
names, zero-gradient invariance over ten steps, exact equivalence to explicit
`1e-7` over twelve steps with serialized resume, alias precedence, positive
defaults, and negative rejection. Targeted checks: **120 passed**. Default native
discovery: **189 passed, 1 CUDA skip**, with existing multiprocessing/sparse
warnings. Logs and JUnit counts are recorded in `dev_tests/validation_summary.json`
under `epsilon_constructor_followup` and `unit_tests`.

Before/after captures of ten omitted/default/positive epsilon paths across both
optimizers produced bitwise-identical parameters and complete optimizer state
for twelve updates per path. Both capture artifacts have SHA256
`69a9b23cd6379a2aaf1e4729841a9c6429fdcf59de66db53113651f13057f340`.
The existing frozen initializer/default-training fingerprints also pass and
remain unchanged. Of 65 protected source files, only the optimizer module
changed; all notebook and other recorded implementation hashes still match.
No notebook, training setting, or coordinator-owned validation document was
edited. Notebook execution was not repeated for this constructor correction.
Windows/CUDA and independent acceptance for this candidate remain coordinator/B
work; this report claims only the actual Linux CPU checks above.

Updated `nif/optimizers/external_optimizers.py` SHA256:
`e63840e3cd096a077dc4afcc47496a855afba9e9811146db3154d984c2145e2e`.


## Display-only notebook follow-up

Changed source files are `dev_tests/generate_native_notebooks.py`, native
notebooks 1, 2, and 6, and the new public regression module
`dev_tests/test_notebook_displays.py`. Only five notebook code cells changed;
all notebook IDs, outputs, metadata, narrative, and other cells are preserved.
This report and development logs/artifacts document the checks.

Tutorial 1 now evaluates separate plotting inputs at every unique training
time under `torch.no_grad()`. The original 32-row `p`, `latent`, `w`, and
coordinate comparison demonstrations remain intact. Tutorial 6 computes common
contour levels from the three comparable fields, uses one labeled colorbar,
and reserves space for titles with constrained layout. The initial 3D surfaces
in tutorials 1, 2, and 6 reserve label space using display zoom and label padding;
all x/t/u labels survive ordinary tight-crop export and avoid tick annotations.

Before implementation edits, the executed plotting regressions recorded
**10 failed** (five defects tested through notebook cells and generator output)
in `dev_tests/display_red.log`. Visual inspection then identified axis-label/tick
crowding; a stronger rendered overlap check recorded **6 failed, 4 deselected**
in `dev_tests/display_label_spacing_red.log` before the final padding adjustment.
Final focused regressions: **10 passed** in `dev_tests/display_green.log`.
Final default native suite: **199 passed, 1 CUDA skip** in
`dev_tests/display_final.log`, with JUnit results in
`dev_tests/artifacts/display_final.junit.xml`. Rendering adds Matplotlib/pyparsing
deprecation warnings; existing multiprocessing/sparse warnings remain.

These checks execute the actual plotting code, inspect rendered PNG geometry,
and compare plot values, shared color mappings, and annotation bounds. They
also verify the latent diagnostic preserves model state/mode, Torch/NumPy RNG,
input data, and downstream shared-subnet equality. The 220-file preservation
audit found only the three authorized native notebook changes: all 16 archived
TensorFlow notebooks, all data and checkpoint files, the `nif` package, README,
and the coordinator validation document remain byte-identical. Generator
outputs differ in exactly the five intended plotting cells. Tutorial 6's
scientific function body outside the plotting statements is AST-identical.

Representative figures were rendered without training into
`dev_tests/artifacts/display/`. Visually inspected all three corrected initial
surfaces with ordinary inline tight cropping, the full-domain latent diagnostic
from an existing tutorial 1 smoke checkpoint, the tutorial 6 nearly constant
prediction comparison, and a comparison from the existing tutorial 6 L2=0.01
smoke checkpoint. Uncropped surface PNGs were also produced. The predictions
and checkpoint files were not modified. These are display checks, not new
convergence results. No full workload, notebook-wide training, CUDA, hosted
Colab, or Windows execution is claimed; independent acceptance remains with
coordinator/B. Prior scientific-run summaries are retained as historical
records. The separate display artifact summary is
`dev_tests/artifacts/display_validation_summary.json`.

Final display source SHA256:

- `dev_tests/generate_native_notebooks.py`: `8a8f79e37ddea5738012ae8d3305e70572c1e57200953b384bc8a8373bab7068`
- `dev_tests/test_notebook_displays.py`: `7fc28acda7c795e68a29774c78612d85cb6cce239e970693efe2b33e37b20621`
- `tutorial/1_simple_1d_wave_pytorch.ipynb`: `b9eb9b0e3985ac8302664e80b09990880d02ef445230428485b1d08b060350c6`
- `tutorial/2_multi_scale_NIF_pytorch.ipynb`: `f5f6a35c9527413622ef693af3f5f056f87b0bf7cc3da9a2a98102b133a8addf`
- `tutorial/6_revisit_multi_scale_NIF_with_L1_L2_regularization_pytorch.ipynb`: `b2e66815e2824752ea013dde3e2cc23e3793ebb16597f3bff124e0390b010a36`


## Tutorial 8 initial-surface display follow-up

Applied the existing `SURFACE_LAYOUT` verbatim to tutorial 8's initial analytic
physical-wave surface. Implementation scope is one expression in
`dev_tests/generate_native_notebooks.py` and only the source of cell 5 in
`tutorial/8_revisit_multi_scale_NIF_with_sobolov_training_pytorch.ipynb`.
The shared layout definition itself is unchanged. Notebook metadata, outputs,
IDs, other cells, and every other notebook remain unchanged. The public display
regression was extended by adding tutorial 8 to its existing parameterization;
all previous checks and frozen fingerprints are retained.

Actual RED: **2 failed, 10 deselected**, executing tutorial 8's notebook and
generator surface cells with ordinary tight-crop rendering, recorded in
`dev_tests/tutorial8_display_red.log`. GREEN: **12 display tests passed** in
`dev_tests/tutorial8_display_green.log`. The complete public suite reports
**201 passed, 1 CUDA skip**, recorded in `dev_tests/tutorial8_display_final.log`
and `dev_tests/artifacts/tutorial8_display_final.junit.xml`. Existing
Matplotlib/pyparsing, multiprocessing, and sparse warnings remain.

Rendered and visually inspected the corrected ordinary inline tight-crop PNG
at `dev_tests/artifacts/display/tutorial8_surface_inline.png`; x/t/u labels
are fully visible and separated from tick annotations. Also exported the
uncropped canvas to `tutorial8_surface_canvas.png`. A before/after execution
check verified identical analytic arrays/scalars and unchanged NumPy/Torch RNG.
Neither 50,000-epoch fit was rerun, and no checkpoint was written.

The preservation audit (`dev_tests/tutorial8_display_scope.log`) checked 221
protected file hashes. All other notebooks (including the 16 archived sources
and prior display fixes), package sources, data, checkpoints, and frozen
fingerprints are byte-identical. One concurrent coordinator edit to
`docs/PYTORCH_VALIDATION.md` was observed and left untouched; this follow-up did
not edit that file or README. No scientific setting, optimizer, objective,
sampling, or training code changed. Independent acceptance and the retained
complete training runs remain with coordinator/B; no independent pass is claimed.

Final source SHA256 for this follow-up:

- `dev_tests/generate_native_notebooks.py`: `51b2c6518c19d04d4d5a78b642312d4dc8a365b8850a892fa4ce7205429ce00d`
- `tutorial/8_revisit_multi_scale_NIF_with_sobolov_training_pytorch.ipynb`: `225b728e829ec8bd34426bccf8ad1dfcccbe15592b4a44d58003aca8c1493cac`
- `dev_tests/test_notebook_displays.py`: `ed8263cf880696325dfcd2d24574ea1084aaeff2b028649e45b771d791c4d019`

## Reference Adam and tutorial 7 pruning correction (2026-10-23)

Implementation-owned development evidence only. The supplied baseline native
snapshot is `220771c6fb5e567fb4e38fa5021253c3682b1cca`; reference main is
`f8f9d31f4aa331b6e7dbc48758302c801441bcce`, with compatibility runtime
`ad9113df14590189b12068b6b3fa7c8c8d9187d4`. No private evaluator tests, fixtures,
host resources, TensorFlow runtime, or Git mutations were used. Coordinator/B
retain independent acceptance and full paired training ownership. Tutorial 3's
public diagnostic and AdaBelief are untouched.

### Implementation

- Added `nif.optimizers.ReferenceAdam`, exported through `nif.optimizers`.
  Epsilon is added after `sqrt(raw_variance)`, with the learning-rate multiplier
  `lr*sqrt(1-beta2**t)/(1-beta1**t)`. The default epsilon remains `1e-7`.
  Dense moments follow new Keras Adam's add-difference operation order, including
  computing Python `(1-beta)` before casting its multiplier. Parameter groups,
  closure gradients, AMSGrad, missing gradients, shared iteration, state restore,
  and dtype/device handling are covered. Ordinary `torch.optim.Adam` is unchanged.
- Migrated scientific Adam calls in tutorials 2/5/6/7/8 and their generator.
  Tutorial 4's supplemental native optimizer demonstration remains unchanged.
  All scientific learning rates, epochs, data, objectives, and regularizers remain
  unchanged. The focused API section documents the epsilon convention and limits.
- Added `nif.pruning.PolynomialMagnitudePruning` for tutorial 7's selected final
  HyperLinear kernel. It starts at step zero, increments before training forwards,
  refreshes at steps 4,000 through 200,000, and applies masks by no-gradient
  assignment before every forward and at every epoch end. There is no post-fit
  refresh. Raw gradients and optimizer moments remain active on masked entries;
  refresh ranking can reselect an entry that regrew after an optimizer update.
- Survivor count uses FP32 schedule/count arithmetic, ties-to-even rounding and
  a minimum of one; all magnitudes at or above the threshold survive. Bias is
  unmasked. Mask, threshold, step, refresh count, and schedule configuration are
  included in model state. Attach a controller before loading an active-pruning
  checkpoint; separately restore optimizer state. Stripping remasks and removes
  controls while preserving the Parameter and real sparse values.
- Tutorial 7 alone gains an epoch-end training hook, called before checkpoint
  writes, plus support for saving non-tensor schedule state. Its explanatory
  schedule text and refresh markers are corrected. Native int8 quantization,
  export/restoration, and the existing plots/domain corrections remain intact.

The supplied TFMOT 0.8.1 requirements were corroborated against the public
[pruning implementation](https://github.com/tensorflow/model-optimization/blob/v0.8.0/tensorflow_model_optimization/python/core/sparsity/keras/pruning_impl.py),
[wrapper](https://github.com/tensorflow/model-optimization/blob/v0.8.0/tensorflow_model_optimization/python/core/sparsity/keras/pruning_wrapper.py),
and [epoch-end callback](https://github.com/tensorflow/model-optimization/blob/v0.8.0/tensorflow_model_optimization/python/core/sparsity/keras/pruning_callbacks.py)
in tag v0.8.0. This is public source corroboration, not a local execution of the
independent installed reference. The Adam formula and dense slot ordering are
also documented in the linked Keras source in `PYTORCH_API.md`.

### RED and GREEN evidence

Before changing implementation, wrote and executed
`dev_tests/test_reference_adam.py` and `dev_tests/test_tutorial7_pruning.py`.
The pruning tests execute the actual notebook setup, and exercise the original
epoch callback/reparameterization on RED. Adam's RED fallback is the actual
previous notebook optimizer, ordinary PyTorch Adam. Their expectations use
explicit recurrences and hand-constructed controls, with no reference fixtures.

- `dev_tests/pruning_adam_red.log`: **15 failed, 7 passed**, retained before any
  source edits. Failures reproduce early mask timing, blocked raw gradients,
  missing reselection, threshold ties/rounding, missing epoch-end cleanup, and
  epsilon placement with tiny/ordinary gradients. Zero-gradient controls pass.
  Tests cover float64/float32, decimal and exact-binary betas, and explicitly
  float32-promoted betas in a separate float64 diagnostic.
- `dev_tests/pruning_adam_green.log`: the same unchanged tests reached
  **22 passed** after the initial fix. Their immutable pre-edit hashes remain
  in `dev_tests/pruning_adam_pre_fix_fingerprints.json`.
- A subsequent tighter slot-rounding control was written before adjusting that
  arithmetic. `dev_tests/pruning_adam_rounding_red.log` records **1 failed,
  1 passed** (float32 fails, float64 passes); its test/source hashes are retained
  in `dev_tests/pruning_adam_rounding_pre_fix_fingerprints.json`. This isolates
  slot rounding from epsilon algebra. Both controls pass in the final suite.
- `dev_tests/pruning_adam_full.log`: final full native public development suite
  **239 passed, 2 skipped** (CUDA unavailable). Existing public tests and both
  initial RED test files remain byte-identical. Supplementary lifecycle tests
  cover checkpoint continuation, masks/slots/configuration, eval/no-grad control,
  bias preservation, deepcopy/dtype movement, strip/TorchScript restoration,
  parameter groups, AMSGrad, closures, and atomic sparse-gradient rejection.
  A tiny forward-only 200,004-call schedule control observed exactly the 50
  expected refreshes; it does not execute scientific training.
- `dev_tests/pruning_adam_smoke.log`: all five affected notebooks **2/5/6/7/8
  passed** in clean CPU kernels with explicit `NIF_SMOKE=1`. Reproducible runner:
  `python dev_tests/run_pruning_adam_smoke.py`. Outputs are separately stored
  under `dev_tests/artifacts/pruning_adam/smoke/`, preserving historical runs.
  Tutorial 7 has 1,756/1,951 zero kernel entries (0.9000513 sparsity), three actual
  packed int8 Linear modules, and verified float/sparse/int8 restored predictions.
  These short-run metrics do not establish reconstruction quality or convergence.

Final suite command:
`python -m pytest dev_tests -q --junitxml=dev_tests/artifacts/pruning_adam/public_suite.junit.xml`.
Warnings are the existing plotting/multiprocessing/sparse warnings and native
TorchScript/quantization deprecation warnings. Full scientific CPU training was
not run. Scalar promotion, fused operation rounding, CUDA execution and full
cross-framework training trajectories remain for controlled independent review.

### Scope and preservation

Changed implementation files: `nif/optimizers/reference_adam.py`, its optimizer
export, `nif/pruning.py`, the notebook generator, and generated tutorials
2/5/6/7/8. Documentation changes are this report and the focused Adam API section.
Public tests, smoke runner, immutable fingerprints, logs and the candidate
manifest are under `dev_tests/`.

The pre-edit hash audit verified 52 protected files unchanged and seven authorized
existing-source changes. Tutorials 2/5/6/8 differ only in Adam calls/imports;
notebook metadata, cell IDs and outputs are preserved. All generated cell sources
match the generator. Tutorials 1/3/4, the archived TF notebooks, existing public
tests, unrelated native APIs and other optimizers remain unchanged. Scientific
data and compatibility sources were not edited. Coordinator-owned README,
validation and parity documents were left untouched by this work. `git diff
--check` passes. Final implementation/test/notebook hashes, runtime versions and
smoke results are recorded in `dev_tests/pruning_adam_candidate_manifest.json`.

## ReferenceAdam empty-gradient boundary and local scalar cache follow-up (2026-10-23)

Follow-up to the independently frozen candidate009 requirements supplied by the
coordinator. These are implementation-owned public development checks, not an
independent acceptance result. No evaluator fixtures or tests were accessed.

Before implementation changes, added
`dev_tests/test_reference_adam_empty_gradients.py`. The retained
`dev_tests/reference_adam_empty_red.log` records **5 failed, 3 passed**: all-None
calls failed to raise (both fresh/warmed state and closure/no-closure cases), and
an empty call shifted bias correction on the next real update. Partial missing
gradients, real zero gradients, and a closure creating the first gradient already
passed. Added the smallest guard to the existing validation pass, after closure
evaluation and before iteration, slot or parameter mutation. An all-None call
now raises `ValueError`; a zero tensor is still a real gradient. The unchanged
regression then passed **8/8**, retained in `reference_adam_empty_green.log`.

Only after that GREEN run, froze the guarded uncached source in
`dev_tests/fixtures/reference_adam_guarded_uncached.py` and wrote the separate
`dev_tests/test_reference_adam_cache.py` regression. This fixture is our own
public source snapshot. Its two tests passed before optimization, as recorded in
`reference_adam_cache_before.log`. The cache now stores alpha tensors only within
one step and one parameter group, keyed by the actual parameter device and dtype.
The alpha expression and operation order are unchanged; m/v arithmetic is
unchanged. Coefficients cannot persist across iterations, group schedules or
checkpoint restoration.

Cache tests compare raw tensor bytes (including all optimizer slots), parameters
and serialized state for 32 updates with float32/float64 mixed within groups,
distinct group betas/epsilon/LRs, changing LRs and betas, zero/tiny/ordinary and
partial missing gradients, AMSGrad enabled/disabled, and two save/load round
trips. Both comparisons remain bitwise identical to the frozen uncached source.
The focused suite passed **40 tests, 1 CUDA skip** in
`dev_tests/reference_adam_followup_focused.log`. The final complete native public
suite passed **249 tests, 2 CUDA skips** in
`dev_tests/reference_adam_followup_full.log`; XML is retained in
`dev_tests/artifacts/reference_adam_followup.junit.xml`.

A reproducible CPU-only microbenchmark is
`python dev_tests/benchmark_reference_adam_cache.py`; raw repeats, runtime
versions, and medians are in `dev_tests/reference_adam_cache_benchmark.json` and
its companion log. With one CPU thread and float32 parameters, three alternating
measurement rounds gave median per-step times:

| Parameters × scalars | Uncached | Cached | Uncached/cached |
| --- | ---: | ---: | ---: |
| 64 × 64 | 2.349 ms | 1.133 ms | 2.07 |
| 512 × 4 | 20.110 ms | 9.544 ms | 2.11 |

These are local execution-cost measurements for small tensors, not GPU speed or
full scientific training evidence. No host GPU was accessed. The previous five
all-cell smoke runs were not repeated: their path has real gradients, and the
cache's parameter/slot trajectories are bitwise equivalent in the public controls.

The pruning controller docstring and focused API wording now specify its
supported eager workflow: one forward followed by its backward per optimizer
update, as in tutorial 7. Multiple pending forward graphs followed by a combined
backward with coordinate gradients are unsupported because masking mutates the
kernel in place. No pruning architecture or tutorial flow changed.

Scope: changed `nif/optimizers/reference_adam.py`, pruning docstring wording in
`nif/pruning.py`, focused `docs/PYTORCH_API.md` sections, this report, and newly
added public follow-up tests/fixture/benchmark/evidence. Generator, all notebooks,
scientific settings/data, AdaBelief/model code, other optimizers and archived
sources remain unchanged from the preceding candidate. Coordinator README,
parity and validation documents were not edited. All previous frozen public test
sources and fingerprint files are retained unchanged. `git diff --check` passes.

Immutable pre-edit hashes are in
`dev_tests/reference_adam_followup_pre_fix_fingerprints.json`; the pre-cache
source/test hashes are in `dev_tests/reference_adam_cache_pre_fix_fingerprints.json`.
Final changed-file SHA256 values, environment, test counts, benchmark results and
preservation audit are recorded in
`dev_tests/reference_adam_followup_manifest.json`. Independent acceptance and
full cross-framework/GPU verification remain with coordinator/B.

## Tutorial 1 actual-current-Colab GC default correction (2026-10-23)

Read the current port contract and reference clarification in the coordinator's
parity document. The primary tutorial 1 reference is its separate preserved
`bootstrap_tutorial1.py` runtime: its legacy public GC override is inactive in
modern tf-keras fit. The suite bootstrap's active private-gradient GC is a
different reference, required by tutorial 3. This correction follows that
source-backed requirement before any heavy paired tutorial 1 outcome; no
TensorFlow reference/runtime was modified or executed here. The intended-GC
configuration remains an explicit optional diagnostic, not the primary default.

### Behavioral RED, correction and GREEN

Before changing implementation, added
`dev_tests/test_tutorial1_gc_behavior.py`. It executes the real helper loop,
model/optimizer construction, scheduler and training call extracted separately
from the native notebook and generator. A controlled single batch and epoch use
the actual NIF/AdaBelief path, with deterministic parameter values. A step hook
captures the gradient consumed by the optimizer. An otherwise identical direct
raw-gradient update supplies the primary no-GC control; an explicitly centralized
control checks opt-in behavior. Assertions compare consumed temporal-kernel
gradients, every parameter update and every optimizer slot.

`dev_tests/tutorial1_gc_red.log` records **2 failed, 2 passed** before editing:
both default paths consumed a zeroed first temporal-kernel gradient instead of
the nonzero raw gradient. Both explicit-GC controls passed. The new regression's
unchanged source hash is retained in
`dev_tests/tutorial1_gc_red_test_fingerprint.json`.

The only tutorial changes are the named selector
`use_gradient_centralization = False`, passing it into the AdaBelief training
call, and concise teaching text that enabling GC changes the actual current TF
tutorial 1 behavior. The introduction now describes the default accurately.
All original comments are retained. The generator and only tutorial 1's native
artifact were updated. Default 5,000 AdaBelief epochs, learning rates/schedule,
batch size, data, model, loss, weights, 200 L-BFGS rounds/max_iter=1000, solver
options, checkpoint/subnet and plotting code remain unchanged. Tutorial 3 GC
remains enabled, and no package/model/optimizer implementation changed.

`dev_tests/tutorial1_gc_green.log` records **11 passed**: four behavioral cases,
three impact/scope controls, and four existing native L-BFGS control tests.
The raw-gradient default now moves the temporal kernel; the explicit-GC choice
still consumes zero temporal-kernel gradients in this one-input control.

### Explicit historical scope transition

The exact prior `dev_tests/test_pruning_adam_state.py` source is preserved as
`dev_tests/fixtures/tutorial1_gc_scope_v1/test_pruning_adam_state.py.txt` (a
non-collected artifact), SHA256
`60e83c210615ca5a4d478a484d5ad08940a6e8841ac1d597780e44f13186c268`.
The same directory retains its historical **239 passed, 2 skipped** suite log,
pre-change notebook/generator bytes and a hash manifest. Original RED-test and
historical fingerprints were not regenerated or edited.

The current guard is explicitly named `_v2`. Its original mathematical tests
and frozen Adam/pruning assertions are untouched; tutorials 3/4 retain their
whole-file guards. Tutorial 1 is checked by reconstructing exactly the authorized
GC-selector/call and teaching-note changes from the preserved source. The guard
compares the entire notebook (including metadata/outputs/IDs and every other
cell) and generator source. A separate test verifies that this scope-assertion
replacement and function version are the only changes to the old test file.
This is a bounded, pre-outcome scope transition, not a changed numerical target.

### Execution and preservation evidence

- Ran the appropriate full native public suite once after the correction:
  **256 passed, 2 CUDA skips**, retained in `dev_tests/tutorial1_gc_full.log` and
  `dev_tests/artifacts/tutorial1_gc/public_suite.junit.xml`.
- Executed all **18 tutorial 1 code cells** in a clean CPU kernel with explicit
  `NIF_SMOKE=1`: two AdaBelief epochs and the existing separate smoke L-BFGS
  setting of one round/max_iter=2. Checkpoint restoration, extracted subnet
  consistency, plots, and the full smoke notebook completed successfully.
  Runner: `python dev_tests/run_tutorial1_gc_smoke.py`. Log and public summary:
  `dev_tests/tutorial1_gc_smoke.log` and `dev_tests/tutorial1_gc_smoke_summary.json`.
  Executed notebook/checkpoints/plots are separate under
  `dev_tests/artifacts/tutorial1_gc/smoke/`; historical runs are preserved.
- Audited that every generated cell source matches its retained notebook, without
  writing other notebooks. The scope audit preserves all 16 original/Colab TF
  notebooks, every other native notebook, package/optimizer/model code,
  scientific data, compatibility sources and earlier frozen tests/evidence.
  Coordinator contracts, README, API, validation and parity docs were untouched
  by this task. Git was used read-only; `git diff --check` passes.

Final changed-file hashes and scope verification are in
`dev_tests/tutorial1_gc_candidate_manifest.json`; pre-change scope hashes are in
`dev_tests/tutorial1_gc_pre_fix_fingerprints.json`. The corrected notebook SHA256
is `9c64c3de750de791df2b8f4c7f74a93e5eb4fee618ffe63236d1ddf6ec91a438`.
No private fixtures/tests were accessed. No 5,000-epoch or full-L-BFGS workload,
GPU training, paired TF training, convergence or independent acceptance is
claimed. Coordinator/B retain those evaluations.

## AdaBelief parameter-dtype scalar arithmetic correction (2026-10-23)

The coordinator reported that the first controlled tutorial 3 full-600 pair
failed the predeclared whole-field/input-Jacobian criteria despite close
aggregate weighted MSE. This work addresses the separately isolated arithmetic
precision defect; it does not tune training or claim to explain/rescue that
full-training outcome.

Read the actual preserved dense and sparse AdaBelief implementations from
`ad9113df14590189b12068b6b3fa7c8c8d9187d4:nif/optimizers/external_optimizers.py`
using read-only Git. A byte-identical public source snapshot is retained in
`dev_tests/fixtures/adabelief_dtype/tf_external_optimizers_ad9113d.py.txt`, SHA256
`016b79ec31e9e66418c39c4fc1e0ba174d054798bc6c59b6aef0ff7bed41352a`.
The reference casts beta/step scalars into the parameter dtype before computing
powers, bias corrections, SMA and rectification. It uses **updated momentum**
in the variance recurrence. No TensorFlow source or runtime was modified.

### RED and smallest implementation correction

Before editing implementation, wrote and ran
`dev_tests/test_adabelief_scalar_dtype.py`: **20 failed, 35 passed**, retained in
`dev_tests/adabelief_dtype_red.log`. Tests derive scalar arithmetic from the
preserved source, using NumPy float32/float64 scalar operations in source order,
with explicit matched effective beta values. They cover rectified/unrectified
updates, AMSGrad on/off, steps 1/5/6/7/20/50, zero/tiny/ordinary gradients,
moment values, default-beta bias correction isolated from measured moments,
sparse observed-row updates, mixed-dtype parameter groups, changing learning
rates and checkpoint continuation across the transition. Float64 controls
passed before the fix. The RED test/source fingerprints are retained in
`dev_tests/adabelief_dtype_red_fingerprints.json` and remain unchanged.

Only native `AdaBeliefOptimizer.step` arithmetic and its explanatory docstring
changed in `nif/optimizers/external_optimizers.py`. Beta powers, the local step,
bias-correction factors, SMA/rho and rectification now use scalar tensors on
the actual parameter device/dtype. A per-call, per-group cache keyed by
`(device, dtype)` shares identical expressions. Tensor selection avoids adding
per-parameter host synchronization. Existing branch safeguards are retained.
The moment recurrence, epsilon, parameter/state API, group counters, sparse row
masking, LR/weight-decay schedules, clipping, and other optimizers are unchanged.
No hidden setting, hyperparameter or scientific workload changed.

At step 6, explicit float32 arithmetic produces rho `6.00048828125` and
r `0.02586868219077587`; evaluating the same effective betas in Python/double
precision instead produces about `5.994163826821023` and
`0.025820946567591475`. The control reproduces this source-backed difference.
Further details are in [PYTORCH_ADABELIEF_DTYPE_CORRECTION.md](PYTORCH_ADABELIEF_DTYPE_CORRECTION.md).

### GREEN controls, smoke and honest full-suite result

- Initial GREEN: **102 passed** in `dev_tests/adabelief_dtype_green.log`, covering
  the unchanged new RED tests and existing optimizer/epsilon tests. Existing
  float64 goldens pass without changing their values or tolerances.
- Final focused verification: **112 passed** in
  `dev_tests/adabelief_dtype_focused_final.log`. Additional controls compare
  momentum/variance/AMSGrad slots bitwise with the frozen pre-correction native
  optimizer through dense/sparse updates, GC/clipping, LR and weight-decay
  schedules, warmup/decay and both dtypes. Moments remain bitwise unchanged.
  An AST scope check verifies all package-module content outside AdaBelief's
  step/docstring is unchanged. Save/load continuation is bitwise consistent in
  the mixed-dtype/groups regression before and after rectification begins.
- Final complete public suite: **319 passed, 2 skipped, 2 failed**, retained in
  `dev_tests/adabelief_dtype_full_final.log` and
  `dev_tests/artifacts/adabelief_dtype/public_suite.junit.xml`. The earlier
  complete run (before the ten extra preservation controls) is also retained in
  `dev_tests/adabelief_dtype_full_scope_red.log`: 309 passed, 2 skips, 2 failures.
  Both failures are unchanged historical assertions described below. No test
  was skipped, xfailed, weakened or assigned new expected values to hide them.
- Actual-cell CPU smoke: tutorials **1 and 3 passed all 18 and 12 code cells**.
  Tutorial 1 retained two smoke AdaBelief epochs and its separate one-round,
  max_iter=2 L-BFGS path, checkpoints, subnets and plots. Tutorial 3 retained
  two smoke epochs on 4,096 rows, active GC, full-field prediction/plots and
  checkpoint restoration. Runner: `python dev_tests/run_adabelief_dtype_smoke.py`.
  Log/summary: `dev_tests/adabelief_dtype_smoke.log` and
  `dev_tests/adabelief_dtype_smoke_summary.json`; separate artifacts are under
  `dev_tests/artifacts/adabelief_dtype/smoke/`. Smoke MSE is not convergence evidence.

### Preserved historical assertion conflicts

1. `test_public_api_regressions.py::test_existing_2d_rng_and_default_training_trajectories_unchanged`
   demands the old exact float32 parameter hashes after three updates. The NIF
   hash first differs at update 3, where parameter-dtype bias correction can
   change update rounding even before rectification. Its expected old hash is
   `17e08aabd62ded996d3b0ba3eee9ab7c148d943ae8a3955528a29b095f163cf8`;
   the corrected run gives
   `6680584bf1a5ef2e0db80da29911707594744ca947dfb8f9e15d27d602d8d0bb`.
   The original test and its fingerprint files are untouched. A separately
   labeled diagnostic replays that exact test body with CURRENT model/initializer
   code and the frozen BEFORE optimizer; all its original hashes pass. This
   diagnostic binds the old optimizer only inside its local copied function,
   without monkeypatching native exports. It confirms the historical optimizer
   context while leaving the original collected test and failure visible.
2. `test_tutorial1_gc_scope.py::test_other_notebooks_package_sources_and_previous_frozen_regressions_unchanged`
   demands the entire optimizer module retain its preceding-phase hash
   `e63840e3cd096a077dc4afcc47496a855afba9e9811146db3154d984c2145e2e`.
   The authorized arithmetic correction necessarily changes that file. This
   scope test and all its historical evidence also remain unchanged.

Both conflicts were reported during implementation. They were not converted
to acceptance passes or used to alter scientific/numerical expected values.
The complete public suite is therefore **not claimed green**.

### Scope, provenance and limits

Final native optimizer module SHA256:
`f0976ef6c1abf3040cc65d638604a0c3aa1b38b45bf2b4c0faaa13d20c17ce34`.
All earlier tests, RED/GREEN logs, fingerprints, generator/notebooks, scientific
data, models, other optimizers and original TensorFlow source/notebooks remain
unchanged. No Git mutation, private evaluator access, TensorFlow runtime import,
host GPU use or full-training rerun occurred. Coordinator documentation was not
edited. A concurrent `.gitattributes` change was observed and left untouched.
`git diff --check` passes. The new documentation, public tests, smoke runner,
source snapshots and evidence are owned development artifacts.

Pre-edit hashes are in `dev_tests/adabelief_dtype_pre_fix_fingerprints.json`;
final hashes, environment, test counts, smoke outputs and preservation audit
are in `dev_tests/adabelief_dtype_candidate_manifest.json`. CUDA/backend power
rounding and the failed full-training scientific criteria require independent
controlled reevaluation. No full-training parity or independent acceptance is
claimed.

## AdaBelief atomic empty calls, explicit public-test succession and UTF-8 (2026-10-23)

This coordinator-authorized follow-up retains the preceding scalar-precision
source and all RED/GREEN logs, pre-fix snapshots and manifests. It does not revise
that phase's reported two failures retrospectively. The root `.gitattributes`
fixture byte-preservation change remains intact. No TensorFlow source/notebooks,
scientific settings, private tests/fixtures, Git state or GPU were changed/used.

### Atomic empty-gradient RED and correction

Added `dev_tests/test_adabelief_empty_gradients.py` before implementation edits.
`dev_tests/adabelief_empty_red.log` records **8 failed, 2 passed**: the old empty
call failed to reject, advanced clocks/called schedules, and contaminated the
next real update and restored continuation. Controls retain fresh and warmed
states, closures clearing gradients, partial None, zero-gradient tensors,
fixed groups, a late-added group, and a closure supplying its first gradient.

The native implementation change is a two-line all-None guard immediately after
closure evaluation and before `_gradients`, schedules or state mutation. Actual
zero tensors remain valid updates. Fixed pre-existing groups continue advancing
their existing per-group clocks on a real update; late-added groups start at
their own step one. No global-clock redesign occurred. AdaBelief's corrected
scalar arithmetic, all dense/sparse expressions and Lion/L4Adam are unchanged.
The docstring states this boundary and clock behavior explicitly.

### UTF-8 RED and correction

`dev_tests/test_scope_helper_utf8.py` runs a controlled GBK-default `Path.read_text`
simulation against the real preserved en-dash-containing UTF-8 archive.
`dev_tests/scope_helper_utf8_red.log` records **1 failed, 2 passed**. A separate
scope-test control adds an identical harmless UTF-8 comment to temporary copies
of its two source inputs, preserving their source-transform relation without
changing actual archives. Its pre-edit run is retained in
`dev_tests/scope_helper_utf8_archives_red.log`: **1 failed** with
`UnicodeDecodeError`. This isolates the scope test's own default-encoding reads.

Changed only the relevant `.read_text()` calls in `tutorial1_gc_scope.py` and
`test_tutorial1_gc_scope.py` to explicit `encoding='utf-8'`; the successor helper's
reads are explicit UTF-8 as well. All historical fixture bytes/hashes remain
unchanged. The initial guard/UTF-8/scalar GREEN run reports **69 passed** in
`dev_tests/adabelief_boundary_green.log`. All new RED-test source fingerprints
remain unchanged in their retained fingerprint JSON files. Windows behavior is
simulated; no native Windows run is claimed.

### Authorized historical test succession

Preserved exact original source for both formerly failing collected tests, the
helper and scalar-phase diagnostic test, original expected JSON/environment,
the scalar-phase native source, and the original raw **2 failed, 319 passed,
2 skipped** log in `dev_tests/fixtures/adabelief_boundary_v3/`. Its `manifest.json`
records byte hashes. Original files such as `public_api_pre_fix_fingerprints.json`,
the scalar RED tests and phase manifests were not regenerated or overwritten.

The current collected trajectory test is explicitly named
`test_current_initializers_rng_and_historical_adabelief_trajectories_v2`. Its
initializer/RNG and initial-model assertions still use CURRENT model code and
all original expected hashes; an explicit pre-training initial-model assertion
also uses the original hash. Its historical updates bind only the frozen
PRE-correction AdaBelief from the hashed native snapshot. All original training
hash assertions are still executed. A non-collected compatibility wrapper under
the old callable name delegates to the complete v2 test for the existing
build-portability regression; it does not replace or disable the collected v2
assertions. The frozen source-derived parameter-dtype tests independently cover
the CURRENT corrected optimizer, with unchanged mathematical expectations.

The earlier scalar-phase replay diagnostic now reads the archived original test
body, rather than searching the renamed current body. No numerical assertions
in that diagnostic changed. All other tests in those files remain intact.

The scope successor is explicitly named
`test_other_notebooks_package_sources_and_frozen_expectations_phase_v3`. Its
helper `dev_tests/adabelief_phase_scope.py` checks exact bounded source transforms
against archived originals, then retains the previous notebook/source/data and
frozen-expectation protections. It also verifies preservation against this
follow-up's starting hashes, including `.gitattributes`. The only allowed
existing source files are:

- `nif/optimizers/external_optimizers.py`: empty-gradient guard and docstring only.
- `dev_tests/test_public_api_regressions.py`: explicitly historical optimizer
  binding/version, original initial-model hash assertion and compatibility entry.
- `dev_tests/test_adabelief_dtype_preservation.py`: archived-original source read
  and corresponding diagnostic docstring only.
- `dev_tests/tutorial1_gc_scope.py`: explicit UTF-8 reads only.
- `dev_tests/test_tutorial1_gc_scope.py`: explicit UTF-8 reads and the versioned,
  exact scope succession only.

Negative controls prove that unrelated model changes and a changed AdaBelief
moment expression are rejected. Numerical tolerances/goldens and private
acceptance gates were not changed. History versus current behavior is made
explicit in names, documentation and the new manifest, rather than silently
assigning new output fingerprints.

### Final validation and limits

- `dev_tests/adabelief_boundary_succession_green.log`: **170 focused passes**,
  covering no-gradient, UTF-8, frozen scalar math, historical replay, initializer
  regressions, prior numerical controls and build portability.
- `dev_tests/adabelief_boundary_equivalence_green.log`: **7 passes**, including
  bitwise parameters/slots/state equivalence to the frozen scalar-phase optimizer
  on real updates, float32/float64 groups, changing LRs, partial gradients,
  rectified/unrectified AMSGrad and checkpoint continuation. The scope successor
  passes the controlled GBK simulation and rejects unauthorized changes.
- Final full PUBLIC suite: **342 passed, 2 CUDA skips**, retained in
  `dev_tests/adabelief_boundary_full.log` and
  `dev_tests/artifacts/adabelief_boundary/public_suite.junit.xml`. The historical
  two-failure log remains preserved and is not relabeled as a pass.
- Actual-cell CPU smoke: T1 and T3 passed **18 and 12 code cells** respectively.
  T1 retains its separate two-epoch/one-round/max_iter=2 smoke workload, subnets,
  checkpoint restoration and plots; T3 retains two epochs on 4,096 training rows,
  active GC, full-field prediction/plots and checkpoint restoration. Smoke metrics
  match the scalar-phase smoke results in this environment. Runner/log/summary:
  `run_adabelief_boundary_smoke.py`, `adabelief_boundary_smoke.log`, and
  `adabelief_boundary_smoke_summary.json` under `dev_tests/`. Artifacts are separate
  under `dev_tests/artifacts/adabelief_boundary/smoke/`.

Coordinator-owned documents were not edited; a concurrent parity-document edit
was observed and left untouched. All notebooks, settings, model/data/reference
sources, other optimizers and earlier numerical expectations remain unchanged.
`git diff --check` passes. Focused documentation is in
[PYTORCH_ADABELIEF_BOUNDARY_SUCCESSION.md](PYTORCH_ADABELIEF_BOUNDARY_SUCCESSION.md).
Final source hashes, archive provenance and scope are recorded in
`dev_tests/adabelief_boundary_candidate_manifest.json`. Full scientific training,
GPU execution and independent cross-framework acceptance remain unclaimed.


## AdaBelief parameter-dtype moment coefficients, public phase v4

This follow-up corrects the nominal constructor's moment coefficients, separately
from the previous effective-beta scalar-power/SMA controls. Betas are cast to each
parameter's dtype/device before forming `1-beta`. The ephemeral scalar cache now
also holds those tensors and complements. Explicit moment products preserve the
source's multiply-then-add operation order; a BEFORE-fix control discriminates
that order from `add_(alpha=...)` rounding even with matched coefficients. The
residual uses updated momentum. The parameter update expression remains unchanged.
Native float64 preserves caller beta precision; legacy float32-stored beta values
promoted to float64 are a separately labeled control.

The source-derived oracle and RED log were frozen before implementation:
`dev_tests/adabelief_moment_red.log` records **43 failed, 3 passed**. The same
oracle plus unchanged scalar and atomic-guard controls now gives **111 passed**.
Nominal and explicitly float32-materialized beta groups have bitwise-equal
float32 weights and all moment/AMSGrad slots through the 24-step controls.
Float64 nominal and legacy representations remain distinct as expected.

Old assertions about unchanged moments/trajectories and old exact scope produced
**15 failures, 5 passes**, retained in
`dev_tests/adabelief_moment_historical_scope_red.log`. Their full original test
sources, optimizer snapshot, expected JSON and raw failures are preserved
verbatim under `dev_tests/fixtures/adabelief_moment_v4/`, with a SHA-256 manifest.
Versioned historical tests actually run the PRE-moment optimizer and retain all
original numerical assertions. Current moments/weights use the frozen source
oracle. Scope v4 permits exact transformations in the optimizer and three
phase-specific tests; its model/optimizer negative controls still reject
unrelated changes. The old v3 helper and prior artifacts remain bytewise intact.
Historical replay/scope/API/UTF-8 checks give **97 passed**.

Full PUBLIC suite: **390 passed, 2 skipped** (CUDA unavailable), raw log
`dev_tests/adabelief_moment_full.log`. Actual T1/T3 CPU smoke completes **18/12
cells**, preserving separate-smoke L-BFGS, checkpoint/subnet/plot paths. Two
additional actual-notebook-helper tests train real models with backpropagation,
serialize model and optimizer checkpoints, and verify bitwise continuation of
weights, slots and predictions across the rectification transition. No notebook,
scientific setting, initializer, model, other optimizer, reference or data changed.
Root's `.gitattributes` fixture-preservation edit remains intact.

Final native optimizer SHA-256:
`d855f0e9f516c4048500e32dd764d0702ec9f1eea1140ee31e17280c87e1bd19`.
Complete provenance, bounded source diff, logs and hashes are recorded in
`dev_tests/adabelief_moment_candidate_manifest.json` and
`docs/PYTORCH_ADABELIEF_MOMENT_CORRECTION.md`. Sparse observed-row behavior and
native epsilon-addition order are retained; bitwise TensorFlow scatter-kernel or
assign-sub rounding equivalence is not asserted. CPU development results do not
establish full-training parity or independent acceptance; no GPU was used.
