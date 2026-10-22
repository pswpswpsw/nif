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
Final unit tests: **189 passed, 1 skipped (CUDA unavailable)**. Notebook hashes are recorded in
`dev_tests/validation_summary.json`.

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
