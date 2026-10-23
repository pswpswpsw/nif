# AdaBelief parameter-dtype moment coefficients — public phase v4

The nominal default constructor now forms `1 - beta` after converting beta to
parameter dtype/device. The previous scalar correction fixed powers, bias
correction and SMA/rectification with aligned effective betas; it did not establish
literal-default moment arithmetic. No tutorial beta override is introduced.

For float32, the corrected complements of nominal `.9` and `.999` are
`0.10000002384185791` and `0.0009999871253967285`. Computing the subtraction in
Python before a float32 cast instead gives `0.10000000149011612` and
`0.0010000000474974513`. That difference changes raw momentum and variance from
the first step. Nominal and explicitly float32-materialized beta groups now have
bitwise-equal float32 weights, momentum, variance and AMSGrad slots through the
24-step public controls, with and without rectification.

Native float64 parameters retain the caller's beta precision. Float64 nominal
`.9/.999` and explicitly promoted legacy float32 beta representations are separate
source-oracle controls and intentionally have distinct moments. The native
optimizer does not unconditionally round caller hyperparameters to float32.

## Source and bounded implementation

The preserved TensorFlow source is
`ad9113df14590189b12068b6b3fa7c8c8d9187d4:nif/optimizers/external_optimizers.py`,
verified byte-for-byte against the existing archive (SHA-256
`016b79ec31e9e66418c39c4fc1e0ba174d054798bc6c59b6aef0ff7bed41352a`).
Its dense recurrence at lines 462–493 casts betas before subtraction, forms
`beta1*m + (1-beta1)*grad`, and uses **updated** momentum in
`beta2*v + (1-beta2)*(grad-m)**2 + epsilon`.

The correction extends the existing ephemeral per-step/group/device/dtype cache
with beta tensors and their complements. Each moment contribution is explicitly
multiplied before addition. This operation order is supported by the preserved
source and a BEFORE-fix discriminating test: matched-coefficient operands expose
the difference between separate multiply/add and native `add_(alpha=...)` fusion
in both float32 and float64. The NumPy source-derived oracle requires exact raw
moment and AMSGrad values; it is not a snapshot of corrected native output.

The parameter update remains `p.add_(direction, alpha=-lr)`. Its rounding is
checked with the pre-fix-frozen oracle tolerances and is not claimed to be a
bitwise TensorFlow assign-sub equivalent. Sparse observed-row masking, dense
momentum decay, and the existing native epsilon-addition order remain intact;
this phase does not claim bitwise equality with TensorFlow scatter-kernel
addition order. Scalar powers/SMA, the atomic all-None guard, group clocks,
schedules, decay, clipping, residual definition and epsilon placements are
unchanged. Lion and L4Adam are unchanged. No TensorFlow runtime is imported.

## Historical test succession

`dev_tests/fixtures/adabelief_moment_v4/manifest.json` identifies verbatim archived
collected test sources, the PRE-moment optimizer, original expected JSON, the
previous passing public log, and the raw new RED logs. Earlier manifests, logs,
archives and expected hashes retain their original bytes.

The old unchanged-slot tests actually execute the frozen PRE-moment optimizer
against their prior optimizer, with every assertion retained. Likewise the old
atomic-phase bitwise trajectory/resume tests execute PRE-moment versus the scalar
phase. Test names/docstrings now label those historical claims. Current moments,
weights, AMSGrad, mixed dtype behavior and checkpoint continuation are checked by
`test_adabelief_moment_dtype.py`, frozen before implementation with its RED log.
The scalar oracle, all-None/UTF-8 controls, current initializer/RNG checks and
original historical trajectory assertions remain unchanged.

The v3 scope helper remains verbatim. Its v4 successor permits exact text
transformations in only four preexisting files: the AdaBelief module and three
phase-specific test sources. It checks the protected pre-phase tree and rejects
unrelated model or optimizer edits. No tolerance, numerical expectation,
initializer, notebook, training setting, data or reference source is changed.
The complete bounded diff is `dev_tests/adabelief_moment_scope.diff`.

## Local evidence

- Before implementation: **43 failed, 3 passed**, retained in
  `dev_tests/adabelief_moment_red.log`.
- Corrected moment/scalar/atomic tests: **111 passed**, retained in
  `dev_tests/adabelief_moment_green.log`.
- Before historical succession: **15 failed, 5 passed**, retained verbatim in
  `dev_tests/adabelief_moment_historical_scope_red.log`. Twelve failures assert
  old slot/trajectory equality; three enforce the earlier exact source scope.
- Explicit historical replay, bounded scope, original API and UTF-8 controls:
  **97 passed**, `dev_tests/adabelief_moment_succession_green.log`.
- Real T1/T3 model training through actual notebook helpers, serialized
  model/optimizer checkpoint and continued backpropagation: **2 passed**,
  `dev_tests/adabelief_moment_checkpoint_green.log`. Ten updates cross the
  rectification transition; continued weights, slots and predictions are bitwise
  equal to uninterrupted execution.
- Actual notebook CPU smoke: T1 **18 cells**, T3 **12 cells**, including existing
  checkpoint, subnet and plot paths. T1 retains its separate-smoke L-BFGS round
  (`max_iter=2`). Notebook bytes and full workload defaults are unchanged.

Full public suite: **390 passed, 2 skipped** (CUDA unavailable), retained in
`dev_tests/adabelief_moment_full.log` and its JUnit artifact. Results and phase hashes are recorded in
`dev_tests/adabelief_moment_candidate_manifest.json`. This is CPU development
validation; no GPU, full-training parity or independent acceptance is claimed.
