# AdaBelief scalar arithmetic correction

The preserved TensorFlow AdaBelief at
`ad9113df14590189b12068b6b3fa7c8c8d9187d4:nif/optimizers/external_optimizers.py`
casts its step and hyperparameter scalars to each parameter's dtype before
calculating beta powers, bias correction, SMA and rectification. Its dense and
sparse methods use the same scalar expressions. Native AdaBelief previously
calculated these expressions in Python double precision.

The native correction evaluates those scalars with tensors on each parameter's
actual device and dtype, in the reference expression order. For float32,
step 6 and the actual float32-declared beta values give SMA/rho
`6.00048828125` and rectification coefficient `0.02586868219077587`. Evaluating
the same effective beta values in double precision instead gives approximately
`5.994163826821023` and `0.025820946567591475`. Bias correction is also evaluated
in the parameter dtype when rectification is disabled or has not yet begun.

A cache shares identical scalar tensors only within one step and one parameter
group, keyed by device and dtype. It is recreated on every call; loading a
checkpoint or changing the learning-rate schedule cannot reuse stale values.
Tensor selection retains the existing branch rule without a host scalar
conversion or synchronization per parameter. The existing native `rho > 4`
safety condition is retained. Unselected early-step square roots may be NaN,
as in the reference calculation; tensor selection excludes them from updates.

The variance recurrence still uses **updated momentum**. Moment arithmetic,
epsilon placement/defaults, parameter groups/counters, sparse row masking,
learning-rate and weight-decay schedules, clipping, and other optimizers are
unchanged. This correction does not introduce an optimizer option, change
training settings, or modify any tutorial or TensorFlow source.

Public implementation-owned evidence is recorded in
[the candidate report](PYTORCH_CANDIDATE_REPORT.md) and
[the manifest](../dev_tests/adabelief_dtype_candidate_manifest.json).
The source-derived controls use explicit parameter-dtype arithmetic without a
TensorFlow runtime. Float64 goldens remain intact. CPU controls and smoke runs
do not establish CUDA arithmetic equivalence or full paired-training quality.
Two unchanged historical public guards intentionally remain reported as
failures: one demands the previous float32 training hashes, and one demands the
previous package-source hash. Their expected values have not been rewritten.
