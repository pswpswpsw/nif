# Native PyTorch validation

Status: **full integration validation in progress**, 2026-10-22.
Implementation branch: `pytorch`; TensorFlow reference branch: `main`.

## Independent package acceptance

Agent B keeps its acceptance tests and immutable reference fixtures outside the
repository. Agent A works in a separate container: its repository is writable,
Git metadata is read-only, and coordinator files are masked. The evaluator
directory and user drives are not mounted.
Only sanitized failure descriptions are returned to the implementation agent.
The initial 177-test suite was frozen before implementation. Additional
regression cases were frozen before their corresponding fixes; evaluator
corrections retain versioned provenance and coordinator review.

The corrected native implementation passed **247/247 checks on CUDA**, with no
skips or failures. Its CPU result was **244 passed, 3 CUDA-only checks skipped**.
Coverage includes all three architectures, generated-weight layouts against
TensorFlow numerical references, shared subnets, derivatives and higher-order
autograd, regularization reductions, optimizer formulas and restoration,
normalization, actual legacy TFRecord interoperability, and initializer options.

Separately, the public repository development suite was run on native Windows
CPU: **188 passed, 2 environment-specific checks skipped**. These public tests
are distinct from Agent B's private acceptance suite.

Optional initializer, epsilon-fallback and L4Adam/gradient-centralization corrections were checked
against the earlier full-run source. Eleven actual tutorial configurations had
bitwise-identical initialization, RNG state, forward values, input/parameter
gradients, regularization, and one optimizer update on the same CPU and CUDA
platforms. Those corrections do not change the ongoing tutorial trajectories.

## Full tutorial execution

These are full runs independently observed by Agent B. Each starts in a fresh
kernel. Shortened development smoke runs are recorded separately and do not
count as full passes. Any checkpoint continuation is identified below.

| Tutorial | Full workload | Independent result |
|---|---|---|
| 1 | 5,000 AdaBelief epochs; 200 L-BFGS rounds, max_iter 1,000 | Passed; actual steps, checkpoints and figures checked |
| 2 | Two 5,000-epoch fits | Passed; 20,000 Adam steps per fit; figures checked |
| 3 | 600 epochs over all 191,775 weighted cylinder-flow samples | Passed; 112,800 AdaBelief steps and 115,065,000 trained rows |
| 4 | Full selected Jacobian/Hessian examples and differentiation | Passed; figure checked |
| 5 | One million rows, 100 TFRecord shards, two complete passes | Passed; 2,000,000 actual trained rows, 200 shard visits |
| 6 | Nine 3,000-epoch fits and superresolution | Passed; 108,000 Adam steps, 54,000,000 trained rows; figures checked |
| 7 | 5,000 baseline and 50,000 pruning epochs; int8/export inference | Passed with checkpoint continuation described below; 90.0051% actual sparsity |
| 8 | Two 50,000-epoch fits; Sobolev objective and physical superresolution | Running |

Tutorial 1 retains the original tolerance/history controls. Native PyTorch and
TFP line-search and stopping conventions differ; the notebook documents those
differences. Tutorial 7 uses a conventional Frobenius relative-MSE metric rather
than the archived matrix-spectral-norm ratio; see `PYTORCH_API.md`.

Tutorial 7 completed its full training before an evaluator instrumentation bug
interfered with TorchScript trace checking. The identical trained checkpoint
failed with the original observer and passed without it. The corrected observer
pauses observation during the entire native trace/check operation; tracing
sanity checks remain enabled. The unchanged posttraining cells were continued
from the preserved checkpoint, with no further optimization. Native optimizer
states recorded 20,000 baseline and 200,000 pruning steps; the histories contain
110,000,000 trained rows. Reconstructed model state and full predictions matched
the saved stripped checkpoint bitwise. Floating-point TorchScript restoration
was checked at batch sizes 32, 17 and 2,000; three actual qint8 Linear operators,
int8 export/restoration, plots and metrics were also checked. This is a full
training run plus a documented posttraining continuation, rather than one
uninterrupted observer-instrumented notebook execution.

All sixteen original TensorFlow notebooks and the three demo data artifacts
remain unchanged. The archived TensorFlow validation also passed separately:
29 unittest checks, one skipped, in its dedicated legacy environment.

## Setup and remaining gates

The wheel has been checked in an isolated installation with native inference,
all bundled demo datasets, and no TensorFlow runtime dependency. The candidate
is published on the `pytorch` branch. In an additional fresh Linux Python 3.13
environment with no initial PyTorch or NIF and no local checkout, the unchanged
notebook setup cell completed its actual GitHub installation branch. The
installed distribution's `direct_url.json` recorded implementation commit
`eac45241e011f1f8a467738426136e2b6ba312de`. All seven declared pins matched:
PyTorch `2.14.1`, NumPy `2.1.3`, SciPy `1.16.3`, Matplotlib `3.10.0`, protobuf
`7.36.2`, google-crc32c `1.9.0`, and tfrecord `1.14.6`. CUDA was available,
native `nn.Module` inference and backpropagation were finite, and the three
packaged datasets contained 2,000, 2,000 and 191,775 rows. NIF neither depended
on nor imported TensorFlow.

Tutorial 4 also completed all ten code cells in order in a newly allocated
hosted Google Colab CPU runtime, using the published GitHub notebook and its
actual install cell. It reported PyTorch `2.14.1+cu130`, FULL mode, and maximum
Jacobian and Hessian errors of `5.960464477539063e-08`. Its checkpoint, metrics
file and Jacobian image were created, and the image was visually checked.
Colab displayed an output-JavaScript loading error, but execution completed;
the generated metrics and PNG were inspected through Colab's file browser.
Browser security settings were not changed. This hosted check covers tutorial
4, not full hosted execution of all eight notebooks.

The remaining full runs and their artifact/numerical/visual checks must finish
before full acceptance is claimed.
