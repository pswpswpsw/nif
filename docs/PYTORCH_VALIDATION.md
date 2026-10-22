# Native PyTorch validation

Status: **full integration validation in progress**, 2026-10-22.
Implementation branch: `pytorch`; TensorFlow reference branch: `main`.

## Independent package acceptance

Agent B keeps its acceptance tests and immutable reference fixtures outside the
repository. Agent A works in a separate container: its repository is writable,
Git metadata is read-only, and coordinator files are masked. The evaluator
directory and user drives are not mounted.
Only sanitized failure descriptions are returned to the implementation agent.
Tests were frozen before implementation, and evaluator corrections retain
versioned provenance and coordinator review.

The corrected native implementation passed **247/247 checks on CUDA**, with no
skips or failures. Its CPU result was **244 passed, 3 CUDA-only checks skipped**.
Coverage includes all three architectures, generated-weight layouts against
TensorFlow numerical references, shared subnets, derivatives and higher-order
autograd, regularization reductions, optimizer formulas and restoration,
normalization, actual legacy TFRecord interoperability, and initializer options.

Optional initializer, epsilon-fallback and L4Adam/gradient-centralization corrections were checked
against the earlier full-run source. Eleven actual tutorial configurations had
bitwise-identical initialization, RNG state, forward values, input/parameter
gradients, regularization, and one optimizer update on the same CPU and CUDA
platforms. Those corrections do not change the ongoing tutorial trajectories.

## Full tutorial execution

These are fresh-kernel full runs, independently observed by Agent B. Shortened
development smoke runs are recorded separately and do not count as full passes.

| Tutorial | Full workload | Independent result |
|---|---|---|
| 1 | 5,000 AdaBelief epochs; 200 L-BFGS rounds, max_iter 1,000 | Passed; actual steps, checkpoints and figures checked |
| 2 | Two 5,000-epoch fits | Passed; 20,000 Adam steps per fit; figures checked |
| 3 | 600 epochs over all 191,775 weighted cylinder-flow samples | Running |
| 4 | Full selected Jacobian/Hessian examples and differentiation | Passed; figure checked |
| 5 | One million rows, 100 TFRecord shards, two complete passes | Passed; 2,000,000 actual trained rows, 200 shard visits |
| 6 | Nine 3,000-epoch fits and superresolution | Passed; 108,000 Adam steps, 54,000,000 trained rows; figures checked |
| 7 | 5,000 baseline and 50,000 pruning epochs; int8/export inference | Running |
| 8 | Two 50,000-epoch fits; Sobolev objective and physical superresolution | Running |

Tutorial 1 retains the original tolerance/history controls. Native PyTorch and
TFP line-search and stopping conventions differ; the notebook documents those
differences. Tutorial 7 uses a conventional Frobenius relative-MSE metric rather
than the archived matrix-spectral-norm ratio; see `PYTORCH_API.md`.

All sixteen original TensorFlow notebooks and the three demo data artifacts
remain unchanged. The archived TensorFlow validation also passed separately:
29 unittest checks, one skipped, in its dedicated legacy environment.

## Setup and remaining gates

The wheel has been checked in an isolated installation with native inference,
all bundled demo datasets, and no TensorFlow runtime dependency. Exact clean
GitHub installation and hosted Colab startup are pending publication of the
candidate branch. The remaining full runs and their artifact/numerical/visual
checks must finish before full acceptance is claimed.
