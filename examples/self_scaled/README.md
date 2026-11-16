# Experimental self-scaled optimizers

[Tutorial 9](../../tutorial/9_self_scaled_optimizers_pytorch.ipynb) compares
`nif.optimizers.ReferenceLBFGS` with the paper authors' `SSBFGS_AB` and
`SSBroyden2` on the first tutorial's traveling wave. These adapters are
experimental examples, separate from the supported `nif.optimizers` API.

## Run

Use Python 3.11–3.13 and a separate environment. From the repository root:

```bash
python -m venv .venv-ss
# Activate the environment using the command appropriate to your shell.
python -m pip install -r examples/self_scaled/requirements.txt
python -m pip install --no-deps -e .
python examples/self_scaled/benchmark.py --output pytorch_outputs/self-scaled/full-001
```

For a CPU-only PyTorch wheel, install `torch==2.14.1` from
`https://download.pytorch.org/whl/cpu` before installing the requirements.
CUDA is not needed: all three comparison arms use CPU FP64. The Colab notebook
creates its own environment and runs the benchmark in a subprocess. An already
imported NumPy in the Colab kernel is left intact.

An explicit short development run is available:

```bash
python examples/self_scaled/benchmark.py --smoke --output pytorch_outputs/self-scaled/smoke-001
```

SMOKE keeps the original 5,853-parameter model and all 2,000 training rows, but
runs two AdaBelief epochs and at most three quasi-Newton iterations per arm.
It checks execution, not convergence. `--warmup-epochs` and `--max-iterations`
can select a separately labeled CUSTOM workload. Existing nonempty output
directories are rejected, so previous results are preserved.

## Controlled comparison

- Original case: `z=x-.2-.006*t`, `u=exp(-1000*z**2)*sin(4*z)`.
- Original stored `TravelingWave` data, normalized arrays and statistics are
  converted uniformly from FP32 to FP64; casting does not recover source
  precision. All models, inputs, gradients and quasi-Newton states use FP64.
- A fresh shared 5,000-epoch AdaBelief warmup uses batches of 512, one recorded
  batch sequence, the original learning-rate schedule and no gradient
  centralization. Every comparison arm loads the identical actual checkpoint.
- The full-data objective is
  `mean((prediction-normalized_target)**2) + model.regularization_loss(features)`.
  It remains fixed across all line-search evaluations. These are full-batch
  quasi-Newton experiments, not a stochastic mini-batch implementation.
- Each arm gets one round capped at 1,000 iterations, with fresh curvature.
  Original T1 instead used FP32 and 200 L-BFGS rounds. This is an experimental
  extension, not a replacement for that original workload.
- Reference L-BFGS retains 20 correction pairs and Hager–Zhang line search.
  SSBFGS/SSBroyden use a dense inverse Hessian and Strong Wolfe
  (`c1=1e-4,c2=.9,initial_scale=False`). Their stopping contracts differ.
- CPU arms run sequentially with two numerical threads. A single dense
  5,853×5,853 FP64 matrix occupies 261.4 MiB; temporary matrices can require
  several times that memory. The full experiment can take roughly half an
  hour or more on a laptop; Colab CPU timings vary substantially.

## Outputs and interpretation

Each new run writes `manifest.json`, `summary.json`, `shared_start.pt`,
per-arm checkpoints/parameter vectors, all objective-evaluation logs,
accepted-iteration histories, `predictions.npz`, `REPORT.md`, and PNG figures.
The manifest records versions, source hashes, actual parameter/gradient
fingerprints, normalization, workload and author-source provenance.

`summary.json` retains raw backend status and convergence. Iteration limits,
precision loss, exceptions and non-finite inverse-Hessian states are distinct
from success. Finite predictions can still be inspected when the backend
returns a finite model with an invalid curvature state; that status stays
visible. An exception has no returned final model and is not assigned a
fabricated final-field error.

Errors are physical RMSE, MAE and maximum absolute error. Training-point
metrics use the original stored field on 10×200 points. The additional
45×800 grid uses the analytical field for interpolation within
`t∈[0,90],x∈[0,.995]`; it is a different evaluation set. Quasi-Newton times
include full objective/gradient evaluations, line search, curvature work and
logging. Shared warmup, prediction and plotting times are separate. These
single CPU timings compare complete implementations and do not establish
universal optimizer speed rankings.

Original-style plots retain `contourf`, 50 levels, `seismic`, fixed normalized
limits `[-5,5]`, time horizontal, position vertical, and residual
`prediction-truth`. The time-major/x-fast coordinate pairing is checked point
by point before reshaping. Extra physical-field and absolute-error plots use
shared quantitative scales and are labeled as saved-array redraws.

## Author source and citation

The algorithms come from
[Optimizing_the_Optimizer_PINNs](https://github.com/EliKiani/Optimizing_the_Optimizer_PINNs)
at commit `990a82c1353cbda9f5afee3200b222fdb821c521`, accompanying
[arXiv:2501.16371](https://arxiv.org/abs/2501.16371).
The authors' modified source is downloaded at runtime rather than redistributed
here. `source.py` verifies the exact 160,175 raw bytes against SHA256
`65958e6bba33631ca5aeab9ecc309536199c08a7e380fc0487d404e93ae2687e`
before compiling those same bytes in a separate module. The cache is
versioned, and installed SciPy files are never replaced. Internet access to
the pinned GitHub source is needed for the first run. The adapter explicitly
requires SciPy 1.16.3 because the author code uses SciPy private interfaces.

The original eight tutorials, source data and NIF optimizer implementations
are preserved. Historical local results and this newly runnable tutorial have
separate provenance and must not be presented as the same run.
