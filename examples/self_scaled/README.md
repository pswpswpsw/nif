# Experimental self-scaled optimizers

[Tutorial 9](../../tutorial/9_self_scaled_optimizers_pytorch.ipynb) compares
`ReferenceLBFGS`, the authors' `SSBFGS_AB` and `SSBroyden2` on the T1 traveling
wave. The [experiment guide](../../docs/SELF_SCALED_OPTIMIZERS.md) defines the
controlled comparison, plotting conventions, source provenance and results.
These adapters are experimental examples outside the stable optimizer API.

## Run locally

Use Python 3.11–3.13 and a separate environment. From the repository root:

```bash
python -m venv .venv-ss
# Activate the environment using the command appropriate to your shell.
python -m pip install -r examples/self_scaled/requirements.txt
python -m pip install --no-deps -e .
python examples/self_scaled/benchmark.py --output pytorch_outputs/self-scaled/full-001
```

The default FULL workload performs 5,000 shared warmup epochs and up to 1,000
quasi-Newton iterations per method. It uses CPU FP64 and dense self-scaled
matrices; allow roughly half an hour or more and several times the 261.4 MiB
needed for one inverse Hessian. CUDA is optional. To select a CPU-only wheel,
install `torch==2.14.1` from `https://download.pytorch.org/whl/cpu` first.

For an explicitly shortened execution check:

```bash
python examples/self_scaled/benchmark.py --smoke --output pytorch_outputs/self-scaled/smoke-001
```

SMOKE uses two warmup epochs and three QN iterations. `--warmup-epochs` and
`--max-iterations` select a CUSTOM workload. Every run needs a new output
directory; existing nonempty directories are rejected.

## Outputs

Each run saves `REPORT.md`, `manifest.json`, `summary.json`, shared/final
checkpoints, parameter vectors, predictions, evaluation/iteration histories
and figures. Read physical errors and raw termination states together;
`iteration_limit` and `precision_loss` remain explicit.

The Colab notebook creates a separate environment automatically. Both entry
points download the fixed, SHA256-verified author source on first use and
retain installed SciPy files. The [requirements](requirements.txt) pin its
compatible environment; the guide records the exact external source and
license provenance.

For future case extensions, reuse this runner's parameter packing, objective
recorder and verified source loader. Define each case's data, normalization,
objective and plotting semantics explicitly. Historical local experiments
remain frozen records of their original workloads.
