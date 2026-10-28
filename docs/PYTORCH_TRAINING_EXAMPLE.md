# Distributed training and TensorBoard example

[`examples/train_pytorch.py`](../examples/train_pytorch.py) trains the packaged
traveling-wave dataset using the existing NIF model and normalization. It adds a
standalone PyTorch DDP and TensorBoard workflow alongside the eight tutorials.

## Run from a checkout

Install NIF into the Python environment used to launch the example:

```bash
python -m pip install -e '.[tensorboard]'
python examples/train_pytorch.py --device cpu --epochs 20 --tensorboard
tensorboard --logdir pytorch_outputs/traveling_wave
```

For two CUDA GPUs on Linux or WSL with an NCCL-enabled PyTorch build:

```bash
torchrun --standalone --nproc-per-node=2 examples/train_pytorch.py --device cuda --epochs 20 --tensorboard
```

A Linux CPU/Gloo check uses the same launcher with `--device cpu`:

```bash
torchrun --standalone --nproc-per-node=2 examples/train_pytorch.py --device cpu --epochs 2 --max-samples 128 --batch-size 32 --output-dir pytorch_outputs/ddp-smoke --tensorboard
```

Native Windows builds may lack the libuv backend required by the `torchrun`
launcher. Use Linux/WSL for these launcher examples. The single-process CPU
example also works on Windows.

## Training and output semantics

- `--batch-size` specifies rows per process. DDP averages gradients over the
  processes, so the global batch size is this value times the world size.
- Each process receives the same number of rows. The example rejects a dataset
  size that is not divisible by the world size, preserving all rows without
  padding or dropping. The final batch may be smaller, with equal sizes across
  processes. `DistributedSampler.set_epoch()` updates the shuffle each epoch.
- The objective is normalized field MSE plus `model.regularization_loss(inputs)`.
  The complete objective runs inside the DDP forward pass. `--l2-reg` controls
  ParameterNet L2 regularization; its default is zero.
- Training uses `torch.optim.Adam`. The learning rate defaults to `1e-3` and
  decays by a factor of `0.1` after every ten epochs. The CLI exposes these
  settings. Tutorial L-BFGS remains a separate single-device workflow.
- Epoch metrics are sample-weighted averages of the losses encountered during
  training, reduced across processes. `final_mse` evaluates the final model.
  Both use normalized units.
- Rank zero writes `config.json`, `history.json`, and `checkpoint.pt`. When
  enabled, TensorBoard events go into `tensorboard/` and contain `train/mse`,
  `train/objective`, `train/lr`, and `eval/mse`.
- The checkpoint contains model, optimizer and scheduler state, the completed
  epoch, and the original normalization. The script trains a new model on each
  invocation; it does not offer a resume-training flag.

Restore a saved model for inference:

```python
import json
from pathlib import Path

import torch
from nif import NIF

run = Path("pytorch_outputs/traveling_wave")
config = json.loads((run / "config.json").read_text())
checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
model = NIF(**config)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

The implementation follows PyTorch's
[DDP interface](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
and [SummaryWriter interface](https://docs.pytorch.org/docs/stable/tensorboard).

## Verification scope

The focused public checks in
[`dev_tests/test_pytorch_training_example.py`](../dev_tests/test_pytorch_training_example.py)
exercise real training processes, checkpoint reconstruction, TensorBoard event
readback, equal-global-batch comparisons and rejection of incompatible dataset
sizes. Independent acceptance checks are maintained separately.

On 2026-10-28, the three focused public execution tests and two README regression
checks passed. Independent acceptance also checked two different global batch
configurations against corresponding single-process runs, including partial
final batches and nonzero regularization. Model and optimizer state, sample
counts, losses, learning-rate schedules and logged scalars agreed within the
comparison tolerances. Non-divisible dataset sizes were rejected on both ranks.

The README's seven Python examples were executed in sequence with shorter
training budgets: two first-order epochs and three L-BFGS iterations. The checks
included actual record-shard generation/readback and model checkpoint loading.
Both original GIFs were unchanged, and all eight native tutorial links resolve
to existing notebooks on `main`.

Validation used PyTorch 2.14.1 on Windows CPU/Gloo and WSL CUDA. The documented
`torchrun` launcher completed a two-process CPU/Gloo run under WSL, using two
epochs and 128 rows. The local CUDA device is a single RTX 3080 Ti Laptop GPU.
A two-epoch, 128-row single-GPU run completed and its saved checkpoint and
TensorBoard metrics were read back successfully. These short runs validate
execution and artifact consistency, not scientific convergence or the full
tutorial workloads.

Actual two-GPU CUDA/NCCL training has not been measured on this one-GPU machine.
