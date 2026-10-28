"""Train the packaged traveling wave on one device or with torchrun DDP.

Single device, with optional TensorBoard logging::

    python examples/train_pytorch.py --device cpu --tensorboard

Two GPUs on a CUDA/NCCL-enabled Linux or WSL installation::

    torchrun --standalone --nproc-per-node=2 examples/train_pytorch.py --device cuda --tensorboard

``--batch-size`` is per process. The dataset size must be divisible by the
number of processes: this example neither pads nor drops training rows.
The existing TravelingWave normalization and [parameter, coordinate] order
are used unchanged. This is an Adam training example; tutorial L-BFGS runs
remain single-device workflows.
"""
import argparse
import json
import math
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.utils.data import TensorDataset

from nif import NIF
from nif.demo import TravelingWave


class TrainingObjective(nn.Module):
    """Keep the complete differentiable objective inside DDP's forward pass."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features, target):
        mse = (self.model(features) - target).square().mean()
        return mse, mse + self.model.regularization_loss(features)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Rows per process per update"
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-decay-epoch", type=int, default=10)
    parser.add_argument("--lr-decay-factor", type=float, default=0.1)
    parser.add_argument(
        "--l2-reg", type=float, default=0.0, help="Parameter network L2 coefficient"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional prefix for a short check",
    )
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument(
        "--num-threads", type=int, default=1, help="CPU threads per process"
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Write native SummaryWriter event files",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("pytorch_outputs/traveling_wave")
    )
    args = parser.parse_args(argv)
    for name in ("epochs", "batch_size", "num_threads", "lr_decay_epoch"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    for name in ("lr", "lr_decay_factor"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.l2_reg) or args.l2_reg < 0:
        parser.error("--l2-reg must be finite and nonnegative")
    return args


def process_device(requested, world_size, local_rank):
    use_cuda = requested == "cuda" or (
        requested == "auto" and torch.cuda.is_available()
    )
    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Install a CUDA-enabled PyTorch build or use --device cpu."
            )
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                "Each local process requires a visible GPU; reduce --nproc-per-node."
            )
        if world_size > 1 and not dist.is_nccl_available():
            raise RuntimeError(
                "CUDA DDP requires NCCL. Use a CUDA-enabled Linux/WSL PyTorch build."
            )
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank), "nccl" if world_size > 1 else None
    if world_size > 1 and not dist.is_gloo_available():
        raise RuntimeError("CPU DDP requires a PyTorch build with Gloo support.")
    return torch.device("cpu"), "gloo" if world_size > 1 else None


def train(args):
    """Execute the example and return rank-zero metrics (``None`` on other ranks)."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("Invalid torchrun rank environment")
    device, backend = process_device(args.device, world_size, local_rank)
    writer_class = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise RuntimeError(
                'TensorBoard is optional; install it with python -m pip install -e ".[tensorboard]".'
            ) from error
        writer_class = SummaryWriter

    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    source = TravelingWave()
    rows = torch.as_tensor(source.data, dtype=torch.float32)
    if args.max_samples is not None:
        if args.max_samples > len(rows):
            raise ValueError(f"--max-samples exceeds the {len(rows)} available rows")
        rows = rows[: args.max_samples]
    if len(rows) % world_size:
        raise ValueError(
            "Dataset size must be divisible by world size; no rows are padded or dropped."
        )
    dataset = TensorDataset(rows[:, :2], rows[:, 2:3])
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        drop_last=False,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0
    )
    shape = dict(input_dim=1, output_dim=1, units=20, nlayers=2, activation="tanh")
    parameter = dict(
        input_dim=1,
        latent_dim=2,
        units=10,
        nlayers=2,
        activation="tanh",
        l2_reg=args.l2_reg,
    )
    model = NIF(shape, parameter).to(device)
    objective = TrainingObjective(model)
    writer = None
    owns_group = False
    try:
        if world_size > 1:
            dist.init_process_group(backend=backend, timeout=timedelta(seconds=120))
            owns_group = True
            objective = DistributedDataParallel(
                objective,
                device_ids=[local_rank] if device.type == "cuda" else None,
            )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_decay_epoch,
            gamma=args.lr_decay_factor,
        )
        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            model.save_config(args.output_dir / "config.json")
            if writer_class is not None:
                writer = writer_class(log_dir=str(args.output_dir / "tensorboard"))
        history = []
        model.train()
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)
            learning_rate = optimizer.param_groups[0]["lr"]
            # Equal rank sizes give equal local batch sizes, including the last
            # batch. DDP's gradient mean therefore equals the global batch mean.
            totals = torch.zeros(3, dtype=torch.float64, device=device)
            for features, target in loader:
                features, target = features.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                mse, loss = objective(features, target)
                loss.backward()
                optimizer.step()
                count = len(features)
                totals[0] += mse.detach().double() * count
                totals[1] += loss.detach().double() * count
                totals[2] += count
            if owns_group:
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            mse_sum, loss_sum, count = totals.tolist()
            record = dict(
                epoch=epoch + 1,
                mse=mse_sum / count,
                objective=loss_sum / count,
                lr=learning_rate,
                samples=int(count),
            )
            history.append(record)
            if rank == 0:
                print(
                    "epoch={} mse={:.8g} objective={:.8g} lr={:.4g}".format(
                        epoch + 1, record["mse"], record["objective"], learning_rate
                    )
                )
                if writer is not None:
                    for tag, key in (
                        ("train/mse", "mse"),
                        ("train/objective", "objective"),
                        ("train/lr", "lr"),
                    ):
                        writer.add_scalar(tag, record[key], epoch + 1)
            scheduler.step()

        model.eval()
        evaluation = torch.zeros(2, dtype=torch.float64, device=device)
        with torch.no_grad():
            for features, target in loader:
                features, target = features.to(device), target.to(device)
                evaluation[0] += (model(features) - target).square().double().sum()
                evaluation[1] += target.numel()
        if owns_group:
            dist.all_reduce(evaluation, op=dist.ReduceOp.SUM)
        final_mse = (evaluation[0] / evaluation[1]).item()
        if rank != 0:
            return None
        result = dict(
            world_size=world_size,
            device=device.type,
            backend=backend,
            samples=len(dataset),
            batch_size_per_rank=args.batch_size,
            global_batch_size=args.batch_size * world_size,
            seed=args.seed,
            epochs=history,
            final_mse=final_mse,
        )
        normalization = dict(
            mean=torch.as_tensor(source.mean), std=torch.as_tensor(source.std)
        )
        torch.save(
            dict(
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                epoch=args.epochs,
                normalization=normalization,
            ),
            args.output_dir / "checkpoint.pt",
        )
        (args.output_dir / "history.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        if writer is not None:
            writer.add_scalar("eval/mse", final_mse, args.epochs)
        print(
            "Saved {}; final normalized MSE={:.8g}".format(args.output_dir, final_mse)
        )
        return result
    finally:
        if writer is not None:
            writer.close()
        if owns_group:
            dist.destroy_process_group()


if __name__ == "__main__":
    train(parse_args())
