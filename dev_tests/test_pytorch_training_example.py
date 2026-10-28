"""Executable checks for the public DDP and TensorBoard training example."""
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from nif import NIF
from nif.demo import TravelingWave


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "train_pytorch.py"


def run_example(output, *, distributed=False, samples=64, tensorboard=False):
    assert SCRIPT.is_file(), "The documented native training example is missing"
    command = [sys.executable]
    if distributed:
        with socket.socket() as port_socket:
            port_socket.bind(("127.0.0.1", 0))
            port = port_socket.getsockname()[1]
        if os.name != "nt":
            command += [
                "-m",
                "torch.distributed.run",
                "--nnodes=1",
                "--nproc-per-node=2",
                "--rdzv-backend=static",
                "--master-addr=127.0.0.1",
                f"--master-port={port}",
            ]
    command += [
        str(SCRIPT),
        "--device",
        "cpu",
        "--epochs",
        "2",
        "--max-samples",
        str(samples),
        "--batch-size",
        "8" if distributed else "16",
        "--no-shuffle",
        "--output-dir",
        str(output),
        "--seed",
        "51",
    ]
    if tensorboard:
        command.append("--tensorboard")
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", USE_LIBUV="0")
    # Child processes use the checkout even if the installed package is elsewhere.
    environment["PYTHONPATH"] = (
        str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    if distributed and os.name == "nt":
        # The Windows torchrun agent creates a libuv TCPStore even in builds
        # without libuv. Exercise the same rank environment directly instead;
        # the script's env:// process group honors USE_LIBUV=0.
        processes = []
        try:
            for rank in range(2):
                rank_environment = dict(
                    environment,
                    WORLD_SIZE="2",
                    RANK=str(rank),
                    LOCAL_RANK=str(rank),
                    MASTER_ADDR="127.0.0.1",
                    MASTER_PORT=str(port),
                )
                processes.append(
                    subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=rank_environment,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                )
            outputs = [process.communicate(timeout=120) for process in processes]
            return subprocess.CompletedProcess(
                command,
                max(process.returncode for process in processes),
                "".join(out for out, _ in outputs),
                "".join(err for _, err in outputs),
            )
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.wait()
    return subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=120,
    )


def load_run(output):
    history = json.loads((output / "history.json").read_text(encoding="utf-8"))
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(
        output / "checkpoint.pt", map_location="cpu", weights_only=True
    )
    model = NIF(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    data = torch.as_tensor(
        TravelingWave().data[: history["samples"]], dtype=torch.float32
    )
    with torch.no_grad():
        mse = (model(data[:, :2]) - data[:, 2:3]).square().mean().item()
    assert mse == pytest.approx(history["final_mse"], rel=2e-6, abs=2e-7)
    assert checkpoint["epoch"] == 2
    assert checkpoint["optimizer_state_dict"]["state"]
    assert torch.isfinite(
        torch.tensor([entry["objective"] for entry in history["epochs"]])
    ).all()
    assert all(entry["samples"] == history["samples"] for entry in history["epochs"])
    return history, checkpoint


def test_single_device_checkpoint_and_tensorboard(tmp_path):
    pytest.importorskip("tensorboard")
    output = tmp_path / "single"
    result = run_example(output, tensorboard=True)
    assert result.returncode == 0, result.stdout + result.stderr
    history, checkpoint = load_run(output)
    assert history["world_size"] == 1
    assert history["samples"] == 64
    assert checkpoint["normalization"]["mean"].numel() == 3
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    files = list((output / "tensorboard").glob("events.out.tfevents.*"))
    assert len(files) == 1
    events = EventAccumulator(str(output / "tensorboard")).Reload()
    for tag, key in (
        ("train/mse", "mse"),
        ("train/objective", "objective"),
        ("train/lr", "lr"),
    ):
        scalars = events.Scalars(tag)
        assert [event.step for event in scalars] == [1, 2]
        assert [event.value for event in scalars] == pytest.approx(
            [entry[key] for entry in history["epochs"]],
            rel=2e-6,
        )


@pytest.mark.skipif(
    not torch.distributed.is_gloo_available(), reason="Gloo is unavailable"
)
def test_two_rank_ddp_matches_global_batch_and_logs_once(tmp_path):
    pytest.importorskip("tensorboard")
    single, distributed = tmp_path / "single", tmp_path / "distributed"
    first = run_example(single)
    assert first.returncode == 0, first.stdout + first.stderr
    second = run_example(distributed, distributed=True, tensorboard=True)
    assert second.returncode == 0, second.stdout + second.stderr
    single_history, single_checkpoint = load_run(single)
    ddp_history, ddp_checkpoint = load_run(distributed)
    assert ddp_history["world_size"] == 2
    assert ddp_history["backend"] == "gloo"
    assert ddp_history["global_batch_size"] == single_history["global_batch_size"] == 16
    assert len(list((distributed / "tensorboard").glob("events.out.tfevents.*"))) == 1
    for name, value in single_checkpoint["model_state_dict"].items():
        torch.testing.assert_close(
            ddp_checkpoint["model_state_dict"][name], value, rtol=1e-4, atol=2e-6
        )
    assert [entry["mse"] for entry in ddp_history["epochs"]] == pytest.approx(
        [entry["mse"] for entry in single_history["epochs"]],
        rel=1e-5,
        abs=2e-6,
    )


@pytest.mark.skipif(
    not torch.distributed.is_gloo_available(), reason="Gloo is unavailable"
)
def test_ddp_rejects_sampler_padding(tmp_path):
    result = run_example(tmp_path / "invalid", distributed=True, samples=63)
    assert result.returncode != 0
    assert "divisible by world size" in result.stdout + result.stderr
    assert not (tmp_path / "invalid" / "checkpoint.pt").exists()
