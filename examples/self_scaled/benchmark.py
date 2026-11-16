"""Compare full-batch FP64 L-BFGS, SSBFGS and SSBroyden on the T1 wave."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import platform
import time
import traceback
from pathlib import Path


THREADS = 2
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = str(THREADS)

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import torch  # noqa: E402

import nif  # noqa: E402
from nif.demo import TravelingWave  # noqa: E402
from nif.optimizers import AdaBeliefOptimizer  # noqa: E402
from nif.optimizers import ReferenceLBFGS  # noqa: E402


METHODS = ("lbfgs", "ssbfgs", "ssbroyden")
SHAPE_CONFIG = dict(
    connectivity="full",
    input_dim=1,
    output_dim=1,
    units=30,
    nlayers=2,
    activation="swish",
)
PARAMETER_CONFIG = dict(
    input_dim=1, latent_dim=1, units=30, nlayers=2, activation="swish"
)


def support_module(name):
    spec = importlib.util.spec_from_file_location(
        "nif_ss_support_" + name, Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
    )


def array_hash(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def build_model():
    torch.manual_seed(42)
    return nif.NIF(
        copy.deepcopy(SHAPE_CONFIG), copy.deepcopy(PARAMETER_CONFIG), "float64"
    ).cpu()


def flat_parameters(model):
    return (
        torch.cat([p.detach().reshape(-1) for p in model.parameters()]).numpy().copy()
    )


def assign_parameters(model, position):
    vector = torch.as_tensor(np.asarray(position), dtype=torch.float64)
    if vector.ndim != 1 or vector.numel() != sum(p.numel() for p in model.parameters()):
        raise ValueError("Packed parameter shape does not match this model")
    if not torch.isfinite(vector).all():
        raise FloatingPointError("Non-finite parameter vector")
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            count = parameter.numel()
            parameter.copy_(vector[offset : offset + count].view_as(parameter))
            offset += count


def analytical(t, x):
    z = x - 0.2 - 0.006 * t
    return np.exp(-1000 * z**2) * np.sin(4 * z)


def dataset():
    """Keep the stored T1 arrays/statistics, including their source precision."""
    wave = TravelingWave()
    data = np.asarray(wave.data, dtype=np.float64)
    raw = np.asarray(wave.data_raw, dtype=np.float64)
    mean, std = np.asarray(wave.mean, dtype=np.float64), np.asarray(
        wave.std, dtype=np.float64
    )
    dense_x, dense_t = np.meshgrid(np.linspace(0, 0.995, 800), np.linspace(0, 90, 45))
    dense_raw = np.column_stack((dense_t.ravel(), dense_x.ravel()))
    return dict(
        train_features=data[:, :2],
        train_target=data[:, -1:],
        mean=mean,
        std=std,
        train_t=raw[:, 0].reshape(10, 200),
        train_x=raw[:, 1].reshape(10, 200),
        train_truth_normalized=data[:, -1],
        train_truth_physical=raw[:, -1],
        dense_features=(dense_raw - mean[:2]) / std[:2],
        dense_t=dense_t,
        dense_x=dense_x,
        dense_truth_physical=analytical(dense_t, dense_x).ravel(),
    )


def objective(model, features, targets):
    return (model(features) - targets).square().mean() + model.regularization_loss(
        features
    )


def physical_metrics(prediction, truth):
    difference = np.asarray(prediction) - np.asarray(truth)
    if difference.size == 0 or not np.isfinite(difference).all():
        raise FloatingPointError("Non-finite or empty physical prediction")
    return dict(
        mse=float(np.mean(difference**2)),
        rmse=float(np.sqrt(np.mean(difference**2))),
        mae=float(np.mean(np.abs(difference))),
        max_abs=float(np.max(np.abs(difference))),
    )


def save_checkpoint(path, model, **extra):
    torch.save(
        dict(
            model_state={
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            cfg_shape_net=SHAPE_CONFIG,
            cfg_parameter_net=PARAMETER_CONFIG,
            mixed_policy="float64",
            parameter_hash=array_hash(flat_parameters(model)),
            **extra,
        ),
        path,
    )


def learning_rate(epoch):
    if epoch < 1000:
        return 5e-3
    if epoch < 2000:
        return 1e-3
    if epoch < 4000:
        return 5e-4
    return 1e-4


def warmup(model, data, out, epochs):
    x = torch.as_tensor(data["train_features"], dtype=torch.float64)
    y = torch.as_tensor(data["train_target"], dtype=torch.float64)
    generator = torch.Generator().manual_seed(42)
    optimizer = AdaBeliefOptimizer(model.parameters(), lr=5e-3)
    history = []
    consumed = hashlib.sha256()
    start = time.perf_counter()
    for epoch in range(epochs):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(epoch)
        permutation = torch.randperm(len(x), generator=generator)
        consumed.update(permutation.numpy().tobytes())
        total = 0.0
        for indices in permutation.split(512):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model, x[indices], y[indices])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite shared AdaBelief warmup")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        history.append(
            dict(
                epoch=epoch + 1,
                loss=total / len(x),
                elapsed_seconds=time.perf_counter() - start,
            )
        )
        if epoch % 100 == 0 or epoch == epochs - 1:
            print("warmup", json.dumps(history[-1]), flush=True)
    elapsed = time.perf_counter() - start
    receipt = dict(
        warmup_epochs=epochs,
        warmup_elapsed_seconds=elapsed,
        batch_size=512,
        shuffle_seed=42,
        batch_order_sha256=consumed.hexdigest(),
        gradient_centralization=False,
    )
    save_checkpoint(
        out / "shared_start.pt",
        model,
        **receipt,
        optimizer_state=optimizer.state_dict(),
    )
    write_json(out / "warmup_history.json", history)
    return receipt


class ObjectiveRecorder:
    def __init__(self, model, data, started):
        self.model = model
        self.features = torch.as_tensor(data["train_features"], dtype=torch.float64)
        self.target = torch.as_tensor(data["train_target"], dtype=torch.float64)
        self.started = started
        self.calls = []
        self.first = None

    def value_and_gradient(self, position=None):
        if position is not None:
            assign_parameters(self.model, position)
        self.model.zero_grad(set_to_none=True)
        loss = objective(self.model, self.features, self.target)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite full-batch objective")
        loss.backward()
        gradient = (
            torch.cat([p.grad.detach().reshape(-1) for p in self.model.parameters()])
            .numpy()
            .copy()
        )
        if not np.isfinite(gradient).all():
            raise FloatingPointError("Non-finite full-batch gradient")
        value = float(loss.detach())
        if self.first is None:
            self.first = dict(
                parameter_hash=array_hash(flat_parameters(self.model)),
                loss=value,
                gradient_hash=array_hash(gradient),
                gradient_dtype=str(gradient.dtype),
                gradient_inf=float(np.max(np.abs(gradient))),
            )
        self.calls.append(
            dict(
                evaluation=len(self.calls) + 1,
                loss=value,
                elapsed_seconds=time.perf_counter() - self.started,
            )
        )
        return value, gradient

    def closure(self):
        value, _ = self.value_and_gradient()
        # ReferenceLBFGS consumes the value and already-computed parameter grads.
        return torch.tensor(value, dtype=torch.float64)


def author_status(result):
    finite = {
        "x": bool(np.isfinite(result.x).all()),
        "jac": bool(np.isfinite(result.jac).all()),
        "hess_inv": bool(np.isfinite(result.hess_inv).all()),
        "objective": bool(np.isfinite(result.fun)),
    }
    state_finite = all(finite.values())
    converged = bool(result.success) and state_finite
    termination = (
        "converged"
        if converged
        else {1: "iteration_limit", 2: "precision_loss"}.get(
            int(result.status), "backend_failure"
        )
    )
    if not state_finite:
        termination = "nonfinite_optimizer_state"
    return dict(
        backend_success=bool(result.success),
        backend_status=int(result.status),
        message=str(result.message),
        finite=finite,
        state_finite=state_finite,
        converged=converged,
        termination=termination,
        backend_nit=int(result.nit),
        backend_nfev=int(result.nfev),
        backend_njev=int(result.njev),
    )


def predict(model, features):
    with torch.no_grad():
        value = torch.cat(
            [
                model(part)
                for part in torch.as_tensor(features, dtype=torch.float64).split(1000)
            ]
        ).ravel()
    return value.numpy()


def optimize_arm(method, shared_state, data, out, max_iterations, author=None):
    model = build_model()
    model.load_state_dict(shared_state)
    initial = flat_parameters(model)
    start = time.perf_counter()
    recorder = ObjectiveRecorder(model, data, start)
    history = []
    count = 0
    status = dict(converged=False, termination="exception", state_finite=False)
    try:
        # Record the identical actual starting objective/gradient for every arm.
        initial_value, _ = recorder.value_and_gradient()
        history.append(
            dict(iteration=0, elapsed_seconds=0.0, loss=initial_value, accepted=True)
        )
        if method == "lbfgs":
            optimizer = ReferenceLBFGS(
                model.parameters(),
                max_iter=1,
                num_correction_pairs=20,
                tolerance=1e-15,
                x_tolerance=1e-15,
                f_relative_tolerance=1e-15,
                f_absolute_tolerance=0.0,
                max_line_search_iterations=100,
            )
            for _ in range(max_iterations):
                optimizer.step(recorder.closure)
                result = optimizer.result
                count = int(result.num_iterations)
                # A failed line search is not an accepted new iterate.
                if not bool(result.failed):
                    if not bool(torch.isfinite(result.objective_value)):
                        raise FloatingPointError("Non-finite accepted L-BFGS objective")
                    history.append(
                        dict(
                            iteration=count,
                            elapsed_seconds=time.perf_counter() - start,
                            loss=float(result.objective_value),
                            accepted=True,
                        )
                    )
                if count % 100 == 0:
                    print(method, "iteration", count, flush=True)
                if bool(result.failed) or bool(result.converged):
                    break
            state_finite = all(
                bool(torch.isfinite(value).all())
                for value in (
                    result.position,
                    result.objective_gradient,
                    result.objective_value,
                    result.position_deltas,
                    result.gradient_deltas,
                )
            )
            converged = (
                bool(result.converged) and not bool(result.failed) and state_finite
            )
            termination = "converged" if converged else "iteration_limit"
            if bool(result.failed):
                termination = "line_search_failure"
            if not state_finite:
                termination = "nonfinite_optimizer_state"
            status = dict(
                converged=converged,
                failed=bool(result.failed),
                state_finite=state_finite,
                termination=termination,
                backend_nit=count,
                backend_nfev=int(result.num_objective_evaluations),
            )
        else:
            memo = author.MemoizeJac(recorder.value_and_gradient)

            def callback(intermediate_result):
                nonlocal count
                if (
                    not np.isfinite(intermediate_result.fun)
                    or not np.isfinite(intermediate_result.x).all()
                ):
                    raise FloatingPointError("Non-finite self-scaled accepted state")
                count += 1
                history.append(
                    dict(
                        iteration=count,
                        elapsed_seconds=time.perf_counter() - start,
                        loss=float(intermediate_result.fun),
                        accepted=True,
                    )
                )
                if count % 100 == 0:
                    print(method, "iteration", count, flush=True)

            result = author._minimize_bfgs(
                memo,
                initial.copy(),
                jac=memo.derivative,
                callback=callback,
                gtol=1e-15,
                maxiter=max_iterations,
                xrtol=1e-15,
                c1=1e-4,
                c2=0.9,
                method_bfgs={"ssbfgs": "SSBFGS_AB", "ssbroyden": "SSBroyden2"}[method],
                initial_scale=False,
                disp=False,
            )
            status = author_status(result)
            count = int(result.nit)
            assign_parameters(model, result.x)
        elapsed = time.perf_counter() - start
    except Exception as error:
        elapsed = time.perf_counter() - start
        status = dict(
            converged=False,
            state_finite=False,
            termination="exception",
            exception_type=type(error).__name__,
            message=str(error),
        )
        (out / (method + "_exception.txt")).write_text(
            traceback.format_exc(), encoding="utf-8"
        )
    write_json(out / (method + "_history.json"), history)
    write_json(out / (method + "_evaluations.json"), recorder.calls)
    record = dict(
        method=method,
        elapsed_seconds=elapsed,
        iterations=count,
        evaluations=len(recorder.calls),
        first_evaluation=recorder.first,
        initial_parameter_hash=array_hash(initial),
        status=status,
        train=None,
        dense=None,
    )
    arrays = {}
    # Keep failed optimizer states in the report; do not disguise an exception's
    # last trial vector as that optimizer's returned final scientific result.
    if (
        status["termination"] != "exception"
        and np.isfinite(flat_parameters(model)).all()
    ):
        evaluation_start = time.perf_counter()
        for split in ("train", "dense"):
            normalized = predict(model, data[split + "_features"])
            physical = normalized * data["std"][-1] + data["mean"][-1]
            record[split] = physical_metrics(physical, data[split + "_truth_physical"])
            arrays[method + "_" + split + "_prediction_physical"] = physical
            if split == "train":
                arrays[method + "_train_prediction_normalized"] = normalized
        record["evaluation_seconds"] = time.perf_counter() - evaluation_start
        record["final_parameter_hash"] = array_hash(flat_parameters(model))
        save_checkpoint(
            out / (method + "_final.pt"), model, method=method, status=status
        )
    np.savez_compressed(
        out / (method + "_parameters.npz"),
        initial=initial,
        final=flat_parameters(model),
    )
    return record, arrays


def run_experiment(
    output_dir,
    warmup_epochs=5000,
    max_iterations=1000,
    methods=METHODS,
    render=True,
    smoke=False,
    source_cache=None,
):
    if warmup_epochs < 0 or max_iterations < 1:
        raise ValueError(
            "Use nonnegative warmup epochs and a positive iteration ceiling"
        )
    if (
        not methods
        or len(set(methods)) != len(methods)
        or not set(methods) <= set(METHODS)
    ):
        raise ValueError("Choose distinct supported optimizer methods")
    if smoke:
        warmup_epochs, max_iterations = 2, 3
    mode = (
        "SMOKE"
        if smoke
        else ("FULL" if (warmup_epochs, max_iterations) == (5000, 1000) else "CUSTOM")
    )
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            "Choose a new output directory to preserve earlier evidence"
        )
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    adapter = support_module("source")
    author = adapter.load_author(source_cache) if set(methods) - {"lbfgs"} else None
    data = dataset()
    np.savez_compressed(out / "data.npz", **data)
    model = build_model()
    initial_hash = array_hash(flat_parameters(model))
    # Load numerical kernels before measured warmup and optimization.
    features = torch.as_tensor(data["train_features"][:512], dtype=torch.float64)
    targets = torch.as_tensor(data["train_target"][:512], dtype=torch.float64)
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        objective(model, features, targets).backward()
    model.zero_grad(set_to_none=True)
    warmup_receipt = warmup(model, data, out, warmup_epochs)
    shared_state = copy.deepcopy(model.state_dict())
    shared_hash = array_hash(flat_parameters(model))
    count = sum(parameter.numel() for parameter in model.parameters())
    manifest = dict(
        experiment="Tutorial 9: experimental self-scaled optimizer comparison",
        mode=mode,
        case="exp(-1000*(x-.2-.006*t)^2)*sin(4*(x-.2-.006*t))",
        cfg_shape_net=SHAPE_CONFIG,
        cfg_parameter_net=PARAMETER_CONFIG,
        parameters=count,
        precision="float64",
        device="cpu",
        threads=THREADS,
        one_dense_hessian_bytes=count * count * 8,
        initial_parameter_hash=initial_hash,
        shared_start_parameter_hash=shared_hash,
        mean=data["mean"].tolist(),
        std=data["std"].tolist(),
        source_data="Original stored FP32 TravelingWave arrays/statistics cast uniformly to FP64",
        loss="mean((prediction-normalized_target)**2) + model.regularization_loss(fixed_features)",
        full_batch_rows=len(data["train_features"]),
        methods=list(methods),
        max_iterations=max_iterations,
        rounds=1,
        seed=42,
        **warmup_receipt,
        original_tutorial_budget=dict(
            precision="float32", rounds=200, max_iterations_per_round=1000
        ),
        train_grid=dict(shape=[10, 200], order="time-major; x varies fastest"),
        dense_grid=dict(
            shape=[45, 800], t=[0.0, 90.0], x=[0.0, 0.995], kind="extra interpolation"
        ),
        baseline=dict(
            implementation="ReferenceLBFGS",
            line_search="Hager-Zhang",
            correction_pairs=20,
        ),
        self_scaled=dict(
            implementation="pinned author dense CPU NumPy/SciPy",
            line_search="Strong Wolfe",
            c1=1e-4,
            c2=0.9,
            initial_scale=False,
        ),
        author=adapter.provenance(),
        versions=dict(
            python=platform.python_version(),
            torch=str(torch.__version__),
            numpy=np.__version__,
            scipy=scipy.__version__,
            nif=nif.__version__,
        ),
        source_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(__file__).with_name("source.py"),
                Path(__file__).with_name("report.py"),
            )
        },
        platform=platform.platform(),
        timing="Sequential CPU arms; QN includes objective/gradient, line search, update and logging; "
        "excludes shared warmup, prediction and plotting. Different line searches/stopping contracts.",
    )
    write_json(out / "manifest.json", manifest)
    summary = dict(
        mode=mode,
        warmup_elapsed_seconds=warmup_receipt["warmup_elapsed_seconds"],
        arms={},
    )
    arrays = {
        key: value
        for key, value in data.items()
        if "features" not in key and key != "train_target"
    }
    for method in methods:
        print("ARM_START", method, flush=True)
        record, predictions = optimize_arm(
            method, shared_state, data, out, max_iterations, author
        )
        summary["arms"][method] = record
        arrays.update(predictions)
        summary["shared_start_verified"] = all(
            row["first_evaluation"] is not None
            and row["initial_parameter_hash"] == shared_hash
            and row["first_evaluation"]
            == next(iter(summary["arms"].values()))["first_evaluation"]
            for row in summary["arms"].values()
        )
        summary["execution_valid"] = (
            set(summary["arms"]) == set(methods)
            and summary["shared_start_verified"]
            and all(
                row["status"]["state_finite"]
                and row["status"]["termination"]
                in {"converged", "iteration_limit", "precision_loss"}
                and row["train"] is not None
                and row["dense"] is not None
                for row in summary["arms"].values()
            )
        )
        np.savez_compressed(out / "predictions.npz", **arrays)
        write_json(out / "summary.json", summary)
        print("ARM_DONE", method, json.dumps(record), flush=True)
    if not summary["shared_start_verified"]:
        raise RuntimeError(
            "Actual initial weights/objective/gradient disagree across optimizer arms"
        )
    if render:
        support_module("report").render_report(out)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory; existing evidence is preserved",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Two warmup epochs and three QN steps; not scientific results",
    )
    parser.add_argument("--warmup-epochs", type=int, default=5000)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--source-cache", type=Path)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    summary = run_experiment(
        args.output,
        args.warmup_epochs,
        args.max_iterations,
        smoke=args.smoke,
        source_cache=args.source_cache,
        render=not args.no_render,
    )
    if not summary["execution_valid"]:
        raise RuntimeError(
            "Experiment execution invalid; inspect the saved per-arm statuses and receipts"
        )


if __name__ == "__main__":
    main()
