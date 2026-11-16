"""Plot the experimental comparison from its newly saved arrays."""
import json
from pathlib import Path

import numpy as np


LABELS = {"lbfgs": "L-BFGS", "ssbfgs": "SSBFGS", "ssbroyden": "SSBroyden"}


def coordinate_grid(t, x, shape):
    """Verify paired coordinates point by point before time-major reshaping."""
    nt, nx = shape
    t, x = np.asarray(t).ravel(), np.asarray(x).ravel()
    if t.size != nt * nx or x.size != nt * nx:
        raise ValueError("Coordinate count differs from the declared grid")
    times, positions = np.unique(t), np.unique(x)
    if len(times) != nt or len(positions) != nx:
        raise ValueError("Coordinate axes differ from the declared grid")
    xx, tt = np.meshgrid(positions, times)
    if not np.isfinite(t).all() or not np.isfinite(x).all():
        raise ValueError("Plot coordinates must be finite")
    if not np.array_equal(t, tt.ravel()) or not np.array_equal(x, xx.ravel()):
        raise ValueError("Expected time-major coordinates with x varying fastest")
    return tt, xx


def metrics_markdown(summary):
    rows = [
        f"### {summary['mode']} run: physical absolute errors",
        "",
        "| Method | Grid | RMSE | MAE | Max absolute error | QN seconds | "
        "Iterations | Evaluations | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for method, arm in summary["arms"].items():
        for split in ("train", "dense"):
            metric = arm[split]
            values = (
                f"{metric['rmse']:.8g} | {metric['mae']:.8g} | {metric['max_abs']:.8g}"
                if metric is not None
                else "unavailable | unavailable | unavailable"
            )
            rows.append(
                f"| {LABELS[method]} | {split} | {values} | {arm['elapsed_seconds']:.3f} | "
                f"{arm['iterations']} | {arm['evaluations']} | {arm['status']['termination']} |"
            )
    return "\n".join(rows)


def render_report(output_dir):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    root = Path(output_dir)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    saved = {}

    def save(figure, name):
        path = figures / (name + ".png")
        figure.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        saved[name] = str(path.relative_to(root))

    available = [
        method for method, arm in summary["arms"].items() if arm["train"] is not None
    ]
    with np.load(root / "predictions.npz", allow_pickle=False) as arrays:
        tt, xx = coordinate_grid(arrays["train_t"], arrays["train_x"], (10, 200))
        truth = arrays["train_truth_normalized"].reshape(10, 200)
        for method in available:
            prediction = arrays[method + "_train_prediction_normalized"].reshape(
                10, 200
            )
            fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
            for axis, value, title in zip(
                axes, (truth, prediction, prediction - truth), ("true", "pred", "error")
            ):
                image = axis.contourf(
                    tt, xx, value, levels=50, vmin=-5, vmax=5, cmap="seismic"
                )
                fig.colorbar(image, ax=axis)
                axis.set(title=title, xlabel="t")
            axes[0].set_ylabel("x")
            fig.suptitle(
                f"{summary['mode']}: {LABELS[method]} — original T1 rendering; "
                "normalized u; saved-array redraw"
            )
            fig.tight_layout()
            save(fig, method + "_original_style")

        for split, shape in (("train", (10, 200)), ("dense", (45, 800))):
            if not available:
                break
            plot_t, plot_x = coordinate_grid(
                arrays[split + "_t"], arrays[split + "_x"], shape
            )
            target = arrays[split + "_truth_physical"].reshape(shape)
            predictions = [
                arrays[method + "_" + split + "_prediction_physical"].reshape(shape)
                for method in available
            ]
            maximum = max(
                float(np.max(np.abs(value))) for value in [target, *predictions]
            )
            maximum = max(maximum, np.finfo(float).eps)
            fig, axes = plt.subplots(
                1,
                len(available) + 1,
                figsize=(4 * (len(available) + 1), 3.8),
                squeeze=False,
                sharex=True,
                sharey=True,
                constrained_layout=True,
            )
            for axis, value, title in zip(
                axes[0], [target, *predictions], ["truth", *map(LABELS.get, available)]
            ):
                image = axis.contourf(
                    plot_t,
                    plot_x,
                    value,
                    levels=np.linspace(-maximum, maximum, 51),
                    vmin=-maximum,
                    vmax=maximum,
                    cmap="seismic",
                )
                axis.set(title=title, xlabel="t")
            axes[0, 0].set_ylabel("x")
            fig.colorbar(
                image, ax=list(axes[0]), label="physical u (original field units)"
            )
            fig.suptitle(
                f"{summary['mode']}: {split} grid — common physical color scale; saved-array redraw"
            )
            save(fig, split + "_physical_comparison")

            errors = [np.abs(prediction - target) for prediction in predictions]
            maximum = max(
                max(float(np.max(value)) for value in errors), np.finfo(float).eps
            )
            fig, axes = plt.subplots(
                1,
                len(available),
                figsize=(5 * len(available), 3.8),
                squeeze=False,
                sharex=True,
                sharey=True,
                constrained_layout=True,
            )
            for axis, error, method in zip(axes[0], errors, available):
                image = axis.contourf(
                    plot_t,
                    plot_x,
                    error,
                    levels=np.linspace(0, maximum, 51),
                    vmin=0,
                    vmax=maximum,
                    cmap="magma",
                )
                axis.set(title=LABELS[method], xlabel="t")
            axes[0, 0].set_ylabel("x")
            fig.colorbar(
                image, ax=list(axes[0]), label="|prediction − truth| (physical u units)"
            )
            description = (
                "original 10 × 200 training points"
                if split == "train"
                else "extra 45 × 800 interpolation points"
            )
            fig.suptitle(
                f"{summary['mode']}: {description} — physical absolute error; saved-array redraw"
            )
            save(fig, split + "_absolute_errors")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for method in summary["arms"]:
        history = json.loads(
            (root / (method + "_history.json")).read_text(encoding="utf-8")
        )
        accepted = [row for row in history if row["accepted"]]
        if accepted:
            axes[0].semilogy(
                [row["iteration"] for row in accepted],
                [row["loss"] for row in accepted],
                label=LABELS[method],
            )
            axes[1].semilogy(
                [row["elapsed_seconds"] for row in accepted],
                [row["loss"] for row in accepted],
                label=LABELS[method],
            )
    for axis, xlabel in zip(axes, ("Accepted QN iteration", "QN wall time (s)")):
        axis.set(
            xlabel=xlabel, ylabel="Full fixed training objective (normalized units)"
        )
        axis.grid(alpha=0.2)
        if axis.lines:
            axis.legend()
    fig.suptitle(
        f"{summary['mode']}: identical actual starting tensors — shared warmup excluded"
    )
    save(fig, "convergence")

    report = "\n".join(
        [
            "# Experimental self-scaled optimizer comparison",
            "",
            metrics_markdown(summary),
            "",
            "`train` compares to stored physical u on the original 10×200 points. `dense` compares "
            "to the analytical field on 45×800 additional interpolation points, t∈[0,90],x∈[0,.995]. "
            "These are different evaluation sets. The source NPZ is FP32; "
            "casting does not recover source precision.",
            "",
            f"Shared AdaBelief warmup: {summary['warmup_elapsed_seconds']:.3f} seconds. "
            "Quasi-Newton times include line searches and objective/gradient evaluations; "
            "prediction and plotting are separate. Different line searches/stopping rules "
            "compare complete implementations, and this single "
            "CPU run does not establish a general speed ranking.",
            "",
            "An iteration limit or precision loss is not success. Raw backend status, state finiteness and "
            "convergence are retained in summary.json. Figures use actual saved predictions; "
            "failed arms with "
            "no returned finite field are omitted from field plots and remain in the table.",
            "",
            "The original T1 budget was FP32 with 200×1000 L-BFGS iterations. This experimental extension "
            "uses FP64 and one round capped at 1000 iterations per method. SMOKE and CUSTOM runs do not "
            "replace a full scientific comparison.",
            "",
            "## Configuration and provenance",
            "",
            "```json",
            json.dumps(manifest, indent=2),
            "```",
            "",
            "## Figures",
            "",
            *[f"![{name}]({path})\n" for name, path in saved.items()],
        ]
    )
    (root / "REPORT.md").write_text(report, encoding="utf-8")
    write_path = root / "report_artifacts.json"
    write_path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    return saved
