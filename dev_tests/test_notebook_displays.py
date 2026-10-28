"""Execute public plotting cells and check their rendered artists and data.

No training or private fixtures: use bundled wave data, an untrained native
model, and deterministic nearly constant predictions for the contour diagnostic.
"""
import ast
import copy
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import matplotlib
import nbformat
import numpy as np
import pytest
import torch
from matplotlib.contour import QuadContourSet
from matplotlib.transforms import Bbox
from PIL import Image

import nif
from nif.demo import TravelingWave
from nif.demo import TravelingWaveHighFreq

# Select the noninteractive backend before importing pyplot.
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(None)
def plotting_sources(source, number):
    if source == "notebook":
        path = next((ROOT / "tutorial").glob(f"{number}_*_pytorch.ipynb"))
        return [
            cell.source
            for cell in nbformat.read(path, as_version=4).cells
            if cell.cell_type == "code"
        ]
    path = ROOT / "dev_tests/generate_native_notebooks.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    # Run the real generator expressions but collect emit arguments without
    # writing any notebooks or changing archived reference cells.
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.FunctionDef) and node.name == "emit")
    ]
    captured = {}
    scope = {
        "__file__": str(path),
        "emit": lambda n, replacements, **kwargs: captured.update({n: replacements}),
    }
    exec(compile(tree, str(path), "exec"), scope)
    return list(captured[number].values())


def selected_cell(source, number, marker):
    return next(code for code in plotting_sources(source, number) if marker in code)


@pytest.fixture
def figures(monkeypatch):
    plt.close("all")
    close = plt.close
    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.setattr(plt, "close", lambda *args, **kwargs: None)
    with plt.rc_context(
        {"figure.dpi": 100, "savefig.dpi": 100, "savefig.pad_inches": 0.1}
    ):
        yield
    close("all")


def tight_render(fig):
    """Use the ordinary inline tight crop, including its standard padding."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    crop = fig.get_tightbbox(renderer).padded(plt.rcParams["savefig.pad_inches"])
    pixels = crop.transformed(fig.dpi_scale_trans)
    output = BytesIO()
    fig.savefig(output, format="png", bbox_inches="tight")
    output.seek(0)
    rendered = np.asarray(Image.open(output))
    assert rendered.shape[0] > 100 and rendered.shape[1] > 100
    assert rendered[..., :3].std() > 5
    return renderer, pixels


def assert_inside(inner, outer):
    assert (
        inner.x0 >= outer.x0
        and inner.y0 >= outer.y0
        and inner.x1 <= outer.x1
        and inner.y1 <= outer.y1
    ), f"Visible label bounds {inner.bounds} exceed exported bounds {outer.bounds}"


@pytest.mark.parametrize("source", ["notebook", "generator"])
def test_latent_plot_covers_training_times_and_preserves_subnet_demonstration(
    source, figures, tmp_path
):
    tw = TravelingWave()
    data = tw.data.copy()
    with torch.random.fork_rng():
        torch.manual_seed(271)
        model = nif.NIF(
            dict(
                connectivity="full",
                input_dim=1,
                output_dim=1,
                units=30,
                nlayers=2,
                activation="swish",
            ),
            dict(input_dim=1, latent_dim=1, units=30, nlayers=2, activation="swish"),
        )
    scope = {
        "torch": torch,
        "np": np,
        "plt": plt,
        "OUT": tmp_path,
        "train_data": data,
        "model_ori": model,
        "model_opt": model,
        "tensor": lambda x: torch.as_tensor(x, dtype=torch.float32),
    }
    state = copy.deepcopy(model.state_dict())
    modes = [m.training for m in model.modules()]
    rng = torch.get_rng_state().clone()
    numpy_rng = np.random.get_state()
    exec(selected_cell(source, 1, "model_p_to_lr ="), scope)
    fig = plt.gcf()
    line = fig.axes[0].lines[0]
    plotted_times = np.asarray(line.get_xdata()).ravel()
    np.testing.assert_array_equal(
        np.unique(plotted_times), np.unique(data[:, 0]).astype(np.float32)
    )
    with torch.no_grad():
        expected = (
            model.model_p_to_lr()(scope["tensor"](plotted_times[:, None]))
            .numpy()
            .ravel()
        )
    np.testing.assert_allclose(
        np.asarray(line.get_ydata()).ravel(), expected, rtol=0, atol=0
    )
    exec(selected_cell(source, 1, "model_lr_to_w ="), scope)
    exec(selected_cell(source, 1, "model_x_to_u_given_w ="), scope)
    assert scope["p"].shape == (32, 1) and scope["latent"].shape[0] == 32
    assert scope["w"].shape[0] == 32 and scope["latent"].requires_grad
    for name, value in state.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    assert modes == [m.training for m in model.modules()]
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert np.random.get_state()[0] == numpy_rng[0]
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
    assert np.random.get_state()[2:] == numpy_rng[2:]
    np.testing.assert_array_equal(data, tw.data)
    tight_render(fig)


@pytest.mark.parametrize("source", ["notebook", "generator"])
def test_regularization_fields_share_color_mapping_and_readable_layout(
    source, figures, tmp_path
):
    code = selected_cell(source, 6, "def run_regularization")
    function = next(
        n
        for n in ast.parse(code).body
        if isinstance(n, ast.FunctionDef) and n.name == "run_regularization"
    )
    # Execute the actual plotting block only: deliberately do not call training,
    # construct an optimizer, or replace the scientific predictions in a notebook.
    start = next(
        i
        for i, node in enumerate(function.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Tuple)
            and any(isinstance(v, ast.Name) and v.id == "fig" for v in t.elts)
            for t in node.targets
        )
    )
    stop = next(
        i
        for i, node in enumerate(function.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "physical_prediction"
            for t in node.targets
        )
    )
    plotting = ast.Module(body=function.body[start:stop], type_ignores=[])
    tw = TravelingWaveHighFreq()
    data = tw.data.copy()
    xx, tt = np.meshgrid(
        np.linspace(0, 1, 200, endpoint=False), np.linspace(0, 100, 10, endpoint=False)
    )
    test_xx, test_tt = np.meshgrid(np.linspace(-1, 1, 41), np.linspace(-1, 1, 41))
    train_pred = 0.02 + np.linspace(-1e-10, 1e-10, len(data))
    test_pred = 0.02 + 1e-10 * np.sin(test_xx) * np.cos(test_tt)
    originals = [v.copy() for v in (data, train_pred, test_pred)]
    scope = dict(
        np=np,
        plt=plt,
        OUT=tmp_path,
        train_data=data,
        train_pred=train_pred,
        test_pred=test_pred,
        tt=tt,
        xx=xx,
        test_tt=test_tt,
        test_xx=test_xx,
        name="l2_0.01",
    )
    exec(compile(plotting, "<notebook contour display>", "exec"), scope)
    fig, axes = scope["fig"], scope["axs"]
    contours = [
        next(c for c in ax.collections if isinstance(c, QuadContourSet)) for ax in axes
    ]
    for contour in contours[1:]:
        np.testing.assert_array_equal(contour.levels, contours[0].levels)
        np.testing.assert_array_equal(
            contour.to_rgba([0.0, 0.02, 0.1]), contours[0].to_rgba([0.0, 0.02, 0.1])
        )
    low, high = contours[0].get_clim()
    assert low <= min(v.min() for v in (data[:, 2], train_pred, test_pred))
    assert high >= max(v.max() for v in (data[:, 2], train_pred, test_pred))
    # Tiny prediction variations must occupy a tiny part of the true field scale.
    assert np.ptp(contours[1].norm(train_pred)) < 1e-7
    renderer, crop = tight_render(fig)
    text = [fig._suptitle] + [ax.title for ax in axes]
    for ax in fig.axes:
        text.extend(
            [
                ax.xaxis.label,
                ax.yaxis.label,
                ax.xaxis.get_offset_text(),
                ax.yaxis.get_offset_text(),
            ]
        )
        if ax not in axes:
            text.extend(ax.get_xticklabels() + ax.get_yticklabels())
    bounds = [
        artist.get_window_extent(renderer)
        for artist in text
        if artist is not None and artist.get_visible() and artist.get_text()
    ]
    for bound in bounds:
        assert_inside(bound, crop)
    for i, a in enumerate(bounds):
        for b in bounds[i + 1 :]:
            overlap = Bbox.intersection(a, b)
            assert (
                overlap is None or overlap.width * overlap.height <= 1
            ), "Display annotations overlap"
    for actual, original in zip((data, train_pred, test_pred), originals):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("source", ["notebook", "generator"])
@pytest.mark.parametrize("number", [1, 2, 6, 8])
def test_initial_surface_labels_survive_ordinary_inline_tight_crop(
    source, number, figures
):
    scope = {"np": np, "plt": plt}
    exec(selected_cell(source, number, "ax.plot_surface"), scope)
    ax = scope["ax"]
    renderer, crop = tight_render(ax.figure)
    for axis, expected in zip((ax.xaxis, ax.yaxis, ax.zaxis), ("x", "t", "u")):
        assert expected in axis.label.get_text()
        bounds = axis.label.get_window_extent(renderer)
        assert_inside(bounds, ax.figure.bbox)
        assert_inside(bounds, crop)
        low, high = sorted(axis.get_view_interval())
        for tick in axis.get_major_ticks():
            if low <= tick.get_loc() <= high and tick.label1.get_visible():
                overlap = Bbox.intersection(
                    bounds, tick.label1.get_window_extent(renderer)
                )
                assert (
                    overlap is None or overlap.width * overlap.height <= 1
                ), f"{expected}-axis label overlaps a tick annotation"
