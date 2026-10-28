"""Build self-contained native tutorials while retaining the reference teaching narrative."""
from pathlib import Path

import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]

SETUP = r"""# Run this cell first in a clean Colab Python 3.11–3.13 session.
import os, sys, subprocess, importlib.util
from pathlib import Path
# A local checkout is used directly for development; Colab installs the public pytorch branch.
repo = next((p for p in (Path.cwd(), *Path.cwd().parents)
             if (p / "nif" / "model.py").exists()), None)
if repo is not None:
    sys.path.insert(0, str(repo))
else:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
        "torch==2.14.1", "numpy==2.1.3", "scipy==1.16.3", "matplotlib==3.10.0",
        "protobuf==7.36.2", "google-crc32c==1.9.0", "tfrecord==1.14.6",
        "git+https://github.com/pswpswpsw/nif.git@pytorch"])
import copy, json, time, math
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
import nif
# FULL is the default. Set NIF_SMOKE=1 before starting only for a short development check.
SMOKE = os.environ.get("NIF_SMOKE", "0") == "1"
device = torch.device(os.environ.get("NIF_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
torch.manual_seed(42)
np.random.seed(42)
if device.type == "cpu":
    torch.set_num_threads(min(4, os.cpu_count() or 1))
print({"torch": torch.__version__, "device": str(device), "mode": "SMOKE" if SMOKE else "FULL"})
"""
HELPERS = (
    "# Explicit native loops replace compile/fit/callbacks. The loss "
    "reduction is per row,\n"
    "# then the mean over the batch; area weights multiply each row's"
    " MSE (not weight-normalized).\n"
    "def dense_tanh(inputs, outputs):\n"
    "    # Preserve the original Dense default: Glorot uniform kernel"
    " and zero bias.\n"
    "    layer = nn.Linear(inputs, outputs)\n"
    "    nn.init.xavier_uniform_(layer.weight)\n"
    "    nn.init.zeros_(layer.bias)\n"
    "    return nn.Sequential(layer, nn.Tanh())\n"
    "\n"
    "def epochs(full):\n"
    "    return 2 if SMOKE else full\n"
    "\n"
    "def tensor(array, on_device=device):\n"
    "    return torch.as_tensor(array, dtype=torch.float32, device=on"
    "_device)\n"
    "\n"
    "def loader(features, target, batch_size, weight=None):\n"
    '    arrays = [tensor(features, "cpu"), tensor(target, "cpu")]\n'
    "    if weight is not None:\n"
    '        arrays.append(tensor(weight, "cpu").reshape(-1))\n'
    "    return DataLoader(TensorDataset(*arrays), batch_size=batch_s"
    "ize, shuffle=True,\n"
    "                      drop_last=False, pin_memory=device.type =="
    ' "cuda")\n'
    "\n"
    "def predict(model, features, batch_size=1024):\n"
    "    model.eval()\n"
    "    result = []\n"
    "    with torch.no_grad():\n"
    "        for xb in torch.as_tensor(features, dtype=torch.float32)"
    ".split(batch_size):\n"
    "            result.append(model(xb.to(next(model.parameters()).d"
    "evice)).detach().cpu())\n"
    "    return torch.cat(result).numpy()\n"
    "\n"
    "def centralize_gradients(model):\n"
    "    # TF kernels are [input,output]; native Linear stores [outpu"
    "t,input].\n"
    "    # Centralize along the input axes, preserving the original m"
    "athematical operation.\n"
    "    for module in model.modules():\n"
    "        for name, parameter in module.named_parameters(recurse=F"
    "alse):\n"
    "            if parameter.grad is not None and parameter.ndim > 1"
    ":\n"
    "                axes = tuple(range(1, parameter.ndim)) if isinst"
    "ance(module, nn.Linear) else tuple(range(parameter.ndim-1))\n"
    "                parameter.grad.sub_(parameter.grad.mean(dim=axes"
    ", keepdim=True))\n"
    "\n"
    "def save_checkpoint(model, optimizer, path, epoch):\n"
    '    payload = {"model_state": {k: v.detach().cpu() for k,v in mo'
    "del.state_dict().items()},\n"
    '               "optimizer_state": optimizer.state_dict(), "epoch'
    '": epoch,\n'
    '               "framework": "pytorch", "smoke": SMOKE}\n'
    '    if hasattr(model, "cfg_shape_net"):\n'
    "        payload.update(model_class=type(model).__name__, cfg_sha"
    "pe_net=model.cfg_shape_net,\n"
    "                       cfg_parameter_net=model.cfg_parameter_net"
    ', mixed_policy="float32")\n'
    "    torch.save(payload, path)\n"
    "\n"
    "def save_and_check(model, optimizer, name, example, factory=None"
    "):\n"
    '    path = OUT / (name + ".pt")\n'
    "    save_checkpoint(model, optimizer, path, -1)\n"
    "    checkpoint = torch.load(path, map_location=device, weights_o"
    "nly=True)\n"
    "    if factory is None:\n"
    "        factory = lambda: type(model)(copy.deepcopy(model.cfg_sh"
    "ape_net),\n"
    "                                       copy.deepcopy(model.cfg_p"
    'arameter_net), "float32")\n'
    "    restored = factory().to(device)\n"
    '    restored.load_state_dict(checkpoint["model_state"])\n'
    "    restored.eval(); model.eval()\n"
    "    with torch.no_grad():\n"
    "        expected, actual = model(tensor(example)), restored(tens"
    "or(example))\n"
    "    torch.testing.assert_close(actual, expected, rtol=1e-5, atol"
    "=1e-6)\n"
    '    print("Checkpoint predictions verified:", path)\n'
    "    return restored\n"
    "\n"
    "def train(model, batches, optimizer, nepoch, name, schedule=None"
    ",\n"
    "          centralization=False, loss_function=None, before_epoch"
    "=None):\n"
    "    history = []\n"
    "    start = time.monotonic()\n"
    "    for epoch in range(nepoch):\n"
    "        model.train()\n"
    "        if schedule is not None:\n"
    "            for group in optimizer.param_groups:\n"
    '                group["lr"] = schedule(epoch, group["lr"])\n'
    "        if before_epoch is not None:\n"
    "            before_epoch(epoch)\n"
    "        summed, seen = {}, 0\n"
    "        for batch in batches:\n"
    "            xb, yb = (item.to(device) for item in batch[:2])\n"
    "            optimizer.zero_grad(set_to_none=True)\n"
    "            if loss_function is None:\n"
    "                per_row = (model(xb) - yb).square().mean(dim=-1)"
    "\n"
    "                if len(batch) == 3:\n"
    "                    per_row = per_row * batch[2].to(device).resh"
    "ape(-1)\n"
    "                data_loss = per_row.mean()\n"
    '                values = {"mse": data_loss}\n'
    "            else:\n"
    "                data_loss, values = loss_function(model, xb, yb)"
    "\n"
    "            regularization = model.regularization_loss(xb) if ha"
    'sattr(model, "regularization_loss") else xb.new_zeros(())\n'
    "            loss = data_loss + regularization\n"
    "            if not torch.isfinite(loss):\n"
    '                raise FloatingPointError(f"Non-finite loss in {n'
    'ame}, epoch {epoch}")\n'
    "            loss.backward()\n"
    "            if centralization:\n"
    "                centralize_gradients(model)\n"
    "            optimizer.step()\n"
    '            for key, value in {**values, "regularization": regul'
    'arization, "loss": loss}.items():\n'
    "                summed[key] = summed.get(key, 0.0) + float(value"
    ".detach()) * len(xb)\n"
    "            seen += len(xb)\n"
    '        row = {"epoch": epoch, "rows": seen, **{key: value / see'
    "n for key,value in summed.items()}}\n"
    "        history.append(row)\n"
    "        if epoch % 100 == 0 or epoch == nepoch - 1:\n"
    '            print(name, row, "elapsed_seconds", round(time.monot'
    "onic()-start, 2))\n"
    '            (OUT / (name + "_history.json")).write_text(json.dum'
    "ps(history, indent=2))\n"
    "        if epoch % 1000 == 0 or epoch == nepoch - 1:\n"
    '            save_checkpoint(model, optimizer, OUT / (name + "_la'
    'test.pt"), epoch)\n'
    "    fig, ax = plt.subplots(figsize=(7, 3))\n"
    '    ax.semilogy([row["loss"] for row in history]); ax.set(xlabel'
    '="epoch", ylabel="loss", title=name)\n'
    '    fig.tight_layout(); fig.savefig(OUT / (name + "_loss.png"));'
    " plt.show(); plt.close(fig)\n"
    "    return history\n"
    "\n"
    "def wave_figure(truth, prediction, name):\n"
    "    fig, axs = plt.subplots(1,3,figsize=(15,3))\n"
    "    for ax, values, title in zip(axs, (truth,prediction,predicti"
    'on-truth), ("true","pred","error")):\n'
    "        im=ax.contourf(tt,xx,values.reshape(10,200),levels=50,cm"
    'ap="seismic")\n'
    '        fig.colorbar(im,ax=ax); ax.set(xlabel="t",ylabel="x",tit'
    "le=title)\n"
    '    fig.tight_layout(); fig.savefig(OUT / (name+"_reconstruction'
    '.png")); plt.show(); plt.close(fig)\n'
    "\n"
    "def write_metrics(values):\n"
    "    print(json.dumps(values, indent=2))\n"
    '    (OUT / "metrics.json").write_text(json.dumps({"mode": "smoke'
    '" if SMOKE else "full", **values}, indent=2))\n'
)
SWISH = """cfg_shape_net = {"connectivity": "full", "input_dim": 1, "output_dim": 1,
                 "units": 30, "nlayers": 2, "activation": "swish"}
cfg_parameter_net = {"input_dim": 1, "latent_dim": 1, "units": 30,
                     "nlayers": 2, "activation": "swish"}
"""
SIREN = """cfg_shape_net = {"use_resblock": False, "connectivity": "full", "input_dim": 1,
                 "output_dim": 1, "units": 30, "nlayers": 2,
                 "weight_init_factor": 0.01, "omega_0": 30.0}
cfg_parameter_net = {"use_resblock": False, "input_dim": 1, "latent_dim": 1,
                     "units": 30, "nlayers": 2, "activation": "swish"}
"""
WAVE_DATA = """from nif.demo import TravelingWaveHighFreq
tw = TravelingWaveHighFreq()
train_data = tw.data
train_dataset = loader(train_data[:, :2], train_data[:, 2:3], batch_size=512)
"""
SCHEDULE_MS = """def scheduler(epoch, lr):
    if epoch < 1000: return lr
    if epoch < 2000: return 1e-4
    if epoch < 4000: return 5e-5
    return 1e-5
"""
SURFACE_LAYOUT = """
# Keep all three axis labels inside the ordinary inline tight crop.
ax.set_box_aspect(None, zoom=0.75)
ax.xaxis.labelpad = ax.yaxis.labelpad = ax.zaxis.labelpad = 12
"""


def original(number):
    return next(
        p
        for p in (ROOT / ("tutorial")).glob(str(number) + ("_*.ipynb"))
        if not p.stem.endswith((("_colab"), ("_pytorch")))
    )


def markdown(source):
    # Retain scientific narrative, modernize framework-specific instructions.
    changes = {
        (
            "2. output the real NIF model that needs to be trained, and compi"
            "le this model\n"
            "  - note that here we take Adam optimizer, but it won't be used,"
            " it is just for compilation purpose."
        ): (
            "2. Reconstruct the native module from its saved configuration an"
            "d load its state dictionary. Verify predictions after restoratio"
            "n; no compilation is needed."
        ),
        (
            "we first prune 10% then, train for 1000 epochs. We repeat 50 tim"
            "es of this process, each time we prune a bit more, until we hit "
            "the limit of 90% of included weights pruned."
        ): (
            "the cubic schedule starts at 10%, but its first mask refresh is "
            "before training step 4,000 (epoch 1,000), after 3,999 dense step"
            "s. It refreshes every 4,000 steps, for 50 refreshes, reaching 90"
            "% before step 200,000. We remask at every epoch end without an e"
            "xtra final refresh."
        ),
        "`tfmot.sparsity.keras.PolynomialDecay`": "the native cubic magnitude-pruning schedule",
        "`TFPLBFGS`": "native `ReferenceLBFGS`",
        "given the `n_features = 3` and `n_target = 3`": "given the `n_feature = 4` and `n_target = 3`",
        ("So we have 7 `Example` in this demo."): (
            "Each shard has one `Example` containing seven column-vector feat" "ures."
        ),
        "keras.Model": "torch.nn.Module",
        "Keras Model": "PyTorch module",
        "`Keras`": "the explicit PyTorch loop",
        "`tf.data.Dataset`": "a `DataLoader`",
        "`model.fit`": "the explicit training loop",
        ".fit`": "the training loop`",
        "tensorflow_model_optimization": "nif.pruning.PolynomialMagnitudePruning",
        "https://www.tensorflow.org/model_optimization": "../docs/PYTORCH_API.md",
        "HDF5 format": "native state_dict format",
        "tflite": "TorchScript",
        "TFlite": "TorchScript",
        "`Callback`": "training-loop hook",
        "`callbacks`": "training-loop hooks",
        "`TensorBoard`": "loss logging",
        "tensorboard": "loss logging",
    }
    for a, b in changes.items():
        source = source.replace(a, b)
    return source


T7_INFERENCE_HELPERS = r"""

def save_inference_and_check(model, name, example, factory=None):
    # Match include_optimizer=False: benchmark only model parameters/configuration.
    # train() still writes separate *_latest.pt checkpoints with optimizer state.
    path = OUT / (name + ".pt")
    payload = {"model_state": {k: v.detach().cpu() for k,v in model.state_dict().items()},
               "framework": "pytorch", "artifact_format": "pytorch_state_dict"}
    if hasattr(model, "cfg_shape_net"):
        payload.update(model_class=type(model).__name__, cfg_shape_net=model.cfg_shape_net,
                       cfg_parameter_net=model.cfg_parameter_net, mixed_policy="float32")
    torch.save(payload, path)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if factory is None:
        factory = lambda: type(model)(copy.deepcopy(model.cfg_shape_net),
                                       copy.deepcopy(model.cfg_parameter_net), "float32")
    restored = factory().to(device)
    restored.load_state_dict(checkpoint["model_state"])
    restored.eval(); model.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(tensor(example)), model(tensor(example)),
                                   rtol=1e-5, atol=1e-6)
    print("Inference predictions verified:", path)
    return restored
"""


def emit(number, replacements, notes="", omitted_markdown=()):
    path = original(number)
    src = nb.read(path, as_version=4)
    cells = [
        nb.v4.new_markdown_cell(
            f"# Native PyTorch tutorial {number}\n\n"
            + notes
            + (
                "\n"
                "\n"
                "Run all cells in order. Full scientific workloads are the defaul"
                "t; `NIF_SMOKE=1` is a separate development run and does not esta"
                "blish full convergence. The original TensorFlow and `_colab` not"
                "ebooks are preserved as legacy references."
            )
        ),
        nb.v4.new_code_cell(
            SETUP
            + (
                '\nOUT = Path(os.environ.get("NIF_OUTPUT_DIR", "pytorch_outputs")) / '
                f'"tutorial{number}"\n'
                "OUT.mkdir(parents=True, exist_ok=True)\n"
            )
        ),
        nb.v4.new_code_cell(
            HELPERS.replace(
                "before_epoch=None):", "before_epoch=None, after_epoch=None):"
            )
            .replace(
                '        row = {"epoch": epoch',
                (
                    "        if after_epoch is not None:\n"
                    "            after_epoch(epoch)\n"
                    '        row = {"epoch": epoch'
                ),
            )
            .replace(
                "v.detach().cpu() for k,v",
                "v.detach().cpu() if isinstance(v, torch.Tensor) else v for k,v",
            )
            + T7_INFERENCE_HELPERS
            if number == 7
            else HELPERS
        ),
    ]
    for i, c in enumerate(src.cells):
        if c.cell_type == "markdown" and i not in omitted_markdown:
            cells.append(nb.v4.new_markdown_cell(markdown(c.source)))
        if i in replacements:
            s = replacements[i]
            if s.strip():
                cells.append(nb.v4.new_code_cell(s))
    out = nb.v4.new_notebook(
        cells=cells,
        metadata={
            ("kernelspec"): {
                ("display_name"): ("Python 3"),
                ("language"): ("python"),
                ("name"): ("python3"),
            },
            ("language_info"): {("name"): ("python"), ("version"): ("3.13")},
            ("nif"): {
                ("framework"): ("pytorch"),
                ("full_default"): True,
                ("reference"): path.name,
            },
        },
    )
    dest = path.with_name(path.stem + "_pytorch.ipynb")
    nb.write(out, dest)
    print(dest)


src1 = nb.read(original(1), as_version=4)
# Narrative about compilation/distribution is replaced by the explicit native loop notes.
emit(
    1,
    {
        3: src1.cells[3].source + SURFACE_LAYOUT,
        7: SWISH,
        9: """enable_multi_gpu = False
enable_mixed_precision = False
mixed_policy = "float32"
# Set device=NIF_DEVICE/cuda for GPU execution; this full workload uses FP32.
""",
        11: """nepoch = 5000
lr = 5e-3
batch_size = 512
checkpt_epoch, display_epoch, print_figure_epoch = 1000, 100, 100
""",
        13: """# alternatively, get the training dataset from nif.demo for this example
from nif.demo import TravelingWave
tw = TravelingWave()
train_data = tw.data
train_dataset = loader(train_data[:, :2], train_data[:, -1:], batch_size)
""",
        19: """def scheduler(epoch, lr):
    if epoch < 1000: return lr
    if epoch < 2000: return 1e-3
    if epoch < 4000: return 5e-4
    return 1e-4
""",
        23: """from nif.optimizers import AdaBeliefOptimizer
model_ori = nif.NIF(cfg_shape_net, cfg_parameter_net, mixed_policy).to(device)
model_opt = model_ori.build()
assert model_opt is model_ori
# Match actual current TensorFlow tutorial 1: its Colab fit does not activate GC.
# Set True to opt in to native GC; this changes that primary reference behavior.
use_gradient_centralization = False
optimizer = AdaBeliefOptimizer(model_opt.parameters(), lr=lr)
history = train(model_opt, train_dataset, optimizer, epochs(nepoch), "traveling_wave",
                schedule=scheduler, centralization=use_gradient_centralization)
wave_figure(train_data[:, -1], predict(model_opt, train_data[:, :2]).ravel(), "traveling_wave")
""",
        27: (
            "model_p_to_lr = model_ori.model_p_to_lr()\n"
            "print(model_p_to_lr)\n"
            "p = tensor(train_data[:32, :1])\n"
            "latent = model_p_to_lr(p)\n"
            "# Plot every training time separately from the 32-row shared-sub"
            "net example.\n"
            "plot_p = tensor(np.unique(train_data[:, 0]).reshape(-1, 1))\n"
            "with torch.no_grad():\n"
            "    plot_latent = model_p_to_lr(plot_p)\n"
            'fig, ax=plt.subplots(); ax.plot(plot_p.cpu(),plot_latent.cpu(),"'
            'o"); ax.set(xlabel="normalized t",ylabel="latent")\n'
            'fig.savefig(OUT / "latent.png"); plt.show(); plt.close(fig)\n'
        ),
        29: """torch.save(model_p_to_lr.state_dict(), OUT / "parameter_to_latent.pt")
""",
        31: """model_lr_to_w = model_ori.model_lr_to_w()
w = model_lr_to_w(latent)
torch.testing.assert_close(w, model_ori.model_p_to_w()(p))
print("latent-to-weights:", w.shape)
""",
        33: """model_x_to_u_given_w = model_ori.model_x_to_u_given_w()
torch.testing.assert_close(model_x_to_u_given_w([tensor(train_data[:32,1:2]), w]),
                           model_opt(tensor(train_data[:32,:2])))
print(model_x_to_u_given_w)
""",
        40: """new_model = save_and_check(model_opt, optimizer, "traveling_wave", train_data[:64,:2])
""",
        44: """data_feature = tensor(train_data[:, :2])
data_label = tensor(train_data[:, -1:])
""",
        46: """# Reference L-BFGS uses a closure: recompute loss and gradients at every line search.
from nif.optimizers import ReferenceLBFGS
lbfgs_history = []
rounds, max_iter = (1, 2) if SMOKE else (200, 1000)
# Original TFPLBFGS controls, retained independently in full and smoke modes.
tolerance = x_tolerance = f_relative_tolerance = 1e-15
num_correction_pairs = 20
max_line_search_iterations = 100
# Each round creates a fresh optimizer with up to max_iter additional iterations.
# Hager-Zhang line search has its own per-search iteration limit.
""",
        48: (
            "for round_index in range(rounds):\n"
            "    fine_tuner = ReferenceLBFGS(new_model.parameters(), max_iter"
            "=max_iter,\n"
            "                                num_correction_pairs=num_correct"
            "ion_pairs,\n"
            "                                tolerance=tolerance, x_tolerance"
            "=x_tolerance,\n"
            "                                f_relative_tolerance=f_relative_"
            "tolerance,\n"
            "                                f_absolute_tolerance=0.,\n"
            "                                max_line_search_iterations=max_l"
            "ine_search_iterations)\n"
            "    def closure():\n"
            "        fine_tuner.zero_grad(set_to_none=True)\n"
            "        loss = (new_model(data_feature)-data_label).square().mea"
            "n() + new_model.regularization_loss(data_feature)\n"
            '        if not torch.isfinite(loss): raise FloatingPointError("n'
            'on-finite LBFGS loss")\n'
            "        loss.backward(); lbfgs_history.append(float(loss.detach("
            ")))\n"
            "        return loss\n"
            "    fine_tuner.step(closure)\n"
            'new_model = save_and_check(new_model, fine_tuner, "fine_tuned", '
            "train_data[:64,:2])\n"
            "fig,ax=plt.subplots(figsize=(8,2)); ax.semilogy(lbfgs_history); "
            'ax.set(xlabel="closure evaluation",ylabel="loss")\n'
            'fig.tight_layout(); fig.savefig(OUT / "fine_tune_loss.png"); plt'
            ".show(); plt.close(fig)\n"
        ),
        50: """prediction = predict(new_model, train_data[:, :2]).ravel()
wave_figure(train_data[:,-1], prediction, "fine_tuned")
write_metrics({"adabelief_epochs": epochs(nepoch), "lbfgs_rounds": rounds, "lbfgs_max_iter": max_iter,
               "final_mse": float(np.mean((prediction-train_data[:,-1])**2))})
""",
    },
    notes=(
        "5,000 AdaBelief epochs without gradient centralization by defaul"
        "t, followed by 200 L-BFGS rounds of up to 1,000 iterations. Nati"
        "ve build/model return the same trainable module. Logs and checkp"
        "oints are written by the visible training loop; GPU selection us"
        "es PyTorch devices."
    ),
    omitted_markdown=(14, 16, 20, 22, 35, 37, 39),
)

src2 = nb.read(original(2), as_version=4)
emit(
    2,
    {
        3: src2.cells[3].source + SURFACE_LAYOUT,
        5: SWISH
        + WAVE_DATA
        + """nepoch, batch_size, lr = 5000, 512, 1e-3
def scheduler_swish(epoch, lr):
    if epoch < 1000: return lr
    if epoch < 2000: return 5e-4
    if epoch < 4000: return 1e-4
    return 5e-4
""",
        7: (
            'model_swish = nif.NIF(cfg_shape_net,cfg_parameter_net,"float32")'
            ".to(device)\n"
            "optimizer = nif.optimizers.ReferenceAdam(model_swish.parameters("
            "), lr=lr, eps=1e-7)\n"
            "history_swish = train(model_swish, train_dataset, optimizer, epo"
            'chs(nepoch), "swish", schedule=scheduler_swish)\n'
            'model_swish = save_and_check(model_swish, optimizer, "swish", tr'
            "ain_data[:64,:2])\n"
        ),
        9: SIREN
        + SCHEDULE_MS
        + (
            "model_siren = nif.NIFMultiScale(cfg_shape_net,cfg_parameter_net,"
            '"float32").to(device)\n'
            "optimizer = nif.optimizers.ReferenceAdam(model_siren.parameters("
            "), lr=lr, eps=1e-7)\n"
            "history_siren = train(model_siren, train_dataset, optimizer, epo"
            'chs(nepoch), "siren", schedule=scheduler)\n'
            'model_siren = save_and_check(model_siren, optimizer, "siren", tr'
            "ain_data[:64,:2])\n"
        ),
        11: """pred_swish=predict(model_swish,train_data[:,:2]).ravel()
pred_siren=predict(model_siren,train_data[:,:2]).ravel()
wave_figure(train_data[:,2],pred_swish,"swish")
wave_figure(train_data[:,2],pred_siren,"siren")
write_metrics({"epochs_per_fit":epochs(nepoch),"fits":2,
    "swish_mse":float(np.mean((pred_swish-train_data[:,2])**2)),
    "siren_mse":float(np.mean((pred_siren-train_data[:,2])**2))})
""",
    },
    notes=(
        "Two complete 5,000-epoch fits compare Swish and SIREN at the ori"
        "ginal width, depth, frequency, learning rates and schedules."
    ),
)

src3 = nb.read(original(3), as_version=4)
emit(
    3,
    {
        3: """from nif.demo import CylinderFlow
tmp = CylinderFlow()
train_data, sample_weight = tmp.data, tmp.sample_weight
assert len(train_data) == 191775
print(train_data.shape, sample_weight.shape, tmp.std)
""",
        6: src3.cells[6].source,
        7: """nepoch, lr, batch_size = 600, 1e-3, 1024
# Only smoke mode selects fewer training rows; FULL uses all 191,775 AMR points each epoch.
training_rows = 4096 if SMOKE else len(train_data)
train_dataset = loader(train_data[:training_rows,:3], train_data[:training_rows,3:5],
                       batch_size, sample_weight[:training_rows])
def scheduler(epoch, lr):
    if epoch < 200: return lr
    if epoch < 400: return 5e-4
    if epoch < 600: return 2e-4
    return 3e-5
""",
        10: (
            "from nif.optimizers import AdaBeliefOptimizer\n"
            "model_ori = nif.NIFMultiScaleLastLayerParameterized(cfg_shape_ne"
            't,cfg_parameter_net,"float32").to(device)\n'
            "model = model_ori.build()\n"
            "optimizer = AdaBeliefOptimizer(model.parameters(),lr=lr)\n"
            'history = train(model,train_dataset,optimizer,epochs(nepoch),"cy'
            'linder",schedule=scheduler,centralization=True)\n'
            'model = save_and_check(model,optimizer,"cylinder",train_data[:64'
            ",:3])\n"
            "model_ori = model\n"
        ),
        12: """model_p_to_lr = model_ori.model_p_to_lr()
""",
        13: """t_arr = np.linspace(-1,1,100)
with torch.no_grad():
    lr_arr = model_p_to_lr(tensor(t_arr[:,None])).cpu().numpy()
""",
        14: (
            'fig,ax=plt.subplots();ax.plot(t_arr,lr_arr);ax.set(xlabel="norma'
            'lized t",ylabel="latent representation")\n'
            'fig.savefig(OUT / "latent.png");plt.show();plt.close(fig)\n'
            "# The spatial basis and time coefficients reconstruct the same l"
            "earned linear representation.\n"
            "phi = model.model_x_to_phi()(tensor(train_data[:32,1:3]))\n"
            'print("spatial basis shape",tuple(phi.shape))\n'
        ),
        16: """input_pred = train_data[:,:3]
time_jump_index = np.r_[np.flatnonzero(np.diff(input_pred[:,0]) != 0)+1, len(input_pred)]
uv_pred = predict(model,input_pred)
""",
        18: src3.cells[18].source
        + """\nplt.savefig(OUT / "amr_mesh_counts.png");plt.show();plt.close()
""",
        20: src3.cells[20].source
        + """\nfig.savefig(OUT / "cylinder_reconstruction.png");plt.show();plt.close(fig)
per_row = np.mean((uv_pred-train_data[:,3:5])**2,axis=1)
write_metrics({"epochs":epochs(nepoch),"training_rows":training_rows,"all_rows":len(train_data),
               "weighted_mse":float(np.mean(per_row*np.asarray(sample_weight).reshape(-1))),
               "physical_u_mse":float(np.mean(((uv_pred[:,0]-train_data[:,3])*tmp.std[3])**2)),
               "physical_v_mse":float(np.mean(((uv_pred[:,1]-train_data[:,4])*tmp.std[4])**2))})
""",
    },
    notes=(
        "600 epochs on all 191,775 cylinder-flow samples, area-weighted l"
        "oss and actual gradient centralization. The input ordering is [t"
        ",x,y]."
    ),
)

emit(
    4,
    {
        4: """model = dense_tanh(4,5).to(device)
""",
        6: """from nif.layers import JacobianLayer, HessianLayer
""",
        8: """x_index, y_index = [0,1,2,3], [0,1,2,3,4]
""",
        10: """y_and_dydx_layer = JacobianLayer(model,y_index,x_index)
""",
        12: """x_data=torch.rand(10,4,device=device)
y_pred,dydx=y_and_dydx_layer(x_data)
""",
        14: """assert y_pred.shape == (10,5) and dydx.shape == (10,5,4)
print("f(x) shape",y_pred.shape,"df(x)/dx shape",dydx.shape)
""",
        16: """y_and_dydx_and_dy2dx2_layer=HessianLayer(model,y_index,x_index)
y_pred,dydx,dy2dx2=y_and_dydx_and_dy2dx2_layer(x_data)
""",
        18: """assert dy2dx2.shape == (10,5,4,4)
print("d2f(x)/dx2 shape",dy2dx2.shape)
# Analytic tanh(linear(x)) Jacobian and Hessian validate actual values.
w=model[0].weight
expected_j=(1-y_pred.square())[:,:,None]*w[None,:,:]
expected_h=(-2*y_pred*(1-y_pred.square()))[:,:,None,None]*w[None,:,:,None]*w[None,:,None,:]
torch.testing.assert_close(dydx,expected_j)
torch.testing.assert_close(dy2dx2,expected_h)
# Selecting derivatives does not truncate the full output.
y_full,j_selected,h_selected=HessianLayer(model,[4,1],[3,0])(x_data)
assert y_full.shape == (10,5)
torch.testing.assert_close(j_selected,dydx[:,[4,1]][:,:,[3,0]])
torch.testing.assert_close(h_selected,dy2dx2[:,[4,1]][:,:,[3,0]][:,:,:,[3,0]])
# Linear/constant functions have valid zero higher derivatives.
linear=nn.Linear(4,5).to(device)
_,_,h_linear=HessianLayer(linear,[0],[0,1])(x_data)
torch.testing.assert_close(h_linear,torch.zeros_like(h_linear))
class Constant(nn.Module):
    def __init__(self):
        super().__init__(); self.value=nn.Parameter(torch.ones(5))
    def forward(self,x): return self.value.expand(len(x),-1)
constant=Constant().to(device)
_,j_constant,h_constant=HessianLayer(constant,[0],[0,1])(x_data)
torch.testing.assert_close(j_constant,torch.zeros_like(j_constant))
torch.testing.assert_close(h_constant,torch.zeros_like(h_constant))
# Derivative penalties must differentiate the trainable parameters, as in Sobolev training.
optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,eps=1e-7)
optimizer.zero_grad()
derivative_loss=dydx.square().mean()+dy2dx2.square().mean()
derivative_loss.backward()
assert model[0].weight.grad is not None and torch.isfinite(model[0].weight.grad).all()
assert model[0].weight.grad.norm()>0
optimizer.step()
model=save_and_check(model,optimizer,"derivatives",x_data.cpu(),factory=lambda: dense_tanh(4,5))
fig,ax=plt.subplots();im=ax.imshow(dydx[0].detach().cpu());fig.colorbar(im,ax=ax)
ax.set(xlabel="input index",ylabel="output index",title="Jacobian of first example")
fig.savefig(OUT / "jacobian.png");plt.show();plt.close(fig)
write_metrics({"jacobian_max_error":float((dydx-expected_j).abs().max().detach()),
               "hessian_max_error":float((dy2dx2-expected_h).abs().max().detach()),
               "derivative_loss":float(derivative_loss.detach())})
""",
    },
    notes=(
        "Full outputs, selected Jacobians/Hessians, analytic numerical ch"
        "ecks, zero higher derivatives and differentiable derivative pena"
        "lties."
    ),
)

emit(
    5,
    {
        3: """n_rows = 10000 if SMOKE else 1000000
points_per_shard = 1000 if SMOKE else 10000
data = np.random.uniform(0,1,(n_rows,7))
np.savez(OUT / "Big_npz_file.npz",data=data)
""",
        6: """from nif.data.tfr_dataset import TFRDataset
""",
        8: """fh = TFRDataset(n_feature=4,n_target=3)
""",
        11: """# Genuine TFRecord framing and tf.train.Example protobufs, without a TensorFlow runtime.
tfr_path = OUT / "TFR_dir"
fh.create_from_npz(num_pts_per_file=points_per_shard,npz_path=str(OUT / "Big_npz_file.npz"),
                   npz_key="data",tfr_path=str(tfr_path),prefix="case1")
shards=sorted(tfr_path.glob("*.tfrecord"))
assert len(shards) == (10 if SMOKE else 100)
print("TFRecord shards:",len(shards))
""",
        16: """model=dense_tanh(4,3).to(device)
from nif.optimizers import ReferenceAdam
optimizer=ReferenceAdam(model.parameters(),lr=1e-3,eps=1e-7)
""",
        18: """epoch=2
meta_dataset=fh.get_tfr_meta_dataset(tfr_path=str(tfr_path),epoch=epoch)
""",
        20: (
            "batch_size=128\n"
            "rows_seen, shard_visits, accumulated_loss = 0,0,0.0\n"
            "stream_losses=[]\n"
            "# Two complete passes through every shard, including every final"
            " batch/remainder.\n"
            "for batch_file in meta_dataset:\n"
            "    batch_dataset=fh.gen_dataset_from_batch_file(batch_file,batc"
            "h_size)\n"
            "    shard_rows,shard_loss=0,0.0\n"
            "    for xb,yb in batch_dataset:\n"
            "        xb,yb=xb.to(device),yb.to(device)\n"
            "        optimizer.zero_grad(set_to_none=True)\n"
            "        loss=(model(xb)-yb).square().mean()\n"
            '        if not torch.isfinite(loss): raise FloatingPointError("n'
            'on-finite streamed loss")\n'
            "        loss.backward();optimizer.step()\n"
            "        shard_rows+=len(xb);shard_loss+=float(loss.detach())*len"
            "(xb)\n"
            "    rows_seen+=shard_rows;shard_visits+=1;accumulated_loss+=shar"
            "d_loss\n"
            "    stream_losses.append(shard_loss/shard_rows)\n"
            "assert rows_seen == 2*n_rows\n"
            "assert shard_visits == 2*len(shards)\n"
            'model=save_and_check(model,optimizer,"streamed",data[:64,:4],fac'
            "tory=lambda:dense_tanh(4,3))\n"
            'fig,ax=plt.subplots();ax.plot(stream_losses);ax.set(xlabel="shar'
            'd visit",ylabel="MSE",title="Two streamed passes")\n'
            'fig.savefig(OUT / "streamed_loss.png");plt.show();plt.close(fig)'
            "\n"
            'write_metrics({"generated_rows":n_rows,"shards":len(shards),"pas'
            'ses":epoch,\n'
            '               "rows_seen":rows_seen,"shard_visits":shard_visits'
            ',"mse":accumulated_loss/rows_seen})\n'
        ),
    },
    notes=(
        "Full mode generates one million rows, writes 100 original-format"
        " TFRecord shards and trains through two complete streamed passes"
        ". A native DataLoader batches each shard; global counters retain"
        " epoch/row information."
    ),
)

src6 = nb.read(original(6), as_version=4)
REG_TRAIN = (
    "def run_regularization(kind=None, coefficient=0.0):\n"
    "    config=copy.deepcopy(cfg_parameter_net)\n"
    '    if kind is not None: config[kind+"_reg"]=coefficient\n'
    '    name="no_reg" if kind is None else f"{kind}_{coefficient}"\n'
    "    model=nif.NIFMultiScale(copy.deepcopy(cfg_shape_net),config,"
    '"float32").to(device)\n'
    "    optimizer=nif.optimizers.ReferenceAdam(model.parameters(),lr"
    "=1e-3,eps=1e-7)\n"
    "    history=train(model,train_dataset,optimizer,epochs(nepoch),n"
    "ame)\n"
    "    model=save_and_check(model,optimizer,name,train_data[:64,:2]"
    ")\n"
    "    train_pred=predict(model,train_data[:,:2]).ravel()\n"
    "    test_pred=predict(model,test_data).reshape(test_tt.shape)\n"
    '    fig,axs=plt.subplots(1,3,figsize=(15,3.5),layout="constraine'
    'd")\n'
    "    # A common normalized-u scale makes nearly constant predicti"
    "ons look constant.\n"
    "    levels=np.linspace(min(train_data[:,2].min(),train_pred.min("
    "),test_pred.min()),\n"
    "                       max(train_data[:,2].max(),train_pred.max("
    "),test_pred.max()),51)\n"
    "    for ax,values,title in zip(axs[:2],(train_data[:,2],train_pr"
    'ed),("true","pred")):\n'
    "        im=ax.contourf(tt,xx,values.reshape(10,200),levels=level"
    's,cmap="seismic");ax.set_title(title)\n'
    "    im=axs[2].contourf(test_tt,test_xx,test_pred,levels=levels,c"
    'map="seismic");axs[2].set_title("super-resolution")\n'
    '    fig.colorbar(im,ax=axs,label="normalized u")\n'
    '    fig.suptitle(name);fig.savefig(OUT/(name+"_superresolution.p'
    'ng"));plt.show();plt.close(fig)\n'
    "    physical_prediction=test_pred*tw.std[2]+tw.mean[2]\n"
    '    result={"kind":kind,"coefficient":coefficient,"epochs":epoch'
    "s(nepoch),\n"
    '            "training_mse":float(np.mean((train_pred-train_data['
    ":,2])**2)),\n"
    '            "physical_superresolution_mse":float(np.mean((physic'
    "al_prediction-u_test)**2)),\n"
    '            "regularization_loss":float(model.regularization_los'
    "s(tensor(train_data[:64,:2])).detach())}\n"
    "    results.append(result)\n"
    "    return model\n"
)
emit(
    6,
    {
        4: src6.cells[4].source + SURFACE_LAYOUT,
        5: """# prepare testing data: finer normalized coordinates; ALWAYS [parameter, coordinate] = [t,x].
resolution=41 if SMOKE else 371
test_x=np.linspace(-1,1,resolution)
test_t=np.linspace(-1,1,resolution)
test_xx,test_tt=np.meshgrid(test_x,test_t)
test_data=np.column_stack([test_tt.ravel(),test_xx.ravel()])
""",
        7: SIREN
        + """nepoch,lr,batch_size = 3000,1e-3,512
results=[]
""",
        10: WAVE_DATA
        + """test_x_phys=test_xx*tw.std[1]+tw.mean[1]
test_t_phys=test_tt*tw.std[0]+tw.mean[0]
u_test=np.exp(-1000*(test_x_phys-x0-c*test_t_phys)**2)*np.sin(omega*(test_x_phys-x0-c*test_t_phys))
"""
        + REG_TRAIN
        + """baseline=run_regularization()
""",
        14: """model_l2_1e5=run_regularization("l2",1e-5)
""",
        17: """model_l2_1e4=run_regularization("l2",1e-4)
""",
        20: """model_l2_1e3=run_regularization("l2",1e-3)
""",
        23: """model_l2_1e2=run_regularization("l2",1e-2)
""",
        28: """model_l1_1e5=run_regularization("l1",1e-5)
""",
        30: """model_l1_1e4=run_regularization("l1",1e-4)
""",
        32: """model_l1_1e3=run_regularization("l1",1e-3)
""",
        34: """model_l1_1e2=run_regularization("l1",1e-2)
""",
        35: """assert len(results)==9
write_metrics({"fits":9,"epochs_per_fit":epochs(nepoch),"resolution":resolution,"results":results})
""",
    },
    notes=(
        "Nine 3,000-epoch fits: baseline, four L2 coefficients and four L"
        "1 coefficients. Regularization is an explicit differentiable los"
        "s, independent of optimizer weight decay. The reference superres"
        "olution input ordering is corrected to [t,x]."
    ),
)

# Pruning and quantization are kept as separate inspectable stages.
emit(
    7,
    {
        4: SIREN
        + SCHEDULE_MS
        + """nepoch,lr,batch_size = 5000,1e-3,512
""",
        8: WAVE_DATA
        + """# Shuffling is handled by DataLoader, keeping the source grid available for diagnostics.
model_ori=nif.NIFMultiScale(cfg_shape_net,cfg_parameter_net,"float32").to(device)
model=model_ori.build()
optimizer=nif.optimizers.ReferenceAdam(model.parameters(),lr=lr,eps=1e-7)
history_baseline=train(model,train_dataset,optimizer,epochs(nepoch),"baseline",schedule=scheduler)
""",
        10: """t=np.linspace(-1,1,10,endpoint=True)
x=np.linspace(-1,1,200,endpoint=True)
tt,xx=np.meshgrid(t,x)
input_feature=np.column_stack([tt.ravel(),xx.ravel()])
model=save_inference_and_check(model,"original",input_feature[:64])
y_pred_unpruned=predict(model,input_feature).reshape(200,10)
""",
        12: """original_file=OUT / "original.pt"
""",
        14: """model_for_pruning=copy.deepcopy(model)
pruning_epochs=50000
prune_epoch=1000
# PolynomialDecay's cubic schedule: begin at 10%, end at 90% of the selected kernel.
# Full mode refreshes before steps 4,000 through 200,000 (50 updates).
# Steps 1 through 3,999 stay dense; no extra refresh follows training.
""",
        16: """print(model_for_pruning)
""",
        18: """from nif.pruning import PolynomialMagnitudePruning
from nif.layers import HyperLinearForSIREN
selected=[(name,layer) for name,layer in model_for_pruning.named_modules()
          if isinstance(layer,HyperLinearForSIREN)]
assert len(selected)==1
selected_name,hyper_siren_layer=selected[0]
print("layer selected =",selected_name,"kernel",tuple(hyper_siren_layer.w.shape))

run_pruning_epochs=epochs(pruning_epochs)
steps_per_epoch=len(train_dataset)
# The separate smoke schedule covers the same stages in two development epochs.
# Full defaults: 2,000 rows / batch 512 -> 4 steps; end=200,000, frequency=4,000.
pruning_controller=PolynomialMagnitudePruning(hyper_siren_layer,"w",
    initial_sparsity=.10,final_sparsity=.90,begin_step=0,
    end_step=run_pruning_epochs*steps_per_epoch,power=3,
    frequency=(1 if SMOKE else prune_epoch)*steps_per_epoch)
# Masks are assigned before each forward, with raw gradients and optimizer slots
# active at masked coordinates. Refresh ranks actual weights, allowing reselection.
""",
        22: """optimizer_prune=nif.optimizers.ReferenceAdam(model_for_pruning.parameters(),lr=1e-4,eps=1e-7)
""",
        26: """history_prune=train(model_for_pruning,train_dataset,optimizer_prune,run_pruning_epochs,
                    "pruning",after_epoch=lambda epoch:pruning_controller.on_epoch_end())
# Epoch-end masking precedes checkpoints, without an extra schedule refresh.
assert pruning_controller.step==run_pruning_epochs*steps_per_epoch
assert pruning_controller.refresh_count==(2 if SMOKE else 50)
""",
        28: (
            'fig,ax=plt.subplots(figsize=(12,4));ax.semilogy([row["loss"] for'
            " row in history_prune])\n"
            "for epoch_index in range((1 if SMOKE else prune_epoch)-1,run_pru"
            "ning_epochs,1 if SMOKE else prune_epoch):ax.axvline(epoch_index,"
            'color="r",alpha=.2)\n'
            'ax.set(xlabel="epoch",ylabel="training loss",title="Magnitude pr'
            'uning: 10% to 90%")\n'
            'fig.savefig(OUT / "pruned_learning_curve.png");plt.show();plt.cl'
            "ose(fig)\n"
        ),
        30: """zero_count=int((hyper_siren_layer.w==0).sum())
weight_count=hyper_siren_layer.w.numel()
sparsity=zero_count/weight_count
assert abs(sparsity-0.90)<=1/weight_count
print({"zero_weights":zero_count,"total_weights":weight_count,"sparsity":sparsity})
""",
        32: """y_pred=predict(model_for_pruning,input_feature).reshape(200,10)
""",
        34: """# Strip the pruning controls: the native .pt contains the actual sparse tensor.
pruning_controller.strip()
model_for_export=save_inference_and_check(model_for_pruning,"stripped",input_feature[:64])
pruned_file=OUT / "stripped.pt"
""",
        36: """model_for_export=model_for_export.cpu().eval()
example=torch.tensor(input_feature[:32],dtype=torch.float32)
traced=torch.jit.trace(model_for_export,example)
exported_file=OUT / "compressed.torchscript.pt"
torch.jit.save(traced,exported_file)
exported=torch.jit.load(exported_file,map_location="cpu")
with torch.no_grad():
    torch.testing.assert_close(exported(example),model_for_export(example))
    torch.testing.assert_close(exported(torch.tensor(input_feature[:17],dtype=torch.float32)),
                               model_for_export(torch.tensor(input_feature[:17],dtype=torch.float32)))
""",
        38: """def get_gzipped_model_size(path):
    # Preserve the reference helper's legacy name; its actual wrapper is ZIP.
    import io, zipfile
    archive = io.BytesIO()
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.write(path)
    return len(archive.getvalue())
""",
        40: (
            "# A real native dynamic int8 kernel executes the generated-weigh"
            "t projection.\n"
            "# The nonlinear SIREN and per-example generated ShapeNet weights"
            " stay floating point.\n"
            "# This is weight/activation dynamic quantization, not a float mo"
            "del saved under an int8 filename.\n"
            "float_quant_model=copy.deepcopy(model_for_export)\n"
            "for name,layer in list(float_quant_model.named_modules()):\n"
            "    if isinstance(layer,HyperLinearForSIREN):\n"
            "        linear=nn.Linear(layer.w.shape[0],layer.w.shape[1])\n"
            "        with torch.no_grad():\n"
            "            linear.weight.copy_(layer.w.T);linear.bias.copy_(lay"
            "er.b)\n"
            '        parent_name,_,child_name=name.rpartition(".")\n'
            "        parent=float_quant_model.get_submodule(parent_name) if p"
            "arent_name else float_quant_model\n"
            "        setattr(parent,child_name,linear)\n"
            "with torch.no_grad():torch.testing.assert_close(float_quant_mode"
            "l(example),model_for_export(example))\n"
            "quantized_model=torch.ao.quantization.quantize_dynamic(float_qua"
            "nt_model,{nn.Linear},dtype=torch.qint8,inplace=False).eval()\n"
            "quantized_linears=[layer for layer in quantized_model.modules()\n"
            "                  if isinstance(layer,torch.ao.nn.quantized.dyna"
            "mic.Linear)]\n"
            "assert quantized_linears and all(layer.weight().dtype==torch.qin"
            "t8 for layer in quantized_linears)\n"
            'quantized_file=OUT / "quantized_and_pruned.torchscript.pt"\n'
            "quantized_script=torch.jit.trace(quantized_model,example)\n"
            "torch.jit.save(quantized_script,quantized_file)\n"
            'restored_quantized=torch.jit.load(quantized_file,map_location="c'
            'pu")\n'
            "with torch.no_grad():\n"
            "    torch.testing.assert_close(restored_quantized(example),quant"
            "ized_model(example),rtol=0,atol=0)\n"
            'print("Native packed int8 Linear kernels:",len(quantized_linears'
            "))\n"
        ),
        42: (
            "with torch.no_grad():\n"
            "    # Full batch predictions are validated after restoring the e"
            "xported artifact.\n"
            "    quant_input=torch.tensor(input_feature,dtype=torch.float32)\n"
            "    y_pred_quant=restored_quantized(quant_input).cpu().numpy().r"
            "eshape(200,10)\n"
            "    torch.testing.assert_close(restored_quantized(quant_input),q"
            "uantized_model(quant_input),rtol=0,atol=0)\n"
            "assert np.isfinite(y_pred_quant).all()\n"
        ),
        46: (
            "fig,axs=plt.subplots(3,2,figsize=(16,16))\n"
            "for column,compressed,difference,label in (\n"
            '        (0,y_pred,y_pred-y_pred_unpruned,"pruned 90%"),\n'
            '        (1,y_pred_quant,y_pred-y_pred_quant,"pruned 90% + int8")'
            "):\n"
            '    for row,values,title in ((0,y_pred_unpruned,"unpruned"),(1,c'
            'ompressed,label),(2,difference,"difference")):\n'
            "        im=axs[row,column].contourf(tt,xx,values,levels=np.linsp"
            'ace(-1,1,50),cmap="seismic")\n'
            "        fig.colorbar(im,ax=axs[row,column]);axs[row,column].set("
            'xlabel="t",ylabel="x",title=title)\n'
            'fig.tight_layout();fig.savefig(OUT / "compression_comparison.png'
            '");plt.show();plt.close(fig)\n'
        ),
        47: (
            '# Legacy "relative MSE" is the squared matrix spectral-norm rati'
            "o (ord=2).\n"
            "denominator=float(np.linalg.norm(y_pred_unpruned,ord=2)**2)\n"
            'write_metrics({"baseline_epochs":epochs(nepoch),"pruning_epochs"'
            ":run_pruning_epochs,\n"
            '    "sparsity":sparsity,"zero_weights":zero_count,"total_weights'
            '":weight_count,\n'
            '    "quantized_linear_count":len(quantized_linears),\n'
            '    "relative_mse_definition":"squared matrix spectral norm of p'
            "rediction difference / squared matrix spectral norm of unpruned "
            'prediction (legacy relative MSE)",\n'
            '    "relative_mse_prune":float(np.linalg.norm(y_pred-y_pred_unpr'
            "uned,ord=2)**2/denominator),\n"
            '    "relative_mse_prune_quant":float(np.linalg.norm(y_pred_quant'
            "-y_pred_unpruned,ord=2)**2/denominator),\n"
            '    "quantization_mse_vs_pruned":float(np.mean((y_pred_quant-y_p'
            "red)**2)),\n"
            '    "compression_method":"ZIP_DEFLATED",\n'
            '    "artifact_formats":{original_file.name:"PyTorch inference st'
            'ate_dict",\n'
            '        pruned_file.name:"PyTorch stripped inference state_dict"'
            ",\n"
            '        exported_file.name:"TorchScript float inference",quantiz'
            'ed_file.name:"TorchScript dynamic int8 inference"},\n'
            '    "zip_deflated_bytes":{p.name:get_gzipped_model_size(p) for p'
            " in (original_file,pruned_file,exported_file,quantized_file)}})\n"
        ),
    },
    notes=(
        "Original 5,000-epoch baseline and 50,000-epoch pruning workload."
        " Prune only the final ParameterNet kernel, increasing sparsity f"
        "rom 10% to 90% with the cubic schedule. Size comparisons use inf"
        "erence-only PyTorch state_dict and TorchScript artifacts, wrappe"
        "d in ZIP_DEFLATED as in the original helper; these backend forma"
        "ts differ from legacy HDF5/TFLite. Training checkpoints separate"
        "ly retain optimizer state. CPU packed int8 Linear inference is r"
        "un and validated after export; generated ShapeNet computation re"
        "mains floating point."
    ),
    omitted_markdown=(5, 21, 23, 35),
)

src8 = nb.read(original(8), as_version=4)
emit(
    8,
    {
        4: src8.cells[4].source + SURFACE_LAYOUT,
        6: src8.cells[6].source,
        8: SIREN
        + """nepoch,lr,batch_size=50000,1e-4,512
""",
        10: """coef_grad=1e-3
""",
        12: WAVE_DATA
        + """train_data_ng=tw.data
train_mean_ng,train_std_ng=tw.mean,tw.std
""",
        15: src8.cells[15].source,
        16: """train_dataset=loader(train_data[:,:2],train_data[:,2:5],batch_size)
""",
        20: (
            "from nif.layers import JacobianLayer\n"
            "\n"
            "def sobolev_loss(model,xb,yb,supervise_spatial=True):\n"
            "    # Full y plus derivatives ordered [t,x]; supervise ONLY spat"
            "ial derivative.\n"
            "    y,jac=JacobianLayer(model,[0],[0,1])(xb)\n"
            "    field_mse=(y-yb[:,:1]).square().mean()\n"
            "    temporal_mse=(jac[:,0,0]-yb[:,1]).square().mean()\n"
            "    spatial_mse=(jac[:,0,1]-yb[:,2]).square().mean()\n"
            "    loss=field_mse+coef_grad*spatial_mse if supervise_spatial el"
            "se field_mse\n"
            '    return loss,{"field_mse":field_mse,"temporal_derivative_mse"'
            ":temporal_mse,\n"
            '                 "spatial_derivative_mse":spatial_mse,"weighted_'
            'temporal_mse":coef_grad*temporal_mse,\n'
            '                 "weighted_spatial_mse":coef_grad*spatial_mse}\n'
            "\n"
            "model_opt=nif.NIFMultiScale(copy.deepcopy(cfg_shape_net),copy.de"
            'epcopy(cfg_parameter_net),"float32").to(device)\n'
            "optimizer=nif.optimizers.ReferenceAdam(model_opt.parameters(),lr"
            "=lr,eps=1e-7)\n"
            "history_sobolev=train(model_opt,train_dataset,optimizer,epochs(n"
            'epoch),"with_grad",loss_function=sobolev_loss)\n'
            'model_opt=save_and_check(model_opt,optimizer,"with_grad",train_d'
            "ata[:64,:2])\n"
        ),
        22: (
            "model_opt_ng=nif.NIFMultiScale(copy.deepcopy(cfg_shape_net),copy"
            '.deepcopy(cfg_parameter_net),"float32").to(device)\n'
            "optimizer_ng=nif.optimizers.ReferenceAdam(model_opt_ng.parameter"
            "s(),lr=lr,eps=1e-7)\n"
            "history_baseline=train(model_opt_ng,train_dataset,optimizer_ng,e"
            'pochs(nepoch),"without_grad",\n'
            "                       loss_function=lambda m,x,y:sobolev_loss(m"
            ",x,y,False))\n"
            'model_opt_ng=save_and_check(model_opt_ng,optimizer_ng,"without_g'
            'rad",train_data[:64,:2])\n'
        ),
        26: """def predict_derivatives(model,features):
    values=[]
    for xb in tensor(features).split(512):
        y,jac=JacobianLayer(model,[0],[0,1])(xb)
        values.append(torch.cat([y,jac[:,0,:]],dim=1).detach().cpu())
    return torch.cat(values).numpy()
pred_ng=predict_derivatives(model_opt_ng,train_data[:,:2])
pred_wg=predict_derivatives(model_opt,train_data[:,:2])
u_ng,u_wg=pred_ng[:,0].reshape(10,200),pred_wg[:,0].reshape(10,200)
""",
        27: """# Value and derivative diagnostics are displayed in consistent physical units.
physical_scale=np.array([tw.std[2],tw.std[2]/tw.std[0],tw.std[2]/tw.std[1]])
physical_mean=np.array([tw.mean[2],0,0])
truth_physical=train_data[:,2:5]*physical_scale+physical_mean
pred_ng_physical=pred_ng*physical_scale+physical_mean
pred_wg_physical=pred_wg*physical_scale+physical_mean
fig,axs=plt.subplots(3,2,figsize=(14,9))
for column,snapshot in enumerate((0,5)):
    take=slice(snapshot*200,(snapshot+1)*200)
    for component,label in enumerate(("u","du/dt","du/dx")):
        ax=axs[component,column]
        ax.plot(x,truth_physical[take,component],"k-",label="true")
        ax.plot(x,pred_ng_physical[take,component],"r--",label="without gradient")
        ax.plot(x,pred_wg_physical[take,component],"b-.",label="with gradient")
        ax.set(xlabel="physical x",ylabel=label,title=f"t={t[snapshot]}");ax.legend()
fig.tight_layout();fig.savefig(OUT / "physical_value_derivatives.png");plt.show();plt.close(fig)
""",
        29: (
            "dudt_ng,dudx_ng=pred_ng[:,1].reshape(10,200),pred_ng[:,2].reshap"
            "e(10,200)\n"
            "dudt_wg,dudx_wg=pred_wg[:,1].reshape(10,200),pred_wg[:,2].reshap"
            "e(10,200)\n"
            'print("physical derivative MSE [du/dt,du/dx], baseline:",np.mean'
            "((pred_ng_physical[:,1:]-truth_physical[:,1:])**2,axis=0))\n"
            'print("physical derivative MSE [du/dt,du/dx], Sobolev:",np.mean('
            "(pred_wg_physical[:,1:]-truth_physical[:,1:])**2,axis=0))\n"
        ),
        35: src8.cells[35].source,
        36: """# Correct inverse scaling: analytic u_test is physical, so predictions must be physical too.
u_ng_test=predict(model_opt_ng,test_data)[:,0].reshape(10,x_new_res)*tw.std[2]+tw.mean[2]
u_wg_test=predict(model_opt,test_data)[:,0].reshape(10,x_new_res)*tw.std[2]+tw.mean[2]
assert u_ng_test.shape==u_test.shape==u_wg_test.shape
""",
        38: (
            "for i in range(10):\n"
            "    fig,axs=plt.subplots(1,2,figsize=(16,3))\n"
            '    for ax,prediction,label in ((axs[0],u_ng_test,"without gradi'
            'ent"),(axs[1],u_wg_test,"with gradient")):\n'
            "        ax.plot(test_x,prediction[i],label=label);ax.plot(test_x"
            ',u_test[i],"k-",label="true")\n'
            '        ax.set(xlabel="physical x",ylabel="physical u",ylim=(-1.'
            '2,1.2),title=f"t={test_t[i]}");ax.legend()\n'
            '    fig.tight_layout();fig.savefig(OUT / f"superresolution_{i}.p'
            'ng");plt.show();plt.close(fig)\n'
            'write_metrics({"epochs_per_fit":epochs(nepoch),"fits":2,"spatial'
            '_loss_coefficient":coef_grad,\n'
            '    "baseline_physical_component_mse":np.mean((pred_ng_physical-'
            "truth_physical)**2,axis=0).tolist(),\n"
            '    "sobolev_physical_component_mse":np.mean((pred_wg_physical-t'
            "ruth_physical)**2,axis=0).tolist(),\n"
            '    "baseline_physical_superresolution_mse":float(np.mean((u_ng_'
            "test-u_test)**2)),\n"
            '    "sobolev_physical_superresolution_mse":float(np.mean((u_wg_t'
            "est-u_test)**2))})\n"
        ),
    },
    notes=(
        "Two 50,000-epoch fits: spatial Sobolev supervision (coefficient "
        "1e-3) versus value-only training. Temporal derivatives are measu"
        "red, not supervised. Predictions, derivatives and superresolutio"
        "n comparisons are all converted back to physical units."
    ),
    omitted_markdown=(17, 23),
)
