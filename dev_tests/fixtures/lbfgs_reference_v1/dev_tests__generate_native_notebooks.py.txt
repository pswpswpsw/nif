"""Build self-contained native tutorials while retaining the reference teaching narrative."""
from pathlib import Path
import copy
import nbformat as nb
ROOT = Path(__file__).resolve().parents[1]

SETUP = r'''# Run this cell first in a clean Colab Python 3.11–3.13 session.
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
'''
HELPERS = r'''# Explicit native loops replace compile/fit/callbacks. The loss reduction is per row,
# then the mean over the batch; area weights multiply each row's MSE (not weight-normalized).
def dense_tanh(inputs, outputs):
    # Preserve the original Dense default: Glorot uniform kernel and zero bias.
    layer = nn.Linear(inputs, outputs)
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return nn.Sequential(layer, nn.Tanh())

def epochs(full):
    return 2 if SMOKE else full

def tensor(array, on_device=device):
    return torch.as_tensor(array, dtype=torch.float32, device=on_device)

def loader(features, target, batch_size, weight=None):
    arrays = [tensor(features, "cpu"), tensor(target, "cpu")]
    if weight is not None:
        arrays.append(tensor(weight, "cpu").reshape(-1))
    return DataLoader(TensorDataset(*arrays), batch_size=batch_size, shuffle=True,
                      drop_last=False, pin_memory=device.type == "cuda")

def predict(model, features, batch_size=1024):
    model.eval()
    result = []
    with torch.no_grad():
        for xb in torch.as_tensor(features, dtype=torch.float32).split(batch_size):
            result.append(model(xb.to(next(model.parameters()).device)).detach().cpu())
    return torch.cat(result).numpy()

def centralize_gradients(model):
    # TF kernels are [input,output]; native Linear stores [output,input].
    # Centralize along the input axes, preserving the original mathematical operation.
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if parameter.grad is not None and parameter.ndim > 1:
                axes = tuple(range(1, parameter.ndim)) if isinstance(module, nn.Linear) else tuple(range(parameter.ndim-1))
                parameter.grad.sub_(parameter.grad.mean(dim=axes, keepdim=True))

def save_checkpoint(model, optimizer, path, epoch):
    payload = {"model_state": {k: v.detach().cpu() for k,v in model.state_dict().items()},
               "optimizer_state": optimizer.state_dict(), "epoch": epoch,
               "framework": "pytorch", "smoke": SMOKE}
    if hasattr(model, "cfg_shape_net"):
        payload.update(model_class=type(model).__name__, cfg_shape_net=model.cfg_shape_net,
                       cfg_parameter_net=model.cfg_parameter_net, mixed_policy="float32")
    torch.save(payload, path)

def save_and_check(model, optimizer, name, example, factory=None):
    path = OUT / (name + ".pt")
    save_checkpoint(model, optimizer, path, -1)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if factory is None:
        factory = lambda: type(model)(copy.deepcopy(model.cfg_shape_net),
                                       copy.deepcopy(model.cfg_parameter_net), "float32")
    restored = factory().to(device)
    restored.load_state_dict(checkpoint["model_state"])
    restored.eval(); model.eval()
    with torch.no_grad():
        expected, actual = model(tensor(example)), restored(tensor(example))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    print("Checkpoint predictions verified:", path)
    return restored

def train(model, batches, optimizer, nepoch, name, schedule=None,
          centralization=False, loss_function=None, before_epoch=None):
    history = []
    start = time.monotonic()
    for epoch in range(nepoch):
        model.train()
        if schedule is not None:
            for group in optimizer.param_groups:
                group["lr"] = schedule(epoch, group["lr"])
        if before_epoch is not None:
            before_epoch(epoch)
        summed, seen = {}, 0
        for batch in batches:
            xb, yb = (item.to(device) for item in batch[:2])
            optimizer.zero_grad(set_to_none=True)
            if loss_function is None:
                per_row = (model(xb) - yb).square().mean(dim=-1)
                if len(batch) == 3:
                    per_row = per_row * batch[2].to(device).reshape(-1)
                data_loss = per_row.mean()
                values = {"mse": data_loss}
            else:
                data_loss, values = loss_function(model, xb, yb)
            regularization = model.regularization_loss(xb) if hasattr(model, "regularization_loss") else xb.new_zeros(())
            loss = data_loss + regularization
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in {name}, epoch {epoch}")
            loss.backward()
            if centralization:
                centralize_gradients(model)
            optimizer.step()
            for key, value in {**values, "regularization": regularization, "loss": loss}.items():
                summed[key] = summed.get(key, 0.0) + float(value.detach()) * len(xb)
            seen += len(xb)
        row = {"epoch": epoch, "rows": seen, **{key: value / seen for key,value in summed.items()}}
        history.append(row)
        if epoch % 100 == 0 or epoch == nepoch - 1:
            print(name, row, "elapsed_seconds", round(time.monotonic()-start, 2))
            (OUT / (name + "_history.json")).write_text(json.dumps(history, indent=2))
        if epoch % 1000 == 0 or epoch == nepoch - 1:
            save_checkpoint(model, optimizer, OUT / (name + "_latest.pt"), epoch)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.semilogy([row["loss"] for row in history]); ax.set(xlabel="epoch", ylabel="loss", title=name)
    fig.tight_layout(); fig.savefig(OUT / (name + "_loss.png")); plt.show(); plt.close(fig)
    return history

def wave_figure(truth, prediction, name):
    fig, axs = plt.subplots(1,3,figsize=(15,3))
    for ax, values, title in zip(axs, (truth,prediction,prediction-truth), ("true","pred","error")):
        im=ax.contourf(tt,xx,values.reshape(10,200),levels=50,cmap="seismic")
        fig.colorbar(im,ax=ax); ax.set(xlabel="t",ylabel="x",title=title)
    fig.tight_layout(); fig.savefig(OUT / (name+"_reconstruction.png")); plt.show(); plt.close(fig)

def write_metrics(values):
    print(json.dumps(values, indent=2))
    (OUT / "metrics.json").write_text(json.dumps({"mode": "smoke" if SMOKE else "full", **values}, indent=2))
'''
SWISH = '''cfg_shape_net = {"connectivity": "full", "input_dim": 1, "output_dim": 1,
                 "units": 30, "nlayers": 2, "activation": "swish"}
cfg_parameter_net = {"input_dim": 1, "latent_dim": 1, "units": 30,
                     "nlayers": 2, "activation": "swish"}
'''
SIREN = '''cfg_shape_net = {"use_resblock": False, "connectivity": "full", "input_dim": 1,
                 "output_dim": 1, "units": 30, "nlayers": 2,
                 "weight_init_factor": 0.01, "omega_0": 30.0}
cfg_parameter_net = {"use_resblock": False, "input_dim": 1, "latent_dim": 1,
                     "units": 30, "nlayers": 2, "activation": "swish"}
'''
WAVE_DATA = '''from nif.demo import TravelingWaveHighFreq
tw = TravelingWaveHighFreq()
train_data = tw.data
train_dataset = loader(train_data[:, :2], train_data[:, 2:3], batch_size=512)
'''
SCHEDULE_MS = '''def scheduler(epoch, lr):
    if epoch < 1000: return lr
    if epoch < 2000: return 1e-4
    if epoch < 4000: return 5e-5
    return 1e-5
'''
SURFACE_LAYOUT = '''
# Keep all three axis labels inside the ordinary inline tight crop.
ax.set_box_aspect(None, zoom=0.75)
ax.xaxis.labelpad = ax.yaxis.labelpad = ax.zaxis.labelpad = 12
'''

def original(number):
    return next(p for p in (ROOT/'tutorial').glob(str(number)+'_*.ipynb') if not p.stem.endswith(('_colab','_pytorch')))

def markdown(source):
    # Retain scientific narrative, modernize framework-specific instructions.
    changes = {
       '2. output the real NIF model that needs to be trained, and compile this model\n  - note that here we take Adam optimizer, but it won\'t be used, it is just for compilation purpose.':'2. Reconstruct the native module from its saved configuration and load its state dictionary. Verify predictions after restoration; no compilation is needed.',
       'we first prune 10% then, train for 1000 epochs. We repeat 50 times of this process, each time we prune a bit more, until we hit the limit of 90% of included weights pruned.':'the cubic schedule starts at 10%, but its first mask refresh is before training step 4,000 (epoch 1,000), after 3,999 dense steps. It refreshes every 4,000 steps, for 50 refreshes, reaching 90% before step 200,000. We remask at every epoch end without an extra final refresh.',
       '`tfmot.sparsity.keras.PolynomialDecay`':'the native cubic magnitude-pruning schedule',
       '`TFPLBFGS`':'native `torch.optim.LBFGS`',
       'given the `n_features = 3` and `n_target = 3`':'given the `n_feature = 4` and `n_target = 3`',
       'So we have 7 `Example` in this demo.':'Each shard has one `Example` containing seven column-vector features.',
       'keras.Model':'torch.nn.Module', 'Keras Model':'PyTorch module',
       '`Keras`':'the explicit PyTorch loop', '`tf.data.Dataset`':'a `DataLoader`',
       '`model.fit`':'the explicit training loop', '.fit`':'the training loop`',
       'tensorflow_model_optimization':'nif.pruning.PolynomialMagnitudePruning',
       'https://www.tensorflow.org/model_optimization':'../docs/PYTORCH_API.md',
       'HDF5 format':'native state_dict format', 'tflite':'TorchScript', 'TFlite':'TorchScript',
       '`Callback`':'training-loop hook', '`callbacks`':'training-loop hooks',
       '`TensorBoard`':'loss logging', 'tensorboard':'loss logging',
    }
    for a,b in changes.items(): source=source.replace(a,b)
    return source

def emit(number, replacements, notes="", omitted_markdown=()):
    path=original(number); src=nb.read(path,as_version=4)
    cells=[nb.v4.new_markdown_cell(f"# Native PyTorch tutorial {number}\n\n"+notes+"\n\nRun all cells in order. Full scientific workloads are the default; `NIF_SMOKE=1` is a separate development run and does not establish full convergence. The original TensorFlow and `_colab` notebooks are preserved as legacy references."),
           nb.v4.new_code_cell(SETUP+f'\nOUT = Path(os.environ.get("NIF_OUTPUT_DIR", "pytorch_outputs")) / "tutorial{number}"\nOUT.mkdir(parents=True, exist_ok=True)\n'),nb.v4.new_code_cell(
               HELPERS.replace('before_epoch=None):', 'before_epoch=None, after_epoch=None):')
               .replace('        row = {"epoch": epoch',
                        '        if after_epoch is not None:\n            after_epoch(epoch)\n        row = {"epoch": epoch')
               .replace('v.detach().cpu() for k,v',
                        'v.detach().cpu() if isinstance(v, torch.Tensor) else v for k,v')
               if number == 7 else HELPERS)]
    for i,c in enumerate(src.cells):
        if c.cell_type=='markdown' and i not in omitted_markdown:
            cells.append(nb.v4.new_markdown_cell(markdown(c.source)))
        if i in replacements:
            s=replacements[i]
            if s.strip():cells.append(nb.v4.new_code_cell(s))
    out=nb.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.13"},"nif":{"framework":"pytorch","full_default":True,"reference":path.name}})
    dest=path.with_name(path.stem+'_pytorch.ipynb');nb.write(out,dest)
    print(dest)

src1=nb.read(original(1),as_version=4)
# Narrative about compilation/distribution is replaced by the explicit native loop notes.
emit(1,{
3:src1.cells[3].source+SURFACE_LAYOUT,
7:SWISH,
9:'''enable_multi_gpu = False
enable_mixed_precision = False
mixed_policy = "float32"
# Set device=NIF_DEVICE/cuda for GPU execution; this full workload uses FP32.
''',
11:'''nepoch = 5000
lr = 5e-3
batch_size = 512
checkpt_epoch, display_epoch, print_figure_epoch = 1000, 100, 100
''',
13:'''# alternatively, get the training dataset from nif.demo for this example
from nif.demo import TravelingWave
tw = TravelingWave()
train_data = tw.data
train_dataset = loader(train_data[:, :2], train_data[:, -1:], batch_size)
''',
19:'''def scheduler(epoch, lr):
    if epoch < 1000: return lr
    if epoch < 2000: return 1e-3
    if epoch < 4000: return 5e-4
    return 1e-4
''',
23:'''from nif.optimizers import AdaBeliefOptimizer
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
''',
27:'''model_p_to_lr = model_ori.model_p_to_lr()
print(model_p_to_lr)
p = tensor(train_data[:32, :1])
latent = model_p_to_lr(p)
# Plot every training time separately from the 32-row shared-subnet example.
plot_p = tensor(np.unique(train_data[:, 0]).reshape(-1, 1))
with torch.no_grad():
    plot_latent = model_p_to_lr(plot_p)
fig, ax=plt.subplots(); ax.plot(plot_p.cpu(),plot_latent.cpu(),"o"); ax.set(xlabel="normalized t",ylabel="latent")
fig.savefig(OUT / "latent.png"); plt.show(); plt.close(fig)
''',
29:'''torch.save(model_p_to_lr.state_dict(), OUT / "parameter_to_latent.pt")
''',
31:'''model_lr_to_w = model_ori.model_lr_to_w()
w = model_lr_to_w(latent)
torch.testing.assert_close(w, model_ori.model_p_to_w()(p))
print("latent-to-weights:", w.shape)
''',
33:'''model_x_to_u_given_w = model_ori.model_x_to_u_given_w()
torch.testing.assert_close(model_x_to_u_given_w([tensor(train_data[:32,1:2]), w]),
                           model_opt(tensor(train_data[:32,:2])))
print(model_x_to_u_given_w)
''',
40:'''new_model = save_and_check(model_opt, optimizer, "traveling_wave", train_data[:64,:2])
''',
44:'''data_feature = tensor(train_data[:, :2])
data_label = tensor(train_data[:, -1:])
''',
46:'''# Native torch.optim.LBFGS uses a closure: recompute loss and gradients at every line search.
lbfgs_history = []
rounds, max_iter = (1, 2) if SMOKE else (200, 1000)
# Public TFPLBFGS scientific controls, retained in full AND smoke modes.
tolerance = x_tolerance = f_relative_tolerance = 1e-15
num_correction_pairs = 20
max_line_search_iterations = 100
# Native convention (the same mapping as nif.optimizers.TFPLBFGS):
# PyTorch uses strong-Wolfe line search and a per-round max_eval budget of
# max_iter * max_line_search_iterations. Its internal trial accounting is not
# TFP's separate limit of 100 iterations per individual line search.
# tolerance_change applies to absolute step/objective changes (and the descent
# directional derivative); it cannot express TFP's independent relative-objective
# tolerance. Both original change thresholds are 1e-15 and map to that value.
# Thus the controls are retained, but solver trajectories/stopping can differ.
''',
48:'''for round_index in range(rounds):
    fine_tuner = torch.optim.LBFGS(new_model.parameters(), max_iter=max_iter,
                                 max_eval=max_iter * max_line_search_iterations,
                                 history_size=num_correction_pairs,
                                 tolerance_grad=tolerance,
                                 tolerance_change=max(x_tolerance, f_relative_tolerance),
                                 line_search_fn="strong_wolfe")
    def closure():
        fine_tuner.zero_grad(set_to_none=True)
        loss = (new_model(data_feature)-data_label).square().mean() + new_model.regularization_loss(data_feature)
        if not torch.isfinite(loss): raise FloatingPointError("non-finite LBFGS loss")
        loss.backward(); lbfgs_history.append(float(loss.detach()))
        return loss
    fine_tuner.step(closure)
new_model = save_and_check(new_model, fine_tuner, "fine_tuned", train_data[:64,:2])
fig,ax=plt.subplots(figsize=(8,2)); ax.semilogy(lbfgs_history); ax.set(xlabel="closure evaluation",ylabel="loss")
fig.tight_layout(); fig.savefig(OUT / "fine_tune_loss.png"); plt.show(); plt.close(fig)
''',
50:'''prediction = predict(new_model, train_data[:, :2]).ravel()
wave_figure(train_data[:,-1], prediction, "fine_tuned")
write_metrics({"adabelief_epochs": epochs(nepoch), "lbfgs_rounds": rounds, "lbfgs_max_iter": max_iter,
               "final_mse": float(np.mean((prediction-train_data[:,-1])**2))})
'''
},notes="5,000 AdaBelief epochs without gradient centralization by default, followed by 200 L-BFGS rounds of up to 1,000 iterations. Native build/model return the same trainable module. Logs and checkpoints are written by the visible training loop; GPU selection uses PyTorch devices.",omitted_markdown=(14,16,20,22,35,37,39))

src2=nb.read(original(2),as_version=4)
emit(2,{
3:src2.cells[3].source+SURFACE_LAYOUT,
5:SWISH+WAVE_DATA+'''nepoch, batch_size, lr = 5000, 512, 1e-3
def scheduler_swish(epoch, lr):
    if epoch < 1000: return lr
    if epoch < 2000: return 5e-4
    if epoch < 4000: return 1e-4
    return 5e-4
''',
7:'''model_swish = nif.NIF(cfg_shape_net,cfg_parameter_net,"float32").to(device)
optimizer = nif.optimizers.ReferenceAdam(model_swish.parameters(), lr=lr, eps=1e-7)
history_swish = train(model_swish, train_dataset, optimizer, epochs(nepoch), "swish", schedule=scheduler_swish)
model_swish = save_and_check(model_swish, optimizer, "swish", train_data[:64,:2])
''',
9:SIREN+SCHEDULE_MS+'''model_siren = nif.NIFMultiScale(cfg_shape_net,cfg_parameter_net,"float32").to(device)
optimizer = nif.optimizers.ReferenceAdam(model_siren.parameters(), lr=lr, eps=1e-7)
history_siren = train(model_siren, train_dataset, optimizer, epochs(nepoch), "siren", schedule=scheduler)
model_siren = save_and_check(model_siren, optimizer, "siren", train_data[:64,:2])
''',
11:'''pred_swish=predict(model_swish,train_data[:,:2]).ravel()
pred_siren=predict(model_siren,train_data[:,:2]).ravel()
wave_figure(train_data[:,2],pred_swish,"swish")
wave_figure(train_data[:,2],pred_siren,"siren")
write_metrics({"epochs_per_fit":epochs(nepoch),"fits":2,
    "swish_mse":float(np.mean((pred_swish-train_data[:,2])**2)),
    "siren_mse":float(np.mean((pred_siren-train_data[:,2])**2))})
'''
},notes="Two complete 5,000-epoch fits compare Swish and SIREN at the original width, depth, frequency, learning rates and schedules.")

src3=nb.read(original(3),as_version=4)
emit(3,{
3:'''from nif.demo import CylinderFlow
tmp = CylinderFlow()
train_data, sample_weight = tmp.data, tmp.sample_weight
assert len(train_data) == 191775
print(train_data.shape, sample_weight.shape, tmp.std)
''',
6:src3.cells[6].source,
7:'''nepoch, lr, batch_size = 600, 1e-3, 1024
# Only smoke mode selects fewer training rows; FULL uses all 191,775 AMR points each epoch.
training_rows = 4096 if SMOKE else len(train_data)
train_dataset = loader(train_data[:training_rows,:3], train_data[:training_rows,3:5],
                       batch_size, sample_weight[:training_rows])
def scheduler(epoch, lr):
    if epoch < 200: return lr
    if epoch < 400: return 5e-4
    if epoch < 600: return 2e-4
    return 3e-5
''',
10:'''from nif.optimizers import AdaBeliefOptimizer
model_ori = nif.NIFMultiScaleLastLayerParameterized(cfg_shape_net,cfg_parameter_net,"float32").to(device)
model = model_ori.build()
optimizer = AdaBeliefOptimizer(model.parameters(),lr=lr)
history = train(model,train_dataset,optimizer,epochs(nepoch),"cylinder",schedule=scheduler,centralization=True)
model = save_and_check(model,optimizer,"cylinder",train_data[:64,:3])
model_ori = model
''',
12:'''model_p_to_lr = model_ori.model_p_to_lr()
''',
13:'''t_arr = np.linspace(-1,1,100)
with torch.no_grad():
    lr_arr = model_p_to_lr(tensor(t_arr[:,None])).cpu().numpy()
''',
14:'''fig,ax=plt.subplots();ax.plot(t_arr,lr_arr);ax.set(xlabel="normalized t",ylabel="latent representation")
fig.savefig(OUT / "latent.png");plt.show();plt.close(fig)
# The spatial basis and time coefficients reconstruct the same learned linear representation.
phi = model.model_x_to_phi()(tensor(train_data[:32,1:3]))
print("spatial basis shape",tuple(phi.shape))
''',
16:'''input_pred = train_data[:,:3]
time_jump_index = np.r_[np.flatnonzero(np.diff(input_pred[:,0]) != 0)+1, len(input_pred)]
uv_pred = predict(model,input_pred)
''',
18:src3.cells[18].source+'''\nplt.savefig(OUT / "amr_mesh_counts.png");plt.show();plt.close()
''',
20:src3.cells[20].source+'''\nfig.savefig(OUT / "cylinder_reconstruction.png");plt.show();plt.close(fig)
per_row = np.mean((uv_pred-train_data[:,3:5])**2,axis=1)
write_metrics({"epochs":epochs(nepoch),"training_rows":training_rows,"all_rows":len(train_data),
               "weighted_mse":float(np.mean(per_row*np.asarray(sample_weight).reshape(-1))),
               "physical_u_mse":float(np.mean(((uv_pred[:,0]-train_data[:,3])*tmp.std[3])**2)),
               "physical_v_mse":float(np.mean(((uv_pred[:,1]-train_data[:,4])*tmp.std[4])**2))})
'''
},notes="600 epochs on all 191,775 cylinder-flow samples, area-weighted loss and actual gradient centralization. The input ordering is [t,x,y].")

emit(4,{
4:'''model = dense_tanh(4,5).to(device)
''',
6:'''from nif.layers import JacobianLayer, HessianLayer
''',
8:'''x_index, y_index = [0,1,2,3], [0,1,2,3,4]
''',
10:'''y_and_dydx_layer = JacobianLayer(model,y_index,x_index)
''',
12:'''x_data=torch.rand(10,4,device=device)
y_pred,dydx=y_and_dydx_layer(x_data)
''',
14:'''assert y_pred.shape == (10,5) and dydx.shape == (10,5,4)
print("f(x) shape",y_pred.shape,"df(x)/dx shape",dydx.shape)
''',
16:'''y_and_dydx_and_dy2dx2_layer=HessianLayer(model,y_index,x_index)
y_pred,dydx,dy2dx2=y_and_dydx_and_dy2dx2_layer(x_data)
''',
18:'''assert dy2dx2.shape == (10,5,4,4)
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
'''
},notes="Full outputs, selected Jacobians/Hessians, analytic numerical checks, zero higher derivatives and differentiable derivative penalties.")

emit(5,{
3:'''n_rows = 10000 if SMOKE else 1000000
points_per_shard = 1000 if SMOKE else 10000
data = np.random.uniform(0,1,(n_rows,7))
np.savez(OUT / "Big_npz_file.npz",data=data)
''',
6:'''from nif.data.tfr_dataset import TFRDataset
''',
8:'''fh = TFRDataset(n_feature=4,n_target=3)
''',
11:'''# Genuine TFRecord framing and tf.train.Example protobufs, without a TensorFlow runtime.
tfr_path = OUT / "TFR_dir"
fh.create_from_npz(num_pts_per_file=points_per_shard,npz_path=str(OUT / "Big_npz_file.npz"),
                   npz_key="data",tfr_path=str(tfr_path),prefix="case1")
shards=sorted(tfr_path.glob("*.tfrecord"))
assert len(shards) == (10 if SMOKE else 100)
print("TFRecord shards:",len(shards))
''',
16:'''model=dense_tanh(4,3).to(device)
from nif.optimizers import ReferenceAdam
optimizer=ReferenceAdam(model.parameters(),lr=1e-3,eps=1e-7)
''',
18:'''epoch=2
meta_dataset=fh.get_tfr_meta_dataset(tfr_path=str(tfr_path),epoch=epoch)
''',
20:'''batch_size=128
rows_seen, shard_visits, accumulated_loss = 0,0,0.0
stream_losses=[]
# Two complete passes through every shard, including every final batch/remainder.
for batch_file in meta_dataset:
    batch_dataset=fh.gen_dataset_from_batch_file(batch_file,batch_size)
    shard_rows,shard_loss=0,0.0
    for xb,yb in batch_dataset:
        xb,yb=xb.to(device),yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss=(model(xb)-yb).square().mean()
        if not torch.isfinite(loss): raise FloatingPointError("non-finite streamed loss")
        loss.backward();optimizer.step()
        shard_rows+=len(xb);shard_loss+=float(loss.detach())*len(xb)
    rows_seen+=shard_rows;shard_visits+=1;accumulated_loss+=shard_loss
    stream_losses.append(shard_loss/shard_rows)
assert rows_seen == 2*n_rows
assert shard_visits == 2*len(shards)
model=save_and_check(model,optimizer,"streamed",data[:64,:4],factory=lambda:dense_tanh(4,3))
fig,ax=plt.subplots();ax.plot(stream_losses);ax.set(xlabel="shard visit",ylabel="MSE",title="Two streamed passes")
fig.savefig(OUT / "streamed_loss.png");plt.show();plt.close(fig)
write_metrics({"generated_rows":n_rows,"shards":len(shards),"passes":epoch,
               "rows_seen":rows_seen,"shard_visits":shard_visits,"mse":accumulated_loss/rows_seen})
'''
},notes="Full mode generates one million rows, writes 100 original-format TFRecord shards and trains through two complete streamed passes. A native DataLoader batches each shard; global counters retain epoch/row information.")

src6=nb.read(original(6),as_version=4)
REG_TRAIN='''def run_regularization(kind=None, coefficient=0.0):
    config=copy.deepcopy(cfg_parameter_net)
    if kind is not None: config[kind+"_reg"]=coefficient
    name="no_reg" if kind is None else f"{kind}_{coefficient}"
    model=nif.NIFMultiScale(copy.deepcopy(cfg_shape_net),config,"float32").to(device)
    optimizer=nif.optimizers.ReferenceAdam(model.parameters(),lr=1e-3,eps=1e-7)
    history=train(model,train_dataset,optimizer,epochs(nepoch),name)
    model=save_and_check(model,optimizer,name,train_data[:64,:2])
    train_pred=predict(model,train_data[:,:2]).ravel()
    test_pred=predict(model,test_data).reshape(test_tt.shape)
    fig,axs=plt.subplots(1,3,figsize=(15,3.5),layout="constrained")
    # A common normalized-u scale makes nearly constant predictions look constant.
    levels=np.linspace(min(train_data[:,2].min(),train_pred.min(),test_pred.min()),
                       max(train_data[:,2].max(),train_pred.max(),test_pred.max()),51)
    for ax,values,title in zip(axs[:2],(train_data[:,2],train_pred),("true","pred")):
        im=ax.contourf(tt,xx,values.reshape(10,200),levels=levels,cmap="seismic");ax.set_title(title)
    im=axs[2].contourf(test_tt,test_xx,test_pred,levels=levels,cmap="seismic");axs[2].set_title("super-resolution")
    fig.colorbar(im,ax=axs,label="normalized u")
    fig.suptitle(name);fig.savefig(OUT/(name+"_superresolution.png"));plt.show();plt.close(fig)
    physical_prediction=test_pred*tw.std[2]+tw.mean[2]
    result={"kind":kind,"coefficient":coefficient,"epochs":epochs(nepoch),
            "training_mse":float(np.mean((train_pred-train_data[:,2])**2)),
            "physical_superresolution_mse":float(np.mean((physical_prediction-u_test)**2)),
            "regularization_loss":float(model.regularization_loss(tensor(train_data[:64,:2])).detach())}
    results.append(result)
    return model
'''
emit(6,{
4:src6.cells[4].source+SURFACE_LAYOUT,
5:'''# prepare testing data: finer normalized coordinates; ALWAYS [parameter, coordinate] = [t,x].
resolution=41 if SMOKE else 371
test_x=np.linspace(-1,1,resolution)
test_t=np.linspace(-1,1,resolution)
test_xx,test_tt=np.meshgrid(test_x,test_t)
test_data=np.column_stack([test_tt.ravel(),test_xx.ravel()])
''',
7:SIREN+'''nepoch,lr,batch_size = 3000,1e-3,512
results=[]
''',
10:WAVE_DATA+'''test_x_phys=test_xx*tw.std[1]+tw.mean[1]
test_t_phys=test_tt*tw.std[0]+tw.mean[0]
u_test=np.exp(-1000*(test_x_phys-x0-c*test_t_phys)**2)*np.sin(omega*(test_x_phys-x0-c*test_t_phys))
'''+REG_TRAIN+'''baseline=run_regularization()
''',
14:'''model_l2_1e5=run_regularization("l2",1e-5)
''',
17:'''model_l2_1e4=run_regularization("l2",1e-4)
''',
20:'''model_l2_1e3=run_regularization("l2",1e-3)
''',
23:'''model_l2_1e2=run_regularization("l2",1e-2)
''',
28:'''model_l1_1e5=run_regularization("l1",1e-5)
''',
30:'''model_l1_1e4=run_regularization("l1",1e-4)
''',
32:'''model_l1_1e3=run_regularization("l1",1e-3)
''',
34:'''model_l1_1e2=run_regularization("l1",1e-2)
''',
35:'''assert len(results)==9
write_metrics({"fits":9,"epochs_per_fit":epochs(nepoch),"resolution":resolution,"results":results})
'''
},notes="Nine 3,000-epoch fits: baseline, four L2 coefficients and four L1 coefficients. Regularization is an explicit differentiable loss, independent of optimizer weight decay. The reference superresolution input ordering is corrected to [t,x].")

# Pruning and quantization are kept as separate inspectable stages.
emit(7,{
4:SIREN+SCHEDULE_MS+'''nepoch,lr,batch_size = 5000,1e-3,512
''',
8:WAVE_DATA+'''# Shuffling is handled by DataLoader, keeping the source grid available for diagnostics.
model_ori=nif.NIFMultiScale(cfg_shape_net,cfg_parameter_net,"float32").to(device)
model=model_ori.build()
optimizer=nif.optimizers.ReferenceAdam(model.parameters(),lr=lr,eps=1e-7)
history_baseline=train(model,train_dataset,optimizer,epochs(nepoch),"baseline",schedule=scheduler)
''',
10:'''t=np.linspace(-1,1,10,endpoint=True)
x=np.linspace(-1,1,200,endpoint=True)
tt,xx=np.meshgrid(t,x)
input_feature=np.column_stack([tt.ravel(),xx.ravel()])
model=save_and_check(model,optimizer,"original",input_feature[:64])
y_pred_unpruned=predict(model,input_feature).reshape(200,10)
''',
12:'''original_file=OUT / "original.pt"
''',
14:'''model_for_pruning=copy.deepcopy(model)
pruning_epochs=50000
prune_epoch=1000
# PolynomialDecay's cubic schedule: begin at 10%, end at 90% of the selected kernel.
# Full mode refreshes before steps 4,000 through 200,000 (50 updates).
# Steps 1 through 3,999 stay dense; no extra refresh follows training.
''',
16:'''print(model_for_pruning)
''',
18:'''from nif.pruning import PolynomialMagnitudePruning
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
''',
22:'''optimizer_prune=nif.optimizers.ReferenceAdam(model_for_pruning.parameters(),lr=1e-4,eps=1e-7)
''',
26:'''history_prune=train(model_for_pruning,train_dataset,optimizer_prune,run_pruning_epochs,
                    "pruning",after_epoch=lambda epoch:pruning_controller.on_epoch_end())
# Epoch-end masking precedes checkpoints, without an extra schedule refresh.
assert pruning_controller.step==run_pruning_epochs*steps_per_epoch
assert pruning_controller.refresh_count==(2 if SMOKE else 50)
''',
28:'''fig,ax=plt.subplots(figsize=(12,4));ax.semilogy([row["loss"] for row in history_prune])
for epoch_index in range((1 if SMOKE else prune_epoch)-1,run_pruning_epochs,1 if SMOKE else prune_epoch):ax.axvline(epoch_index,color="r",alpha=.2)
ax.set(xlabel="epoch",ylabel="training loss",title="Magnitude pruning: 10% to 90%")
fig.savefig(OUT / "pruned_learning_curve.png");plt.show();plt.close(fig)
''',
30:'''zero_count=int((hyper_siren_layer.w==0).sum())
weight_count=hyper_siren_layer.w.numel()
sparsity=zero_count/weight_count
assert abs(sparsity-0.90)<=1/weight_count
print({"zero_weights":zero_count,"total_weights":weight_count,"sparsity":sparsity})
''',
32:'''y_pred=predict(model_for_pruning,input_feature).reshape(200,10)
''',
34:'''# Strip the pruning controls: the native .pt contains the actual sparse tensor.
pruning_controller.strip()
model_for_export=save_and_check(model_for_pruning,optimizer_prune,"stripped",input_feature[:64])
pruned_file=OUT / "stripped.pt"
''',
36:'''model_for_export=model_for_export.cpu().eval()
example=torch.tensor(input_feature[:32],dtype=torch.float32)
traced=torch.jit.trace(model_for_export,example)
exported_file=OUT / "compressed.torchscript.pt"
torch.jit.save(traced,exported_file)
exported=torch.jit.load(exported_file,map_location="cpu")
with torch.no_grad():
    torch.testing.assert_close(exported(example),model_for_export(example))
    torch.testing.assert_close(exported(torch.tensor(input_feature[:17],dtype=torch.float32)),
                               model_for_export(torch.tensor(input_feature[:17],dtype=torch.float32)))
''',
38:'''def get_gzipped_model_size(path):
    import gzip
    return len(gzip.compress(Path(path).read_bytes()))
''',
40:'''# A real native dynamic int8 kernel executes the generated-weight projection.
# The nonlinear SIREN and per-example generated ShapeNet weights stay floating point.
# This is weight/activation dynamic quantization, not a float model saved under an int8 filename.
float_quant_model=copy.deepcopy(model_for_export)
for name,layer in list(float_quant_model.named_modules()):
    if isinstance(layer,HyperLinearForSIREN):
        linear=nn.Linear(layer.w.shape[0],layer.w.shape[1])
        with torch.no_grad():
            linear.weight.copy_(layer.w.T);linear.bias.copy_(layer.b)
        parent_name,_,child_name=name.rpartition(".")
        parent=float_quant_model.get_submodule(parent_name) if parent_name else float_quant_model
        setattr(parent,child_name,linear)
with torch.no_grad():torch.testing.assert_close(float_quant_model(example),model_for_export(example))
quantized_model=torch.ao.quantization.quantize_dynamic(float_quant_model,{nn.Linear},dtype=torch.qint8,inplace=False).eval()
quantized_linears=[layer for layer in quantized_model.modules()
                  if isinstance(layer,torch.ao.nn.quantized.dynamic.Linear)]
assert quantized_linears and all(layer.weight().dtype==torch.qint8 for layer in quantized_linears)
quantized_file=OUT / "quantized_and_pruned.torchscript.pt"
quantized_script=torch.jit.trace(quantized_model,example)
torch.jit.save(quantized_script,quantized_file)
restored_quantized=torch.jit.load(quantized_file,map_location="cpu")
with torch.no_grad():
    torch.testing.assert_close(restored_quantized(example),quantized_model(example),rtol=0,atol=0)
print("Native packed int8 Linear kernels:",len(quantized_linears))
''',
42:'''with torch.no_grad():
    # Full batch predictions are validated after restoring the exported artifact.
    quant_input=torch.tensor(input_feature,dtype=torch.float32)
    y_pred_quant=restored_quantized(quant_input).cpu().numpy().reshape(200,10)
    torch.testing.assert_close(restored_quantized(quant_input),quantized_model(quant_input),rtol=0,atol=0)
assert np.isfinite(y_pred_quant).all()
''',
46:'''fig,axs=plt.subplots(3,2,figsize=(12,12))
for column,compressed,label in ((0,y_pred,"pruned 90%"),(1,y_pred_quant,"pruned 90% + int8")):
    for row,values,title in ((0,y_pred_unpruned,"unpruned"),(1,compressed,label),(2,compressed-y_pred_unpruned,"difference")):
        im=axs[row,column].contourf(tt,xx,values,levels=50,cmap="seismic");fig.colorbar(im,ax=axs[row,column]);axs[row,column].set_title(title)
fig.tight_layout();fig.savefig(OUT / "compression_comparison.png");plt.show();plt.close(fig)
''',
47:'''denominator=float(np.sum(y_pred_unpruned**2))
write_metrics({"baseline_epochs":epochs(nepoch),"pruning_epochs":run_pruning_epochs,
    "sparsity":sparsity,"zero_weights":zero_count,"total_weights":weight_count,
    "quantized_linear_count":len(quantized_linears),
    "relative_mse_prune":float(np.sum((y_pred-y_pred_unpruned)**2)/denominator),
    "relative_mse_prune_quant":float(np.sum((y_pred_quant-y_pred_unpruned)**2)/denominator),
    "quantization_mse_vs_pruned":float(np.mean((y_pred_quant-y_pred)**2)),
    "gzip_bytes":{p.name:get_gzipped_model_size(p) for p in (original_file,pruned_file,exported_file,quantized_file)}})
'''
},notes="Original 5,000-epoch baseline and 50,000-epoch pruning workload. Prune only the final ParameterNet kernel, increasing sparsity from 10% to 90% with the cubic schedule. Native checkpoints and TorchScript replace legacy HDF5/TFLite. CPU packed int8 Linear inference is run and validated after export; generated ShapeNet computation remains floating point.",omitted_markdown=(5,21,23,35))

src8=nb.read(original(8),as_version=4)
emit(8,{
4:src8.cells[4].source+SURFACE_LAYOUT,
6:src8.cells[6].source,
8:SIREN+'''nepoch,lr,batch_size=50000,1e-4,512
''',
10:'''coef_grad=1e-3
''',
12:WAVE_DATA+'''train_data_ng=tw.data
train_mean_ng,train_std_ng=tw.mean,tw.std
''',
15:src8.cells[15].source,
16:'''train_dataset=loader(train_data[:,:2],train_data[:,2:5],batch_size)
''',
20:'''from nif.layers import JacobianLayer

def sobolev_loss(model,xb,yb,supervise_spatial=True):
    # Full y plus derivatives ordered [t,x]; supervise ONLY spatial derivative.
    y,jac=JacobianLayer(model,[0],[0,1])(xb)
    field_mse=(y-yb[:,:1]).square().mean()
    temporal_mse=(jac[:,0,0]-yb[:,1]).square().mean()
    spatial_mse=(jac[:,0,1]-yb[:,2]).square().mean()
    loss=field_mse+coef_grad*spatial_mse if supervise_spatial else field_mse
    return loss,{"field_mse":field_mse,"temporal_derivative_mse":temporal_mse,
                 "spatial_derivative_mse":spatial_mse,"weighted_temporal_mse":coef_grad*temporal_mse,
                 "weighted_spatial_mse":coef_grad*spatial_mse}

model_opt=nif.NIFMultiScale(copy.deepcopy(cfg_shape_net),copy.deepcopy(cfg_parameter_net),"float32").to(device)
optimizer=nif.optimizers.ReferenceAdam(model_opt.parameters(),lr=lr,eps=1e-7)
history_sobolev=train(model_opt,train_dataset,optimizer,epochs(nepoch),"with_grad",loss_function=sobolev_loss)
model_opt=save_and_check(model_opt,optimizer,"with_grad",train_data[:64,:2])
''',
22:'''model_opt_ng=nif.NIFMultiScale(copy.deepcopy(cfg_shape_net),copy.deepcopy(cfg_parameter_net),"float32").to(device)
optimizer_ng=nif.optimizers.ReferenceAdam(model_opt_ng.parameters(),lr=lr,eps=1e-7)
history_baseline=train(model_opt_ng,train_dataset,optimizer_ng,epochs(nepoch),"without_grad",
                       loss_function=lambda m,x,y:sobolev_loss(m,x,y,False))
model_opt_ng=save_and_check(model_opt_ng,optimizer_ng,"without_grad",train_data[:64,:2])
''',
26:'''def predict_derivatives(model,features):
    values=[]
    for xb in tensor(features).split(512):
        y,jac=JacobianLayer(model,[0],[0,1])(xb)
        values.append(torch.cat([y,jac[:,0,:]],dim=1).detach().cpu())
    return torch.cat(values).numpy()
pred_ng=predict_derivatives(model_opt_ng,train_data[:,:2])
pred_wg=predict_derivatives(model_opt,train_data[:,:2])
u_ng,u_wg=pred_ng[:,0].reshape(10,200),pred_wg[:,0].reshape(10,200)
''',
27:'''# Value and derivative diagnostics are displayed in consistent physical units.
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
''',
29:'''dudt_ng,dudx_ng=pred_ng[:,1].reshape(10,200),pred_ng[:,2].reshape(10,200)
dudt_wg,dudx_wg=pred_wg[:,1].reshape(10,200),pred_wg[:,2].reshape(10,200)
print("physical derivative MSE [du/dt,du/dx], baseline:",np.mean((pred_ng_physical[:,1:]-truth_physical[:,1:])**2,axis=0))
print("physical derivative MSE [du/dt,du/dx], Sobolev:",np.mean((pred_wg_physical[:,1:]-truth_physical[:,1:])**2,axis=0))
''',
35:src8.cells[35].source,
36:'''# Correct inverse scaling: analytic u_test is physical, so predictions must be physical too.
u_ng_test=predict(model_opt_ng,test_data)[:,0].reshape(10,x_new_res)*tw.std[2]+tw.mean[2]
u_wg_test=predict(model_opt,test_data)[:,0].reshape(10,x_new_res)*tw.std[2]+tw.mean[2]
assert u_ng_test.shape==u_test.shape==u_wg_test.shape
''',
38:'''for i in range(10):
    fig,axs=plt.subplots(1,2,figsize=(16,3))
    for ax,prediction,label in ((axs[0],u_ng_test,"without gradient"),(axs[1],u_wg_test,"with gradient")):
        ax.plot(test_x,prediction[i],label=label);ax.plot(test_x,u_test[i],"k-",label="true")
        ax.set(xlabel="physical x",ylabel="physical u",ylim=(-1.2,1.2),title=f"t={test_t[i]}");ax.legend()
    fig.tight_layout();fig.savefig(OUT / f"superresolution_{i}.png");plt.show();plt.close(fig)
write_metrics({"epochs_per_fit":epochs(nepoch),"fits":2,"spatial_loss_coefficient":coef_grad,
    "baseline_physical_component_mse":np.mean((pred_ng_physical-truth_physical)**2,axis=0).tolist(),
    "sobolev_physical_component_mse":np.mean((pred_wg_physical-truth_physical)**2,axis=0).tolist(),
    "baseline_physical_superresolution_mse":float(np.mean((u_ng_test-u_test)**2)),
    "sobolev_physical_superresolution_mse":float(np.mean((u_wg_test-u_test)**2))})
'''
},notes="Two 50,000-epoch fits: spatial Sobolev supervision (coefficient 1e-3) versus value-only training. Temporal derivatives are measured, not supervised. Predictions, derivatives and superresolution comparisons are all converted back to physical units.",omitted_markdown=(17,23))
