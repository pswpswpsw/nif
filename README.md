# Neural Implicit Flow (NIF): mesh-agnostic dimensionality reduction
![example workflow](https://github.com/pswpswpsw/nif/workflows/Tests/badge.svg)
[![License](https://img.shields.io/github/license/pswpswpsw/nif)](https://github.com/pswpswpsw/nif/blob/main/LICENSE)[![DOI](https://zenodo.org/badge/460537754.svg)](https://zenodo.org/badge/latestdoi/460537754)

<p align="center">
  <img src="./misc/myimage.gif" alt="Neural Implicit Flow visualization" />
</p>

- NIF is a mesh-agnostic dimensionality reduction paradigm for parametric spatiotemporal fields. Dimensionality reduction is often the first step in reduced-order modeling of large-scale dynamics.
- NIF represents fields at arbitrary spatial coordinates, including data from adaptive mesh refinement, without requiring interpolation onto a uniform grid. Details can be found in our [paper](https://arxiv.org/pdf/2204.03216.pdf).
- NIF is implemented in **PyTorch**. A ParameterNet maps time or physical parameters to the weights of a coordinate-dependent ShapeNet.

## Highlights

<p align="center">
  <img src="./misc/compare.gif" alt="Comparison of field representations" />
</p>

### Installation and requirements

Python 3.10+ and PyTorch 2.6+ are required; CUDA is optional. Install from a local checkout:

```bash
python -m pip install -e .
# Optional: support TensorBoard logging in the standalone training example.
python -m pip install -e '.[tensorboard]'
```

Choose a suitable PyTorch CUDA wheel for your machine when training on GPU. See [native API and migration notes](docs/PYTORCH_API.md) for optimizer conventions and data formats. The [validation report](docs/PYTORCH_VALIDATION.md) and [comparison records](docs/PYTORCH_TENSORFLOW_PARITY.md) document the measured results and limitations.

### Configuring and training NIF

The example below fits a one-dimensional traveling wave. The configuration dictionaries set the input/output dimensions, network widths, hidden layers, and activations. Input columns are ordered as `[parameters, coordinates]`.

```python
import torch
from nif import NIF
from nif.demo import TravelingWave
from nif.optimizers import AdaBeliefOptimizer

shape = dict(input_dim=1, output_dim=1, units=20, nlayers=2, activation="tanh")
parameter = dict(input_dim=1, latent_dim=2, units=10, nlayers=2, activation="tanh")
model = NIF(shape, parameter).to("cuda" if torch.cuda.is_available() else "cpu")
device = next(model.parameters()).device

data = torch.as_tensor(TravelingWave().data, dtype=torch.float32, device=device)
x, target = data[:, :2], data[:, 2:3]
optimizer = AdaBeliefOptimizer(model.parameters(), lr=1e-3)
for epoch in range(100):
    optimizer.zero_grad(set_to_none=True)
    loss = (model(x) - target).square().mean() + model.regularization_loss(x)
    loss.backward()
    optimizer.step()
print(float(loss.detach()))
```

The default model uses float32. On supported hardware, `mixed_policy="mixed_bfloat16"` or `mixed_policy="mixed_float16"` enables autocast while retaining float32 trainable parameters. See the API notes for details.

### Distributed learning

The standalone [training example](examples/train_pytorch.py) uses PyTorch distributed data parallelism with one process per GPU. For example, on Linux or WSL with two CUDA GPUs and an NCCL-enabled PyTorch build:

```bash
torchrun --standalone --nproc-per-node=2 examples/train_pytorch.py --device cuda --epochs 20 --batch-size 256 --output-dir pytorch_outputs/wave-ddp --tensorboard
```

`--batch-size` is per process. The dataset size must be divisible by the number of processes so that no samples are padded or dropped. The example also runs on one device with ordinary `python`; it trains on the same traveling-wave data used above. The L-BFGS example below uses one device. See the [training example validation and requirements](docs/PYTORCH_TRAINING_EXAMPLE.md) for tested platforms and limits.

### Fine-tuning with L-BFGS

After first-order training, refine the same model using a fixed dataset and a closure. This example continues from the model, `x`, and `target` above.

```python
from nif.optimizers import ReferenceLBFGS

fine_tuner = ReferenceLBFGS(model.parameters(), max_iter=1000)
def closure():
    fine_tuner.zero_grad(set_to_none=True)
    loss = (model(x) - target).square().mean() + model.regularization_loss(x)
    loss.backward()
    return loss

fine_tuner.step(closure)
print("Final loss:", float(fine_tuner.result.objective_value))
print("Iterations:", int(fine_tuner.result.num_iterations))
```

### Training schedules and monitoring

The [training example](examples/train_pytorch.py) records training loss for TensorBoard and saves checkpoints. Run it on one CPU with:

```bash
python examples/train_pytorch.py --device cpu --epochs 20 --batch-size 256 --output-dir pytorch_outputs/wave --tensorboard
tensorboard --logdir pytorch_outputs/wave
```

The example uses a learning-rate schedule and writes scalar logs on the primary process. Reported training and evaluation MSE use the normalized data. Install the optional monitoring dependency as described in the installation section above.

### Extracting subnetworks

Extract the parameter-to-latent, latent-to-weight, and coordinate-to-output subnetworks. These views share the model's tensors and retain their gradient connections.

```python
parameter_inputs, coordinates = x[:, :1], x[:, 1:2]
latent = model.model_p_to_lr()(parameter_inputs)
weights = model.model_lr_to_w()(latent)
u_pred = model.model_x_to_u_given_w()(coordinates, weights)
torch.testing.assert_close(u_pred, model(x))
```

### Input-output Jacobians and Hessians

```python
from nif.layers import JacobianLayer, HessianLayer

output, jacobian = JacobianLayer(model, y_index=[0], x_index=[1])(x)
output, jacobian, hessian = HessianLayer(model, y_index=[0], x_index=[1])(x)
```

The layers return the complete output and selected derivatives, and retain parameter gradients for derivative-based training. See [Tutorial 4](tutorial/4_get_gradients_by_wrapping_model_with_layer_pytorch.ipynb).

### Data normalization

`PointWiseData` keeps parameter, coordinate, and target columns in order and supports min-max or standard normalization. For example, with one parameter, one coordinate, and one output:

```python
from nif.data import PointWiseData

raw_data = TravelingWave().data_raw
normalized, mean, scale = PointWiseData.minmax_normalize(
    raw_data=raw_data, n_para=1, n_x=1, n_target=1
)
```

### Large-scale record data

`TFRDataset` converts an NPZ array into record shards and reads them as PyTorch `DataLoader` batches. This small, self-contained IO example uses the `model`, `optimizer`, and `device` defined above:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from nif.data.tfr_dataset import TFRDataset

with TemporaryDirectory() as directory:
    root = Path(directory)
    np.savez(root / "wave.npz", data=TravelingWave().data)
    records = TFRDataset(n_feature=2, n_target=1)
    records.create_from_npz(
        num_pts_per_file=1000, npz_path=root / "wave.npz", npz_key="data",
        tfr_path=root / "record_shards", prefix="wave"
    )
    for shard in records.get_tfr_meta_dataset(root / "record_shards", epoch=1):
        for features, targets in records.gen_dataset_from_batch_file(shard, batch_size=128):
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(features) - targets).square().mean()
            loss = loss + model.regularization_loss(features)
            loss.backward()
            optimizer.step()
```

[Tutorial 5](tutorial/5_large_scale_training_on_tensorflow_record_data_pytorch.ipynb) demonstrates the same workflow on a one-million-row dataset.

### Save and load models

```python
torch.save(model.state_dict(), "model.pt")
restored = NIF(shape, parameter).to(device)
restored.load_state_dict(torch.load("model.pt", map_location=device, weights_only=True))
torch.testing.assert_close(restored(x), model(x))
model.save_config("config.json")
```

### Network pruning and quantization

[Tutorial 7](tutorial/7_model_pruning_and_quantization_pytorch.ipynb) demonstrates magnitude pruning of the selected ParameterNet kernel, dynamic int8 CPU inference, and TorchScript export.

## Google Colab Tutorial

The notebooks below run the native PyTorch examples. Their scientific workloads are the default; set `NIF_SMOKE=1` before running only for a short development check.

1. **Hello world! A simple fitting on a 1D traveling wave** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/1_simple_1d_wave_pytorch.ipynb)
    - Learn the `nif.NIF` model, its configuration, and training loop.
    - Save and restore checkpoints; refine the fit with L-BFGS.

2. **Tackling multi-scale data** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/2_multi_scale_NIF_pytorch.ipynb)
    - Use `nif.NIFMultiScale` and compare fits to a high-frequency wave.

3. **Learning a linear representation** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/3_multi_scale_linear_NIF_pytorch.ipynb)
    - Use `nif.NIFMultiScaleLastLayerParameterized` on a cylinder-flow dataset from an adaptive mesh.

4. **Getting input-output derivatives** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/4_get_gradients_by_wrapping_model_with_layer_pytorch.ipynb)
    - Use `nif.layers.JacobianLayer` and `nif.layers.HessianLayer`.

5. **Large-scale data training** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/5_large_scale_training_on_tensorflow_record_data_pytorch.ipynb)
    - Create and stream record shards using `nif.data.tfr_dataset.TFRDataset`.
    - Train on batches from a one-million-row example dataset.

6. **Revisit multi-scale data with regularization** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/6_revisit_multi_scale_NIF_with_L1_L2_regularization_pytorch.ipynb)
    - Compare L1 and L2 regularization of ParameterNet weights and biases.
    - Inspect training fits and spurious signals on a refined space-time grid.

7. **NIF compression** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/7_model_pruning_and_quantization_pytorch.ipynb)
    - Explore magnitude pruning, dynamic int8 quantization, and exported inference.

8. **Revisit multi-scale data with Sobolev training** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/8_revisit_multi_scale_NIF_with_sobolov_training_pytorch.ipynb)
    - Add spatial-derivative supervision with `nif.layers.JacobianLayer` and compare field and derivative errors.
    - Inspect how derivative information affects the refined spatial prediction.

## Issues, bugs, requests, ideas

Use the [issues](https://github.com/pswpswpsw/nif/issues) tracker to report bugs.

## How to cite

If you find NIF is helpful, please cite our JMLR [paper](https://www.jmlr.org/papers/volume24/22-0365/22-0365.pdf):

```bibtex
@article{JMLR:v24:22-0365,
  author  = {Shaowu Pan and Steven L. Brunton and J. Nathan Kutz},
  title   = {Neural Implicit Flow: a mesh-agnostic dimensionality reduction paradigm of spatio-temporal data},
  journal = {Journal of Machine Learning Research},
  year    = {2023},
  volume  = {24},
  number  = {41},
  pages   = {1--60},
  url     = {http://jmlr.org/papers/v24/22-0365.html}
}
```

## Contributors

- [Shaowu Pan](http://www.shaowupan.com)
- [Yaser Afshar](https://github.com/yafshar)

## License

[LGPL-2.1 License](https://github.com/pswpswpsw/nif/blob/main/LICENSE)
