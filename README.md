# Neural Implicit Flow — native PyTorch

NIF learns mesh-agnostic representations of spatiotemporal fields. A parameter
network maps time and operating conditions to a coordinate-dependent shape
network. This branch implements the complete package in PyTorch, including
SIREN multiscale fields, linear spatial bases, differentiable derivatives,
streamed TFRecord data, and native optimizers.

Independent acceptance passed **253 checks in the CUDA-enabled environment and
all eight full tutorial workloads**. These establish package behavior and full
execution; matched TensorFlow/PyTorch training validation is in progress. See
the [validation report](docs/PYTORCH_VALIDATION.md) and
[controlled comparison protocol](docs/PYTORCH_TENSORFLOW_PARITY.md).

```bash
python -m pip install -e .
```

Python 3.10+ and PyTorch 2.6+ are required; CUDA is optional. The development
container uses Python 3.13 and PyTorch 2.14.1 CPU. There is no TensorFlow,
Keras, TensorFlow Probability, or TFMOT runtime dependency. Choose an appropriate
PyTorch CUDA wheel for your machine when training on GPU.

## Training

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

Input order is always `[parameters, coordinates]`. Models are `nn.Module`s:
`.build()` and `.model()` return the same module and parameters. Use explicit
training loops and `state_dict()` checkpoints. There is no `compile`/`fit` API.

```python
torch.save(model.state_dict(), "model.pt")
restored = NIF(shape, parameter).to(device)
restored.load_state_dict(torch.load("model.pt", map_location=device, weights_only=True))
torch.testing.assert_close(restored(x), model(x))
model.save_config("config.json")
```

`NIFMultiScale` uses a sine shape network, with optional residual blocks.
`NIFMultiScaleLastLayerParameterized` learns spatial bases and time-dependent
coefficients. Extracted `model_p_to_w`, `model_p_to_lr`, `model_lr_to_w`,
`model_x_to_u_given_w`, and `model_x_to_phi` modules share the original tensors.
They retain autograd connectivity and can be moved with standard `.to()` calls.
For the last-layer model, latent coefficients are weights, so `model_lr_to_w()`
is the identity.

```python
from nif.layers import JacobianLayer, HessianLayer
output, jacobian = JacobianLayer(model, y_index=[0], x_index=[1])(x)
output, jacobian, hessian = HessianLayer(model, y_index=[0], x_index=[1])(x)
```

Derivative wrappers return the full output and selected derivatives, with batch,
selected-output, and selected-input axes. Derivatives remain differentiable with
respect to model parameters for Sobolev training. `regularization_loss(inputs)`
explicitly computes current-batch penalties; it never reads a stale forward cache.
See [native API and migration notes](docs/PYTORCH_API.md) for exact reductions,
optimizer conventions, data formats, and intentional baseline corrections.

## Eight native tutorials

Full scientific workloads are the default. Set `NIF_SMOKE=1` **before executing**
a notebook only for a short development check; smoke runs do not establish
convergence or replace the full workloads. Each native notebook provides setup,
explicit loops, figures, metrics, and portable artifacts.

1. [Traveling wave and L-BFGS](tutorial/1_simple_1d_wave_pytorch.ipynb) · [Open in Colab](https://colab.research.google.com/github/pswpswpsw/nif/blob/pytorch/tutorial/1_simple_1d_wave_pytorch.ipynb)
2. [Multiscale fields](tutorial/2_multi_scale_NIF_pytorch.ipynb) · [Open in Colab](https://colab.research.google.com/github/pswpswpsw/nif/blob/pytorch/tutorial/2_multi_scale_NIF_pytorch.ipynb)
3. [Linear representation of cylinder flow](tutorial/3_multi_scale_linear_NIF_pytorch.ipynb) · [Open in Colab](https://colab.research.google.com/github/pswpswpsw/nif/blob/pytorch/tutorial/3_multi_scale_linear_NIF_pytorch.ipynb)
4. [Jacobians and Hessians](tutorial/4_get_gradients_by_wrapping_model_with_layer_pytorch.ipynb) · [Open in Colab](https://colab.research.google.com/github/pswpswpsw/nif/blob/pytorch/tutorial/4_get_gradients_by_wrapping_model_with_layer_pytorch.ipynb)
5. [Streaming a million TFRecord rows](tutorial/5_large_scale_training_on_tensorflow_record_data_pytorch.ipynb) · [Open in Colab](https://colab.research.google.com/github/pswpswpsw/nif/blob/pytorch/tutorial/5_large_scale_training_on_tensorflow_record_data_pytorch.ipynb)
6. [L1/L2 regularization and superresolution](tutorial/6_revisit_multi_scale_NIF_with_L1_L2_regularization_pytorch.ipynb) · [Open in Colab](https://colab.research.google.com/github/pswpswpsw/nif/blob/pytorch/tutorial/6_revisit_multi_scale_NIF_with_L1_L2_regularization_pytorch.ipynb)
7. [Pruning, native int8 inference, and export](tutorial/7_model_pruning_and_quantization_pytorch.ipynb) · [Open in Colab](https://colab.research.google.com/github/pswpswpsw/nif/blob/pytorch/tutorial/7_model_pruning_and_quantization_pytorch.ipynb)
8. [Sobolev training and physical superresolution](tutorial/8_revisit_multi_scale_NIF_with_sobolov_training_pytorch.ipynb) · [Open in Colab](https://colab.research.google.com/github/pswpswpsw/nif/blob/pytorch/tutorial/8_revisit_multi_scale_NIF_with_sobolov_training_pytorch.ipynb)

The original `tutorial/*.ipynb` and `*_colab.ipynb` files remain unchanged as
**legacy TensorFlow reference artifacts**. `tutorial/colab_support/`, the
existing public compatibility tests under `tests/`, and preexisting generated
HTML API docs describe that legacy workflow. They are not native setup tools.
The [archived README](docs/README_TENSORFLOW_LEGACY.md) preserves its documentation.
Native development checks live separately in `dev_tests/`:

```bash
python -m pytest dev_tests -q
```

## How to cite

If you find NIF is helpful to you, you can cite our JMLR [paper](https://www.jmlr.org/papers/volume24/22-0365/22-0365.pdf) in the following bibtex format

   ```
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
