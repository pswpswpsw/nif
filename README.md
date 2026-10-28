# Neural Implicit Flow (NIF): mesh-agnostic dimensionality reduction
![example workflow](https://github.com/pswpswpsw/nif/workflows/Tests/badge.svg)
[![License](https://img.shields.io/github/license/pswpswpsw/nif)](https://github.com/pswpswpsw/nif/blob/main/LICENSE)[![DOI](https://zenodo.org/badge/460537754.svg)](https://zenodo.org/badge/latestdoi/460537754)

<p align="center">
  <img src="./misc/myimage.gif" alt="animated" />
</p>

- NIF is a mesh-agnostic dimensionality reduction paradigm for parametric spatial temporal fields. For decades, dimensionality reduction (e.g., proper orthogonal decomposition, convolutional autoencoders) has been the very first step in reduced-order modeling of any large-scale spatial-temporal dynamics.

- Unfortunately, these frameworks are either not extendable to realistic industry scenario, e.g., adaptive mesh refinement, or cannot preceed nonlinear operations without resorting to lossy interpolation on a uniform grid. Details can be found in our [paper](https://arxiv.org/pdf/2204.03216.pdf).

- The current default implementation is built on **PyTorch**. The original TensorFlow/Keras implementation remains available as a pinned historical revision; its highlights and tutorials are preserved below.


## Highlights

The following original highlights and examples describe the **legacy TensorFlow/Keras implementation**. They are kept as historical documentation; the current PyTorch API and tutorials follow this section. To use the original package, install or check out the pinned TensorFlow revision [`f8f9d31`](https://github.com/pswpswpsw/nif/tree/f8f9d31f4aa331b6e7dbc48758302c801441bcce).

<p align="center">
  <img src="./misc/compare.gif" alt="animated" />
</p>


- Built on top of **Tensorflow 2.x** with **Keras model subclassing**, hassle-free for many up-to-date advanced concepts and features

    ```python
    from nif import NIF

    # set up the configurations, loading dataset, etc...

    model_ori = NIF(...)
    model_opt = model_ori.build()

    model_opt.compile(optimizer, loss='mse')
    model_opt.fit(...)

    model_opt.predict(...)
    ```

- **Distributed learning**: data parallelism across multiple GPUs on a single node

    ```python
    enable_multi_gpu = True
    cm = tf.distribute.MirroredStrategy().scope() if enable_multi_gpu else contextlib.nullcontext()
    with cm:

        # ...
        model.fit(...)
        # ...
    ```

- Flexible training schedule: e.g., first some standard optimizer (e.g., Adam) then **fine-tunning with L-BFGS**

    ```python
    from nif.optimizers import TFPLBFGS

    # load previous model
    new_model_ori = NIF(cfg_shape_net, cfg_parameter_net, mixed_policy)
    new_model.load_weights(...)

    # prepare the dataset
    data_feature = ... #
    data_label = ... #

    # fine tune with L-BFGS
    loss_fun = tf.keras.losses.MeanSquaredError()
    fine_tuner = TFPLBFGS(new_model, loss_fun, data_feature, data_label, display_epoch=10)
    fine_tuner.minimize(rounds=200, max_iter=1000)
    new_model.save_weights("./fine-tuned/ckpt")
    ```

- Templates for many useful customized callbacks

   ```python
   # setting up the model
   # ...

   # - tensorboard
   tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./tb-logs", update_freq='epoch')

   # - printing, model save checkpoints etc.
   class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
   # ....

   # - learning rate schedule
   def scheduler(epoch, lr):
       if epoch < 1000:
           return lr
       else:
           return 1e-4
   scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

   # - collecting callbacks into model.fit(...)

   callbacks = [tensorboard_callback, LossAndErrorPrintingCallback(), scheduler_callback]
   model_opt.fit(train_dataset, epochs=nepoch, batch_size=batch_size,
             shuffle=False, verbose=0, callbacks=callbacks)
   ```

- Simple extraction of subnetworks

    ```python
    model_ori = NIF(...)

    # ....

    # extract latent space encoder network
    model_p_to_lr = model_ori.model_p_to_lr()
    lr_pred = model_p_to_lr.predict(...)

    # extract latent-to-weight network: from latent representation to weights and biase of shapenet
    model_lr_to_w = model_ori.model_lr_to_w()
    w_pred = model_lr_to_w.predict(...)

    # extract shapenet: inputs are weights and spatial coordinates, output is the field of interests
    model_x_to_u_given_w = model_ori.model_x_to_u_given_w()
    u_pred = model_x_to_u_given_w.predict(...)
    ```

- Get input-output Jacobian or Hessian.
    ```python
    model = ... # your keras.Model
    x = ... # your dataset
    # define both the indices of target and source

    x_index = [0,1,2,3]
    y_index = [0,1,2,3,4]

    # wrap up keras.Model using JacobianLayer
    from nif.layers import JacobianLayer
    y_and_dydx_layer = JacobianLayer(model, y_index, x_index)

    y, dydx = y_and_dydx_layer(x)

    model_with_jacobian = Model([x], [y, dydx])

    # wrap up keras.Model using HessianLayer
    from nif.layers import HessianLayer
    y_and_dydx_and_dy2dx2_layer = HessianLayer(model, y_index, x_index)

    y, dydx, dy2dx2 = y_and_dydx_and_dy2dx2_layer(x)

    model_with_jacobian_and_hessian = Model([x], [y, dydx, dy2dx2])

    ```

- Data normalization for multi-scale problem

    - just simply feed `n_para`: number of parameters, `n_x`: input dimension of shapenet, `n_target`: output dimension of shapenet, and `raw_data`: numpy array with shape = `(number of pointwise data points, number of features, target, coordinates, etc.)`

    ```python
    from nif.data import PointWiseData
    data_n, mean, std = PointWiseData.minmax_normalize(raw_data=data, n_para=1, n_x=3, n_target=1)
    ```

- Large-scale training with tfrecord converter

    - all you need is to prepare a BIG npz file that contains all the point-wise data
    - `.get_tfr_meta_dataset` will read all files under the searched directory that ends with `.tfrecord`

    ```python
    from nif.data.tfr_dataset import TFRDataset
    fh = TFRDataset(n_feature=4, n_target=3)

    # generating tfrecord files from a single big npz file (say gigabytes)
    fh.create_from_npz(...)

    # prepare some model
    model = ...
    model.compile(...)

    # obtaining a meta dataset
    meta_dataset = fh.get_tfr_meta_dataset(...)

    # start sub-dataset-batching
    for batch_file in meta_dataset:
        batch_dataset = fh.gen_dataset_from_batch_file(batch_file, batch_size)
        model.fit(...)
    ```
- Save and load models (via Checkpoints only)
    ```python
    # save the config
    model.save_config("config.json")

    # save the weights
    model.save_weights("./saved_weights/ckpt-{}/ckpt".format(epoch)")

    # load the config
    with open("config.json", "r") as f:
        config = json.load(f)
    model_ori = nif.NIF(**config)
    model = model_ori.model()

    # load the weights
    model.load_weights("./saved_weights/ckpt-999/ckpt")
    ```
- Network pruning and quantization


## PyTorch implementation (current default)

NIF learns mesh-agnostic representations of spatiotemporal fields. A parameter
network maps time and operating conditions to a coordinate-dependent shape
network. The default `main` branch implements the complete package in PyTorch, including
SIREN multiscale fields, linear spatial bases, differentiable derivatives,
streamed TFRecord data, and native optimizers.

Independent acceptance passed **253 checks in the CUDA-enabled environment and
all eight full tutorial workloads**. The tutorial comparisons were reviewed by
the author within the recorded scopes; the reports retain numerical differences
and do not claim identical TensorFlow/PyTorch training trajectories. See
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

1. [Traveling wave and L-BFGS](tutorial/1_simple_1d_wave_pytorch.ipynb) · [Open in Colab](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/1_simple_1d_wave_pytorch.ipynb)
2. [Multiscale fields](tutorial/2_multi_scale_NIF_pytorch.ipynb) · [Open in Colab](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/2_multi_scale_NIF_pytorch.ipynb)
3. [Linear representation of cylinder flow](tutorial/3_multi_scale_linear_NIF_pytorch.ipynb) · [Open in Colab](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/3_multi_scale_linear_NIF_pytorch.ipynb)
4. [Jacobians and Hessians](tutorial/4_get_gradients_by_wrapping_model_with_layer_pytorch.ipynb) · [Open in Colab](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/4_get_gradients_by_wrapping_model_with_layer_pytorch.ipynb)
5. [Streaming a million TFRecord rows](tutorial/5_large_scale_training_on_tensorflow_record_data_pytorch.ipynb) · [Open in Colab](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/5_large_scale_training_on_tensorflow_record_data_pytorch.ipynb)
6. [L1/L2 regularization and superresolution](tutorial/6_revisit_multi_scale_NIF_with_L1_L2_regularization_pytorch.ipynb) · [Open in Colab](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/6_revisit_multi_scale_NIF_with_L1_L2_regularization_pytorch.ipynb)
7. [Pruning, native int8 inference, and export](tutorial/7_model_pruning_and_quantization_pytorch.ipynb) · [Open in Colab](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/7_model_pruning_and_quantization_pytorch.ipynb)
8. [Sobolev training and physical superresolution](tutorial/8_revisit_multi_scale_NIF_with_sobolov_training_pytorch.ipynb) · [Open in Colab](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/8_revisit_multi_scale_NIF_with_sobolov_training_pytorch.ipynb)

The original TensorFlow notebooks and the compatibility copies below remain
available as **legacy TensorFlow reference artifacts**. `tutorial/colab_support/`, the
existing public compatibility tests under `tests/`, and preexisting generated
HTML API docs describe that legacy workflow. They are not native setup tools.
The [archived README](docs/README_TENSORFLOW_LEGACY.md) preserves its documentation.
Native development checks live separately in `dev_tests/`:

```bash
python -m pytest dev_tests -q
```

## Google Colab Tutorial

The following are the original **TensorFlow compatibility tutorials**. Each compatibility notebook pins the original TensorFlow NIF source in its first cell; run from that cell. PyTorch tutorial links are listed above.

The links below open TensorFlow compatibility copies for Colab. Run each notebook from its first cell to install its pinned dependencies. The original notebooks are preserved in `tutorial/`; see the [runtime guides](tutorial/colab_support/README.md) and [validation results](tutorial/colab_support/VALIDATION.md).

1. **Hello world! A simple fitting on 1D travelling wave** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/1_simple_1d_wave_colab.ipynb)
    - learn how to use class `nif.NIF`
    - model checkpoints/restoration
    - mixed precision training
    - L-BFGS fine tuning

2. **Tackling multi-scale data** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/2_multi_scale_NIF_colab.ipynb)

    - learn how to use class `nif.NIFMultiScale`
    - demonstrate the effectiveness of learning high frequency data

3. **Learning linear representation** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/3_multi_scale_linear_NIF_colab.ipynb)
    - learn how to use class `nif.NIFMultiScaleLastLayerParameterized`
    - demonstrate on a (shortened) flow over a cylinder case from an AMR solver

4. **Getting input-output derivatives is super easy** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/4_get_gradients_by_wrapping_model_with_layer_colab.ipynb)
    - learn how to use `nif.layers.JacobianLayer`, `nif.layers.HessianLayer`

5. **Scaling to hundreds of GB data** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/5_large_scale_training_on_tensorflow_record_data_colab.ipynb)
    - learn how to use `nif.data.tfr_dataset.TFRDataset` to create `tfrecord` from `npz`
    - learn how to perform sub-dataset-batch training with `model.fit`

6. **Revisit NIF on multi-scale data with regularization** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/6_revisit_multi_scale_NIF_with_L1_L2_regularization_colab.ipynb)
    - learn how to use L1 or L2 regularization for weights and bias in ParameterNet.
    - a demonstration for the failure of NIF-Multiscale in terms of increasing spatial interpolation when dealing with
      high-frequency signal.
        - this means you need to be cautious about increasing spatial sampling resolution when dealing with
          high-frequency signal.
    - learn that L2 or L1 regularization does not seem to help resolving the above issue.

7. **NIF Compression** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/7_model_pruning_and_quantization_colab.ipynb)
    - learn how to use low magnititute pruning and quantization to compress ParameterNet

8. **Revisit NIF on multi-scale data: Sobolov training helps removing spurious signals** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pswpswpsw/nif/blob/main/tutorial/8_revisit_multi_scale_NIF_with_sobolov_training_colab.ipynb)
    - learn how to use `nif.layers.JacobianLayer` to perform Sobolov training
    - learn how to monitor different loss terms using customized Keras metrics
    - learn that feeding derivative information to the system help resolve the super-resolution issue

## Requirements for the historical TensorFlow examples

These were the original TensorFlow requirements. For the compatibility notebooks, use the pinned runtime installed by the first cell; for current PyTorch requirements, see the PyTorch section above.

```python
matplotlib
numpy
tensorflow-gpu
tensorflow_probability==0.18.0
tensorflow_model_optimization==0.7.3
```

## Issues, bugs, requests, ideas

Use the [issues](https://github.com/pswpswpsw/nif/issues) tracker to report bugs.

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
