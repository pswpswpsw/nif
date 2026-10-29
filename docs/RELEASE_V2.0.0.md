# NIF v2.0.0 — PyTorch edition

NIF 2.0.0 makes the native PyTorch implementation the default package. It retains
the mesh-agnostic ParameterNet/ShapeNet representation of parametric
spatiotemporal fields and provides eight native PyTorch tutorials.

## Included in this release

- NIF, multiscale SIREN fields, and last-layer-parameterized linear spatial bases.
- Differentiable input-output Jacobians and Hessians, normalization, regularization,
  streamed record datasets, and model checkpoints.
- Native first-order optimizers and reference-convention L-BFGS refinement.
- A standalone distributed training example with TensorBoard logging, learning-rate
  scheduling, and checkpoints.
- Magnitude pruning, dynamic int8 CPU inference, and inference export examples.
- A PyTorch homepage with the original visualizations, eight Colab tutorial links,
  and executable examples for training, monitoring, and compression.

## Installation and migration

Python 3.10+ and PyTorch 2.6+ are required. CUDA is optional. Install the release
from its Git tag:

```bash
python -m pip install "git+https://github.com/pswpswpsw/nif.git@v2.0.0"
```

The backend change is the reason for the major version increment. Training uses
PyTorch modules, explicit optimization loops and `state_dict` checkpoints. Legacy
backend checkpoints are not directly loadable by the PyTorch implementation.
See the [API and migration notes](PYTORCH_API.md).

The original notebooks and their compatibility copies remain in the repository
as scientific reference artifacts. Native tutorials use the `_pytorch.ipynb`
filenames linked from the homepage. Local self-scaled optimizer experiments are
not part of this release.

## Validation

The recorded tutorial comparisons and author acceptance decisions are in the
[validation report](PYTORCH_VALIDATION.md) and
[comparison records](PYTORCH_TENSORFLOW_PARITY.md). They retain measured numerical
differences and evaluation scope.

Before release preparation, the complete public CPU regression suite passed
599 checks with four environment-dependent skips. The lint cleanup was also
independently checked using fixed actual weights, batch order, precision and
optimizer settings: 228,112 tensor elements and 3,222 scalar entries were exactly
preserved. Release preparation adds metadata, distribution and installation
checks; it does not introduce a new full-training convergence experiment.

Distributed examples have been executed with two CPU/Gloo processes and one
CUDA device. Two-GPU CUDA/NCCL execution has not been measured on the local
one-GPU machine.

## Citation and archival versions

The [software citation metadata](../CITATION.cff) describes this release. The
[concept DOI](https://doi.org/10.5281/zenodo.7909312) identifies the software across
archived versions; Zenodo assigns a separate DOI to each archived release.
The JMLR paper remains the reference for the original method.

The 2023 releases and their existing DOI records are retained. GitHub Release
publication builds and checks distributions. PyPI upload is a separate manual
workflow action; the GitHub release does not itself imply PyPI publication.
