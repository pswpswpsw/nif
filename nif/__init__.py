"""Neural Implicit Flow: native PyTorch dimensionality reduction."""
from . import data
from . import demo
from . import layers
from . import optimizers
from .__about__ import __version__
from .model import NIF
from .model import NIFMultiScale
from .model import NIFMultiScaleLastLayerParameterized

__all__ = [
    "NIF",
    "NIFMultiScale",
    "NIFMultiScaleLastLayerParameterized",
    "data",
    "demo",
    "optimizers",
    "layers",
    "__version__",
]
