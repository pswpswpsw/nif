"""Neural Implicit Flow: native PyTorch dimensionality reduction."""
from .__about__ import __version__
from . import data, demo, optimizers, layers
from .model import NIF, NIFMultiScale, NIFMultiScaleLastLayerParameterized

__all__ = ['NIF', 'NIFMultiScale', 'NIFMultiScaleLastLayerParameterized',
           'data', 'demo', 'optimizers', 'layers', '__version__']
