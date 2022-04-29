from .siren import SIREN
from .siren import SIREN_ResNet
from .siren import HyperLinearForSIREN
from .mlp import MLP_ResNet
from .mlp import MLP_SimpleShortCut

from .gradient import *

from tensorflow.keras.layers import Dense

__all__ = [
    "SIREN",
    "SIREN_ResNet",
    "Dense",
    "HyperLinearForSIREN",
    "MLP_ResNet",
    "MLP_SimpleShortCut",
    "JacRegLatentLayer",
]