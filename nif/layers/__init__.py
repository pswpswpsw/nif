from nif.layers.siren import SIREN
from nif.layers.siren import SIREN_ResNet
from nif.layers.siren import HyperLinearForSIREN
from nif.layers.mlp import MLP_ResNet
from nif.layers.mlp import MLP_SimpleShortCut

from nif.layers.gradient import *

from tensorflow.keras.layers import Dense

__all__ = [
    "SIREN",
    "SIREN_ResNet",
    "Dense",
    "HyperLinearForSIREN",
    "MLP_ResNet",
    "MLP_SimpleShortCut",
    "JacRegLatentLayer",
    "JacobianLayer",
    "HessianLayer"
]