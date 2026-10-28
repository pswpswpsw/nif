"""Native PyTorch layers and differentiable scientific helpers."""
from torch.nn import Linear as Dense

from .gradient import compute_output_and_augment_grad
from .gradient import compute_output_and_grad
from .gradient import compute_output_and_grad_and_hessian
from .gradient import HessianLayer
from .gradient import JacobianLayer
from .gradient import JacRegLatentLayer
from .mlp import BiasAddLayer
from .mlp import EinsumLayer
from .mlp import MLP_ResNet
from .mlp import MLP_SimpleShortCut
from .regularization import ParameterOutputL1ActReg
from .siren import compute_number_of_weightbias_by_its_position_for_shapenet
from .siren import gen_hypernetwork_weights_bias_for_siren_shapenet
from .siren import HyperLinearForSIREN
from .siren import SIREN
from .siren import SIREN_ResNet

__all__ = [
    "SIREN",
    "SIREN_ResNet",
    "Dense",
    "HyperLinearForSIREN",
    "MLP_ResNet",
    "MLP_SimpleShortCut",
    "JacRegLatentLayer",
    "JacobianLayer",
    "HessianLayer",
    "ParameterOutputL1ActReg",
    "EinsumLayer",
    "BiasAddLayer",
    "compute_output_and_grad",
    "compute_output_and_augment_grad",
    "compute_output_and_grad_and_hessian",
    "compute_number_of_weightbias_by_its_position_for_shapenet",
    "gen_hypernetwork_weights_bias_for_siren_shapenet",
]
