"""Native PyTorch layers and differentiable scientific helpers."""
from torch.nn import Linear as Dense
from .gradient import (HessianLayer, JacobianLayer, JacRegLatentLayer,
    compute_output_and_grad, compute_output_and_augment_grad,
    compute_output_and_grad_and_hessian)
from .mlp import BiasAddLayer, EinsumLayer, MLP_ResNet, MLP_SimpleShortCut
from .regularization import ParameterOutputL1ActReg
from .siren import (HyperLinearForSIREN, SIREN, SIREN_ResNet,
    compute_number_of_weightbias_by_its_position_for_shapenet,
    gen_hypernetwork_weights_bias_for_siren_shapenet)

__all__ = ['SIREN', 'SIREN_ResNet', 'Dense', 'HyperLinearForSIREN', 'MLP_ResNet',
    'MLP_SimpleShortCut', 'JacRegLatentLayer', 'JacobianLayer', 'HessianLayer',
    'ParameterOutputL1ActReg', 'EinsumLayer', 'BiasAddLayer',
    'compute_output_and_grad', 'compute_output_and_augment_grad',
    'compute_output_and_grad_and_hessian',
    'compute_number_of_weightbias_by_its_position_for_shapenet',
    'gen_hypernetwork_weights_bias_for_siren_shapenet']
