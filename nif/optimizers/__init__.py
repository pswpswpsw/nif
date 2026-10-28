"""Native PyTorch optimization algorithms and L-BFGS migration helpers."""
from nif.optimizers.external_optimizers import AdaBeliefOptimizer
from nif.optimizers.external_optimizers import L4Adam
from nif.optimizers.external_optimizers import Lion
from nif.optimizers.gtcf import centralized_gradients_for_optimizer
from nif.optimizers.gtcf import get_centralized_gradients
from nif.optimizers.lbfgs import function_factory
from nif.optimizers.lbfgs import TFPLBFGS
from nif.optimizers.lbfgs_V2 import assign_tensors
from nif.optimizers.lbfgs_V2 import create_value_and_gradient_function
from nif.optimizers.lbfgs_V2 import lbfgs_minimize
from nif.optimizers.lbfgs_V2 import LBFGSOptimizer
from nif.optimizers.lbfgs_V2 import pack_tensors
from nif.optimizers.lbfgs_V2 import unpack_tensors
from nif.optimizers.reference_adam import ReferenceAdam
from nif.optimizers.reference_lbfgs import ReferenceLBFGS

__all__ = [
    "function_factory",
    "lbfgs_minimize",
    "LBFGSOptimizer",
    "TFPLBFGS",
    "ReferenceAdam",
    "ReferenceLBFGS",
    "L4Adam",
    "AdaBeliefOptimizer",
    "centralized_gradients_for_optimizer",
    "get_centralized_gradients",
    "Lion",
    "pack_tensors",
    "unpack_tensors",
    "assign_tensors",
    "create_value_and_gradient_function",
]
