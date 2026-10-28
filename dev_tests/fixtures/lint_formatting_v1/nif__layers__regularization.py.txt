"""Explicit differentiable regularization wrappers."""
from ._common import NativeLayer


class ParameterOutputL1ActReg(NativeLayer):
    """Pass through augmented output and penalize all parameter activations.

    ``model(inputs)`` returns ``(output, parameter_output)``. The original
    reduction is the sum of absolute activations over the entire batch; this
    wrapper intentionally does not divide that sum by the number of samples.
    """
    def __init__(self,model,l1=.1):
        super().__init__(); self.model=model; self.l1=float(l1)

    def forward(self,x,**kwargs):
        y,_=self.model(x)
        return y

    def regularization_loss(self,inputs=None):
        if inputs is None: raise ValueError('inputs are required for parameter-output regularization')
        _,po=self.model(inputs)
        return self.l1*po.abs().sum()

    def get_config(self): return {'l1':self.l1}
