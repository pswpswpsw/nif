"""Public TF AdaBelief's staged final parameter update, without a TF dependency.

Preserved ad9113d/f8f9d31 external_optimizers.py:524 forms lr_t * var_t
before assign_sub. Fixed, self-selected CPU operands expose the difference
from add_(direction, alpha=-lr). This tests only that final execution boundary,
not cross-framework bitwise training or the independently tested moments.
"""

import pytest
import torch

from nif.optimizers import AdaBeliefOptimizer


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64],
                         ids=['float32', 'float64'])
def test_unrectified_final_update_rounds_product_before_subtraction(dtype):
    # Public CPU diagnostic operands; nonzero parameters expose cancellation.
    lr = .0031
    values = [1.125, -.875, 2.75, -3.25, .15625, -.0625,
              31.5, -17.25, 2**-15, -2**-12, 123.125, -91.75]
    initial_values = [1.125, -.875, .0625, -.03125, 2**-10, -2**-8,
                      16.25, -8.5, .000125, -.0025, .375, -.25]
    # Keep the diagnostic's complete 24-value vector, including binary64-rounded
    # lr*g initial values; CPU vector kernels can differ from scalar/tail paths.
    initial = torch.tensor(initial_values + [lr*g for g in values], dtype=dtype)
    gradient = torch.tensor(values + values, dtype=dtype)
    parameter = torch.nn.Parameter(initial.clone())
    optimizer = AdaBeliefOptimizer([parameter], lr=lr, rectify=False)
    parameter.grad = gradient.clone()
    optimizer.step()

    # Reconstruct direction from the actual step's moments. Other public tests
    # independently verify those moments; this isolates the final p update.
    state = optimizer.state[parameter]
    beta1, beta2 = (initial.new_tensor(b) for b in (.9, .999))
    step = initial.new_tensor(1)
    direction = (state['exp_avg'] / (1 - beta1.pow(step))) / (
        (state['exp_avg_var'] / (1 - beta2.pow(step))).sqrt() + 1e-14)
    delta = initial.new_tensor(lr) * direction
    expected = initial - delta
    torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
