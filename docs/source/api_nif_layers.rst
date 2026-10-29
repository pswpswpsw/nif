Layers and derivatives
======================

These public helpers are exported by ``nif.layers``. They use PyTorch tensors
and retain autograd connections for derivative-based training.

Pointwise derivatives
---------------------

For a pointwise model with independent batch rows, the selected Jacobian has
shape ``[batch, selected outputs, selected inputs]``. The Hessian adds a second
selected-input axis. ``y_index`` selects output columns; ``x_index`` selects
input columns.

.. code-block:: python

   import torch
   from nif.layers import JacobianLayer, HessianLayer

   field = torch.nn.Linear(2, 1)
   features = torch.tensor([[0.1, -0.2], [0.3, 0.4]])
   output, jacobian = JacobianLayer(field, y_index=[0], x_index=[1])(features)
   output, jacobian, hessian = HessianLayer(field, y_index=[0], x_index=[1])(features)
   assert jacobian.shape == (2, 1, 1)
   assert hessian.shape == (2, 1, 1, 1)

.. autoclass:: nif.layers.JacobianLayer
   :members: forward, get_config
   :undoc-members:

.. autoclass:: nif.layers.HessianLayer
   :members: forward
   :undoc-members:

The following function forms expose the same calculations. Note that their
positional index arguments are ordered ``x_index, y_index``.

.. autofunction:: nif.layers.compute_output_and_grad

.. autofunction:: nif.layers.compute_output_and_grad_and_hessian

.. autofunction:: nif.layers.compute_output_and_augment_grad

Latent and activation penalties
--------------------------------

These wrappers expect an augmented model returning ``(output, latent)`` or
``(output, parameter_output)``. Call their ``regularization_loss(inputs)``
explicitly in the objective for the current batch.

.. autoclass:: nif.layers.JacRegLatentLayer
   :members: forward, regularization_loss, get_config
   :undoc-members:

.. autoclass:: nif.layers.ParameterOutputL1ActReg
   :members: forward, regularization_loss, get_config
   :undoc-members:

SIREN and hypernetwork layers
------------------------------

Custom kernels use ``[input, output]`` storage. In sinusoidal layers the argument
is ``omega_0 * (x @ w) + b``: the bias is not multiplied by ``omega_0``.

.. autoclass:: nif.layers.SIREN
   :members: forward, regularization_loss, get_config, get_prunable_weights
   :undoc-members:

.. autoclass:: nif.layers.SIREN_ResNet
   :members: forward, get_config, get_prunable_weights
   :undoc-members:

.. autoclass:: nif.layers.HyperLinearForSIREN
   :members: forward, regularization_loss, get_config, get_prunable_weights
   :undoc-members:

.. autofunction:: nif.layers.compute_number_of_weightbias_by_its_position_for_shapenet

.. autofunction:: nif.layers.gen_hypernetwork_weights_bias_for_siren_shapenet

Dense and residual helpers
--------------------------

.. py:class:: nif.layers.Dense(in_features, out_features, bias=True, device=None, dtype=None)

   A direct alias of ``torch.nn.Linear``. Use the native PyTorch constructor
   and its ``weight`` layout ``[output, input]``. See the
   `PyTorch Linear reference <https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html>`_.

.. autoclass:: nif.layers.MLP_ResNet
   :members: forward, regularization_loss, get_config, get_prunable_weights
   :undoc-members:

.. autoclass:: nif.layers.MLP_SimpleShortCut
   :members: forward, regularization_loss, get_config, get_prunable_weights
   :undoc-members:

.. autoclass:: nif.layers.EinsumLayer
   :members: forward, get_config
   :undoc-members:

.. autoclass:: nif.layers.BiasAddLayer
   :members: forward, get_config
   :undoc-members:
