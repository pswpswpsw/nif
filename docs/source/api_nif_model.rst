Models
======

The three public model classes are native ``torch.nn.Module`` objects and
are available directly from ``nif``. Each call accepts a floating-point matrix
with columns ordered as ``[parameters, spatial coordinates]`` and returns a
matrix of field values. Use an explicit PyTorch training loop and add
``model.regularization_loss(features)`` to its objective.

.. code-block:: python

   import torch
   from nif import NIF

   shape = dict(input_dim=1, output_dim=1, units=20, nlayers=2, activation="tanh")
   parameter = dict(input_dim=1, latent_dim=2, units=10, nlayers=2, activation="tanh")
   model = NIF(shape, parameter)
   features = torch.tensor([[0.1, -0.2], [0.3, 0.4]])
   targets = torch.zeros(2, 1)
   prediction = model(features)
   loss = (prediction - targets).square().mean() + model.regularization_loss(features)
   loss.backward()

Configuration and precision
---------------------------

``cfg_shape_net`` sets ``input_dim``, ``output_dim``, ``units`` and ``nlayers``.
``cfg_parameter_net`` sets ``input_dim``, ``latent_dim``, ``units`` and
``nlayers``. Plain ``NIF`` also reads each network's ``activation``.

Multiscale models additionally use ShapeNet ``omega_0``, ``use_resblock`` and
``connectivity``. Use ``connectivity="full"`` for ``NIFMultiScale`` and
``connectivity="last_layer"`` for ``NIFMultiScaleLastLayerParameterized``.
Their ParameterNet configuration includes ``activation`` and ``use_resblock``;
``activation="sine"`` also requires ``omega_0``. The full configurations appear
in the corresponding native tutorial notebooks.

The default ``mixed_policy="float32"`` uses float32 parameters and computation.
``mixed_float16`` and ``mixed_bfloat16`` use autocast computation with float32
trainable parameters. Native ``model.to(...)``, ``model.float()`` and
``model.double()`` control parameter device and dtype.

Parameter-conditioned MLP field
-------------------------------

.. autoclass:: nif.model.NIF
   :members: forward, call, build, model, model_p_to_w, model_p_to_lr, model_lr_to_w, model_x_to_u_given_w, regularization_loss, predict, save_config, variable_Dtype, compute_Dtype
   :undoc-members:

``model_p_to_w()`` maps parameters to generated ShapeNet weights.
``model_p_to_lr()`` extracts the latent representation, while
``model_lr_to_w()`` maps that representation to generated weights. The view
returned by ``model_x_to_u_given_w()`` accepts ``(coordinates, weights)``.
These views share the parent model's layer objects and gradient connections.

``save_config()`` stores construction arguments as JSON. Save trainable tensors
with ``torch.save(model.state_dict(), path)`` and restore them into a model
constructed from that configuration.

Set ``model.eval()`` when switching to evaluation mode. ``predict()`` runs under
``torch.no_grad()`` and returns a tensor on the model's device. The retained
``training`` and ``mask`` call arguments are compatibility parameters; native
module mode is controlled by ``train()`` and ``eval()``.

Multiscale SIREN field
----------------------

.. autoclass:: nif.model.NIFMultiScale

This class shares the training, regularization, checkpoint and subnetwork methods
documented for :class:`nif.model.NIF`. Its ShapeNet uses sinusoidal layers and
optional two-sine residual blocks.

Linear spatial representation
------------------------------

.. autoclass:: nif.model.NIFMultiScaleLastLayerParameterized
   :members: model_p_to_lr, model_lr_to_w, model_x_to_phi
   :undoc-members:

ParameterNet produces the coefficients of a shared spatial basis. The
``model_x_to_phi()`` view returns basis values indexed by
``[batch, output component, latent feature]``. ``model_lr_to_w()`` is an identity
view for this architecture. Other common methods are inherited from
:class:`nif.model.NIF`.
