Magnitude pruning
=================

``nif.pruning.PolynomialMagnitudePruning`` controls one selected kernel of an
eager PyTorch layer. It updates its magnitude mask on scheduled training
forwards and reapplies that mask before every forward, including evaluation.
The bias is unchanged unless it is explicitly selected as the parameter.

.. autoclass:: nif.pruning.PolynomialMagnitudePruning
   :members: on_epoch_end, strip, mask, step, refresh_count
   :undoc-members:

Lifecycle
---------

1. Attach the controller before creating or restoring a pruning checkpoint.
2. For each optimizer update, run one forward followed by its backward and step.
3. Call ``on_epoch_end()`` before saving a checkpoint or evaluating final weights.
4. Call ``strip()`` before copying, quantizing, tracing or exporting the inference
   model. This keeps the sparse parameter values and removes pruning hooks and
   control buffers.

Optimizer updates can regrow masked entries until the mask is reapplied.
Threshold ties mean the observed sparsity can differ from the requested
schedule. Multiple pending forward graphs followed by a combined backward are
unsupported when coordinate gradients are involved because masking changes
the parameter in place.

The parameter name depends on the layer: native ``torch.nn.Linear`` uses
``name="weight"``, while the tutorial's ``HyperLinearForSIREN`` uses ``name="w"``.
The controller's mask, threshold, counter and schedule are part of the model's
``state_dict``. To resume, attach a controller to the same kernel before loading
that state, and restore the optimizer state separately.

See the homepage's network pruning and quantization example for a small
float32 NIF workflow, and the native Tutorial 7 notebook for the full multiscale
experiment and dynamic int8 CPU inference.
