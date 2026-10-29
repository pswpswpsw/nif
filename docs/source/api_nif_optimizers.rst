Optimizers
==========

The public ``nif.optimizers`` exports operate on native PyTorch parameters.
Pass ``model.parameters()`` to an optimizer and add model regularization to
the loss explicitly. Optimizer state is saved and restored with
``state_dict()`` and ``load_state_dict()``.

First-order methods
-------------------

.. code-block:: python

   import torch
   from nif.optimizers import ReferenceAdam

   value = torch.nn.Parameter(torch.tensor([1.0]))
   optimizer = ReferenceAdam([value], lr=1e-3)
   optimizer.zero_grad(set_to_none=True)
   loss = value.square().mean()
   loss.backward()
   optimizer.step()

.. autoclass:: nif.optimizers.ReferenceAdam
   :members: step, state_dict, load_state_dict
   :undoc-members:

``ReferenceAdam`` places epsilon after the square root of the raw second moment.
``torch.optim.Adam`` uses a different epsilon convention. Use the optimizer
selected by the relevant tutorial when reproducing that tutorial's trajectory.

.. autoclass:: nif.optimizers.AdaBeliefOptimizer
   :members: step
   :undoc-members:

The displayed ``eps=_UNSET`` default denotes the internal unspecified sentinel.
Omitting ``eps`` keeps
``epsilon`` in control (AdaBelief's default is ``1e-14``); explicitly passing
``eps=None`` or ``eps=0`` selects the ``1e-7`` fallback. Native ``lr`` and
``betas`` aliases take precedence over their long-form counterparts when supplied.

.. autoclass:: nif.optimizers.Lion
   :members: step
   :undoc-members:

.. autoclass:: nif.optimizers.L4Adam
   :members: step, l_min
   :undoc-members:

For ``L4Adam``, supply the current loss through ``step(loss=loss)`` after
backpropagation, or pass a closure returning that loss. Its learning rate is
the loss-based step multiplier described in the class documentation. Omitting
``eps`` keeps ``epsilon`` in control; its default and the explicit None/zero
fallback are both ``1e-7``.

L-BFGS refinement
------------------

Use a fixed dataset in the closure, clear gradients, recompute the complete
scalar objective and call ``backward()`` on each evaluation. The following
example illustrates the closure interface on a scalar optimization problem.

.. code-block:: python

   import torch
   from nif.optimizers import ReferenceLBFGS

   value = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
   optimizer = ReferenceLBFGS([value], max_iter=10)

   def closure():
       optimizer.zero_grad(set_to_none=True)
       loss = value.square().mean()
       loss.backward()
       return loss

   optimizer.step(closure)
   final_objective = optimizer.result.objective_value

.. autoclass:: nif.optimizers.ReferenceLBFGS
   :members: step, state_dict, load_state_dict
   :undoc-members:

Packed objective helpers
------------------------

The helper APIs expose the same native L-BFGS core through flattened parameter
vectors. ``lbfgs_minimize`` treats ``max_iterations`` as a cumulative budget
when resuming; ``ReferenceLBFGS.max_iter`` limits additional iterations for
each ``step`` call.

.. autofunction:: nif.optimizers.lbfgs_minimize

.. autoclass:: nif.optimizers.LBFGSOptimizer
   :members: minimize, epoch, loss
   :undoc-members:

.. autoclass:: nif.optimizers.TFPLBFGS
   :members: minimize, history
   :undoc-members:

``TFPLBFGS`` is a retained historical class name for native execution. Its
``loss_fun`` follows ``loss_fun(prediction, target)`` argument ordering.

.. autofunction:: nif.optimizers.function_factory

.. autofunction:: nif.optimizers.create_value_and_gradient_function

.. autofunction:: nif.optimizers.pack_tensors

.. autofunction:: nif.optimizers.unpack_tensors

.. autofunction:: nif.optimizers.assign_tensors

Gradient centralization
-----------------------

The installer adds a native step hook. After installation, the ordinary
``loss.backward(); optimizer.step()`` sequence uses centralized gradients.

.. autofunction:: nif.optimizers.centralized_gradients_for_optimizer

.. autofunction:: nif.optimizers.get_centralized_gradients
