# Native PyTorch migration contract

Baseline: `f8f9d31f4aa331b6e7dbc48758302c801441bcce` on `main`.
Implementation branch: `pytorch`.

This is a complete native PyTorch port of NIF, not a TensorFlow wrapper. The
existing TensorFlow notebooks remain preserved as reference artifacts. New
`tutorial/*_pytorch.ipynb` notebooks must demonstrate all eight workflows.

## Public behavior

- `nif.NIF`, `NIFMultiScale`, and `NIFMultiScaleLastLayerParameterized` are
  `torch.nn.Module` classes, retaining their configuration dictionaries and
  scientific definitions. `forward` is the primary execution interface;
  `call` may be retained as a compatibility alias. `build()` and `model()`
  return an executable native module, not a second independent parameter set.
- Preserve parameter/coordinate input ordering, generated weight/bias layout,
  activation and residual definitions, SIREN initialization, and all
  originally supported configuration paths. Resolve documented baseline bugs
  according to their intended mathematical behavior and record corrections.
  Known baseline corrections include tutorial 6's superresolution coordinate
  ordering (`[parameter, coordinate]`), tutorial 8's physical inverse scaling,
  broken last-layer subnet/bias references, and use of the shape network's own
  regularization coefficients rather than parameter-network coefficients.
- Preserve `model_p_to_w`, `model_p_to_lr`, `model_lr_to_w`,
  `model_x_to_u_given_w`, and last-layer `model_x_to_phi`. Extracted native
  modules share the parent parameters and preserve autograd connectivity.
- Preserve every exported layer and mathematical helper in `nif.layers`.
  `Dense` can be the native `torch.nn.Linear` alias. Other existing layer
  constructor parameters retain their meaning, including precision settings.
- Jacobian helpers return the full model output and selected derivatives of
  shape `[batch, selected_outputs, selected_inputs]`. Hessians add a second
  selected-input axis. Derivative outputs remain differentiable with respect
  to trainable parameters for Sobolev training and derivative regularization.
  Constant and linear functions have valid zero higher derivatives.
- Expose `regularization_loss(inputs=None)` as a scalar differentiable tensor
  on NIF models. Preserve kernel/bias, activity, and latent-Jacobian penalties
  and their original reductions. Input-dependent penalties require inputs;
  do not silently use stale cached batches. Do not confuse L2 penalties with
  decoupled optimizer weight decay.
- Use native device/dtype handling, CPU and CUDA execution, `state_dict`,
  optimizer state dictionaries, and explicit PyTorch training loops.
  `compile`/`fit`/Keras checkpoint file compatibility is not required. A
  convenience `predict` is optional. `save_config` retains portable configs.
- `AdaBeliefOptimizer`, `Lion`, and `L4Adam` are native
  `torch.optim.Optimizer` implementations; preserve actual algorithms,
  supported options, parameter-group/state handling, and gradient
  centralization. Do not replace them with Adam/AdamW. Preserve native L-BFGS
  functionality and tensor packing helpers. Existing L-BFGS names can remain
  aliases, but must execute through PyTorch, without TensorFlow Probability.
- The baseline L4Adam update is unfinished (`new_var = None`). Complete its
  intended loss-based global update rather than reproducing that defect.
  Use the published L4-Adam algorithm, a closure/current loss, global
  gradient-direction inner product, and serializable moving loss minimum:
  https://proceedings.neurips.cc/paper_files/paper/2018/file/98b17f068d5d9b7668e19fb8ae470841-Paper.pdf
  Document this intentional completion and the exact update convention.
  Use bias-corrected momentum-averaged gradient `g` and second moment `s`,
  direction `v = g / (sqrt(s) + epsilon)`, and one global `sum(g * v)`
  across every parameter. Initialize `L_min = gamma_0 * L`, take
  `min(L_min, L)` before the update, use
  `alpha * (L - gamma * L_min) / (sum(g * v) + epsilon)` as the global
  step multiplier, and apply forgetting `L_min *= (1 + 1/tau)` afterward.
  The additive epsilon in the direction is an explicit numerical safeguard;
  retain the paper's denominator epsilon separately. State these conventions
  in the optimizer documentation so they can be evaluated independently.
- Preserve pointwise normalization, optional area weights, demo datasets,
  and physical inverse scaling. Check zero-range/invalid inputs explicitly.
- Retain `TFRDataset` method names and read/write the original TFRecord shard
  representation without a TensorFlow runtime dependency. Return native
  PyTorch iterables/DataLoaders for streamed training; preserve every row,
  target column, area weight, shard remainder, and epoch repetition. A new
  native shard format may be added, but cannot replace legacy data reading.
- Importing or running the PyTorch package/tutorials must not import or
  require TensorFlow, Keras, TensorFlow Probability, or TFMOT. Archive legacy
  compatibility tools clearly. Packaging and main documentation describe the
  PyTorch implementation. Do not remove or overwrite reference TF notebooks.

## Tutorial integration acceptance

Port the models, data, original comments/teaching narrative, plotting,
checkpoint restore, and algorithms of all tutorials. Required full workloads:

1. Traveling wave: 5,000 AdaBelief epochs and L-BFGS rounds=200/max_iter=1000.
2. Multiscale: two 5,000-epoch fits.
3. Linear representation: 600 epochs on all 191,775 cylinder-flow samples,
   including weighted loss and actual gradient centralization.
4. Full Jacobian/Hessian examples and differentiable derivative wrappers.
5. One million generated rows, 100 shards, two complete streamed passes.
6. Baseline and eight L1/L2 settings: nine 3,000-epoch fits.
7. 5,000 baseline epochs, 50,000 pruning epochs, actual 90% sparsity,
   real native int8 quantized inference, and restored/exported predictions.
8. Two 50,000-epoch fits, spatial-derivative Sobolev loss, temporal derivative
   metric, and superresolution compared in consistent physical units.

Separate short smoke runs are useful development checks; they do not replace
full workloads. Test actual tensor values, gradients, parameter updates,
checkpoint/prediction consistency, finite losses, exported artifact formats,
and representative figures. Numerical agreement allows documented floating
point differences and different framework random-number generators.

## Blind evaluation protocol

Agent A owns implementation and can create its own development tests.
Independent Agent B writes and freezes acceptance tests outside the repository
before A changes implementation, then runs them on immutable candidate
snapshots. A receives public interface requirements and sanitized failure
logs, never private test sources or fixtures. B cannot weaken tests to accept
an implementation. Necessary evaluator corrections require an evidence-backed
record and coordinator review. Neither self-reported success nor smoke-only
success is a final pass.
