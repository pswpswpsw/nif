# Native API and intentional migration corrections

The public requirements are in [PYTORCH_PORT_CONTRACT.md](PYTORCH_PORT_CONTRACT.md).
Preexisting generated HTML files and the archived TensorFlow README are legacy
reference material. They do not describe the native installation or execution API.

## Models and ordering

`NIF`, `NIFMultiScale`, and `NIFMultiScaleLastLayerParameterized` take the original
shape and parameter dictionaries, plus `mixed_policy` (default `float32`). Standard
`.to(device=..., dtype=...)`, `.double()`, and `.float()` control native tensors.
`mixed_float16` and `mixed_bfloat16` keep float32 trainable variables and use the
selected compute precision. CPU and CUDA use the same mathematical definitions.

Inputs are two-dimensional `[batch, parameter columns + coordinate columns]`.
All generated shape kernels occur first, in layer order, flattened from
`[input, output]`; biases follow, also in layer order. No transpose is introduced
when unpacking the hypernetwork output. Ordinary native `nn.Linear` layers store
transposed kernels internally, as usual.

Plain NIF uses `activation(xW+b)` at the first layer and
`u + activation(uW+b)` in hidden layers. Multiscale uses `sin(omega*xW+b)`;
its residual block is `(u + sin(omega*sin(omega*uW1+b1)W2+b2))/2`.
The frequency does not multiply the bias. Output layers are linear. Parameter
network MLP residual blocks use `activation(x+W2(activation(W1(x))))`.
SIREN initialization and the hypernetwork output-position-dependent bias ranges
match the original scientific definitions (framework random samples differ).

Extracted subnets register shared layer objects and never register the parent
back into itself. Optimizing a subnet therefore updates the parent, and
`state_dict()` traversal is finite. Last-layer subnets consistently evaluate
`u = einsum('bol,bl->bo', phi(x), a(p)) + bias`; the invalid legacy bias reference
is corrected. `model_lr_to_w()` is an identity for this architecture.

## Explicit regularization

Include `model.regularization_loss(inputs)` in the objective. Kernel and bias L1
penalties are coefficients times the sum of absolute values; L2 uses the sum of
squares without a half factor. As in the baseline, a configured numeric L2 takes
precedence over L1 for the same network. Shape parameters use the **shape**
network coefficients, correcting the legacy accidental parameter-net reference.
The final additive bias in the last-layer model is unregularized, as originally.

Parameter-output activity penalties are sums divided by batch size, matching
Keras activity-loss scaling. Latent Jacobian regularization is the coefficient
times the mean squared derivative over batch, latent features, and parameter
inputs. Its latent tensor is the parameter network bottleneck. Input-dependent
penalties require `inputs` every time, even after a forward pass; no stale cached
batch is used. With no penalties the returned zero remains differentiable.

## Derivatives

`JacobianLayer(model, y_index, x_index)` returns `(full_output, selected_jacobian)`.
`HessianLayer` adds the selected Hessian, with shape
`[batch, outputs, inputs, inputs]`. Integer and sequence selections are supported.
Constant/linear fields yield valid zero higher derivatives. The graph is kept
for trainable-parameter differentiation and Sobolev losses.

## Reference Adam epsilon convention

`nif.optimizers.ReferenceAdam` is the native optimizer for the source-dependent
Adam training in tutorials 2, 5, 6, 7 and 8. Defaults are `lr=1e-3`,
`betas=(0.9, 0.999)` and `eps=1e-7`; each notebook retains its original learning
rates, schedules and epochs. At optimizer iteration `t`, it applies
`alpha = lr * sqrt(1-beta2**t) / (1-beta1**t)` and
`parameter -= alpha*m / (sqrt(v)+eps)`, where `m` and `v` are the raw moments.
This is the reference's **epsilon-hat** convention: epsilon follows the square
root of the uncorrected variance. Setting the same epsilon on
`torch.optim.Adam` is not equivalent because that optimizer puts epsilon after
the square root of the bias-corrected variance. See the
[TensorFlow Adam epsilon documentation](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam).
Ordinary PyTorch Adam is unchanged, including tutorial 4's supplemental example.

The dense slot update follows the reference new Adam's add-difference order;
Python `(1-beta)` is evaluated before conversion to the parameter dtype. Powers
and the step scale use the parameter's dtype/device. Cross-backend scalar
promotion and fused-operation rounding are separate from epsilon placement.
In the tested reference, default Python betas can have different effective
precision in the power bases and moment coefficients. Strict float64 comparisons
therefore use explicitly typed beta tensors in TensorFlow and identical numeric
values in the native optimizer; merely transferring the rounded power bases
does not also match the default moment coefficients. Float32 scientific fits
retain the original Python beta values and match both operation conventions.
Bitwise agreement on controlled short traces does not imply identical long
cross-framework training trajectories.

The optimizer supports parameter groups, closures, dense real floating tensors,
AMSGrad, and `state_dict` restoration (including a shared optimizer iteration).
Individual missing gradients leave parameters and slots unchanged; the global
iteration advances when at least one parameter has a gradient tensor. A real
all-zero gradient tensor counts as an update. If every gradient is `None` after
closure evaluation, `step()` raises `ValueError` before changing the iteration,
parameters or slots. Step coefficients are cached only within the current call
and parameter group, keyed by actual device and dtype; schedules and subsequent
steps recompute them. Sparse gradients are rejected. Epsilon must
be positive and finite. No weight decay or clipping is implied; notebook
regularizers remain in the loss. These ReferenceAdam rules do not apply to
AdaBelief, Lion or L4Adam. AdaBelief's separate corrections are documented in
[parameter-dtype arithmetic](PYTORCH_ADABELIEF_DTYPE_CORRECTION.md),
[moment coefficients](PYTORCH_ADABELIEF_MOMENT_CORRECTION.md) and
[empty-gradient handling](PYTORCH_ADABELIEF_BOUNDARY_SUCCESSION.md).


## Native optimizers and streaming data

Pass `model.parameters()` or native parameter groups to `AdaBeliefOptimizer`,
`Lion`, or `L4Adam`. Optimizer state uses native state dictionaries. AdaBelief
tracks squared gradient prediction error; Lion uses sign momentum updates;
neither is replaced with Adam. See each optimizer's docstring for supported
options. Ordinary optimizers apply value clipping before per-tensor norm clipping;
`global_clipnorm` explicitly clips across all groups. The historical centralization
helper retains its global-norm-then-value order. Gradient centralization reduces across input dimensions, including
axis 0 for original-layout custom kernels and axis 1 for native Linear weights.

L4Adam intentionally completes the unfinished baseline. It uses bias-corrected
momentum `g` and second moment `s`, direction `v=g/(sqrt(s)+epsilon)`, and one
global `sum(g*v)` over all parameters. Initialize `L_min=gamma_0*L`, then take
`min(L_min,L)`. Apply global multiplier
`alpha*(L-gamma*L_min)/(sum(g*v)+epsilon)` and afterward forget with
`L_min *= (1+1/tau)`. The direction epsilon and denominator epsilon are separate
numerical safeguards. A closure/current loss is required; the moving minimum is
serialized. Native L-BFGS replaces TensorFlow Probability; retained historical
class names are aliases for native execution, not compatibility wrappers.

`PointWiseData` retains parameter/coordinate/target ordering, normalization, and
optional area weights. Zero ranges and invalid inputs raise explicit errors.
`TFRDataset` preserves legacy Example float-list TFRecord shards without importing
TensorFlow. Column names, targets, weights, remainder rows, and requested epoch
repetitions are retained; training consumes native DataLoaders/iterables.

## Tutorials and artifacts

The eight `*_pytorch.ipynb` notebooks preserve full default scientific workloads.
`NIF_SMOKE=1` is explicitly a separate development mode. Tutorial 6 evaluates
superresolution in `[parameter, coordinate]` order. Tutorial 8 uses
`physical_u = normalized_u * scale_u + mean_u` and derivative chain rules, so
predictions and analytic references are compared in the same physical units.
Tutorial 7 uses native pruning masks and native int8 operators, and validates
restored/exported artifacts by evaluating predictions. Its pruning controller
supports the eager tutorial workflow of one forward followed by its backward per
optimizer update. Multiple pending forward graphs followed by one combined
backward with coordinate gradients are unsupported: preforward masking changes
the kernel in place. Strip pruning controls before tracing/export. PyTorch state dictionaries
are the checkpoint format; historical Keras checkpoints are not supported.

Tutorial 7 reports relative mean-square error as the sum of squared differences
divided by the sum of squared reference values (the squared Frobenius-norm
ratio). The archived TensorFlow notebook instead used the squared matrix
spectral-norm ratio. These reported metrics are different; the native metric
corrects the earlier MSE label and does not change the training objective.
