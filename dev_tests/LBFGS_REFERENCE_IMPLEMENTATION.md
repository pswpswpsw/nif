Reference L-BFGS implementation, public CPU evidence
===================================================

The original actual-T1-constructor regression is unchanged (SHA25616a203...8424c).
It changed from1 failed/1 passed on the original Torch constructor to GREEN on
the explicit ReferenceLBFGS constructor. The activated23-case analytic test is
byte-identical to the phase0 recipe (2abbc4...c81917). All phase0 proof/oracle
records remain unchanged. Combined focused checks:62 passed, one sparse-tensor
construction warning, on Torch2.14.1+cpu/Python3.13.11, CUDA_VISIBLE_DEVICES=-1,
threads1/1. No NIF model training or GPU was run.

One scalar core serves ReferenceLBFGS and lbfgs_V2.lbfgs_minimize. It uses the
TFP-reference two-loop history conditions, Hager-Zhang bracket/secant-square/U3
control flow, initial strict gradient comparison and subsequent inclusive
gradient/displacement/relative-f/absolute-f criteria. Relative-f uses the SIGNED
old objective; default absolute-f0 still stops on an exactly unchanged loss.
It adds no Torch directional-derivative stopping threshold or ys>1e-10 history
filter. There is no Torch strong-Wolfe fallback. Original T1 settings remain
200 FRESH optimizers, each1000 iterations,20 pairs,1e-15 gradient/x/relative-f,
and100 line-search iterations. Model, data, objective, exports and checkpoint
workflow are unchanged by the integration.

ReferenceLBFGS.max_iter is ADDITIONAL iterations per step; result counters and
history are cumulative. Generic max_iterations is cumulative and maps to the
remaining budget. The original generic parallel_iterations positional slot is
preserved; f_absolute_tolerance is appended after it. The class accepts only
one parameter group/device/dtype, real f32/f64 contiguous parameters and dense
gradients. It rejects added groups before mutation. Runtime result namespaces
are not serialized; persisted state uses tensors/primitives and an explicit
version/dtype/device/shape schema checked BEFORE Torch loader casting. Tests
exercise torch.load(weights_only=True), resume, incompatible old Torch state,
malformed dtype/version, and closure-exception parameter restoration.

Measured reference details
--------------------------

The public edge oracle records actual installed TensorFlow2.21/TFP0.25 behavior. A
small quadratic with line-search limit1 demonstrated that TFP unconditionally
runs U3 after an exhausted, still-unbracketed expansion. Representable-point
stagnation can produce BOTH search converged=True and failed=True. The outer
optimizer classifies a failed search by not search.converged. It uses the final
LEFT endpoint, even for algorithmic failure, as the reference does. This is
different from restoring last accepted parameters after a Python closure
exception, which is explicitly tested.

The initial draft omitted that exhausted-interval U3 transition. A new direct
public search test actually failed2/2, then passed after the precise control
flow fix. An earlier development-only assumption of immediate failure after
3 evaluations was disproved by the measured oracle; its original logs and
oracle outputs remain under the ignored evidence directory. It was never a
frozen phase0 expectation. The add_param_group API case likewise recorded
1 RED before the rejection override, and the positional-slot case recorded
1 RED before the signature-only v2 fix. Frozen tests/oracles were not changed.

Twenty-four public scalar/vector CPU comparisons cover unit/small/offset
quadratics, offset quartic, cumulative diagonal-quadratic resume, positive and
negative relative-f shifts, unbounded linear/nonfinite states, and Rosenbrock.
All24 recorded flags/iteration counts/evaluation counts match actual TFP in
these runs. Finite final positions match except the f32 offset quartic, whose
maximum absolute difference is2.9802322387695312e-8. These are observations,
not cross-backend bitwise gates. New tests assert semantic contracts and
dtype-precision accuracy on the analytic Rosenbrock optimum, without asserting
TFP's trajectory/evaluation bits or invalid trial positions.

Attribution, packaging and limits
--------------------------------

The two new numerical modules honestly identify their compact scalar adaptation
of public TFP control flow. The original Apache copyright/license headers,
full TFP_LICENSE.txt, and explicit modification/source-hash attribution in
TFP_NOTICE.txt are retained. Root LICENSE/setup license were not changed.
A fresh public-source packaging stage verified both notice files are present
with exact source bytes in an sdist and wheel. This is distribution evidence,
not a general legal-compatibility conclusion.

The implementation/provenance JSON records exact source hashes, raw development
evidence hashes, canonical LF log lineage, and measured native results. Existing
historical scope protections are handled separately by an exact integration
successor, with original archives and numerical assertions preserved.

Only the unbatched real optimizer subset needed by NIF is implemented. Backend
reductions, transcendental kernels, rounding, and overflow can differ. The CPU
analytic checks establish neither CUDA execution parity nor full scientific
tutorial quality; original full-training FAILs are not reclassified here.
