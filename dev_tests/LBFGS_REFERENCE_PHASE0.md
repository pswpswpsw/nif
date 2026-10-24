Public CPU analytic oracle, phase 0
=================================

No product implementation changed. The demonstrated bug remains the frozen
`test_lbfgs_stopping_contract.py` actual-constructor RED: one small-scale f64
quadratic stops before its first update. A pending solver import is not a RED.

The new independent TFP oracle reads the original public wrapper options from
ad9113df14590189b12068b6b3fa7c8c8d9187d4 and full native T1 settings, then runs
only analytic vectors on CPU with one intra/inter-op thread. TensorFlow 2.21.0,
TFP 0.25.0, and Python 3.13.11 were used. Both physical/logical GPU lists were
empty and CUDA_VISIBLE_DEVICES=-1. Installed-source raw hashes and raw tensor
bytes are recorded in fixtures/lbfgs_reference_phase0/oracle.json.

The original oracle field `one_call_of_original_rounds=200` denotes the original
configured outer rounds. Each original case executed ONE fresh call, not 200.
Its bytes and the original RED remain unchanged. The new record states this
explicitly. Original full budgets remain 200 fresh calls, each max_iterations
1000, history20, gradient/x/relative-f tolerances1e-15, line-search limit100.

Measured results
----------------

* Both scalar quadratics (unit and1e-8 scale) converge in one iteration in f32
  and f64. Small-scale final gradients are5.96046444e-16 and1.11022302e-24.
* Offset quadratic `2^20 + .125/2*x^2` reaches zero gradient in both dtypes;
  it does NOT demonstrate stopping with nonzero gradient.
* Offset quartic `2^20 + .125*x^4`, starting at1, converges after2 iterations/
  8 evaluations in f32 with gradient0.00255622039549 and after7/28 in f64 with
  gradient5.13794943165e-9. Both final objective values round to2^20. This is
  the observed flat-objective/nonstationary-gradient control.
* Diagonal quadratic `.5*(x0^2+10*x1^2)`, starting at[1,1]: one iteration is
  finite, not converged, not failed. Resume to cumulative limit2 matches a fresh
  limit2 run, including raw state/history,2 iterations and7 cumulative function
  evaluations. It still is not converged or failed in either dtype.
* With relative tolerance.1 and other stopping tolerances0, adding100 to that
  quadratic converges after its first iteration; adding-100 does not. Separate
  source-helper boundary calls confirm four independent inclusive conditions,
  including signed old-f in relative tolerance and exact-zero absolute change.
* Initial positive infinity or NaN gradient fails; initial negative infinity
  converges. Unbounded linear f(x)=x: f32 reaches negative infinity and reports
  converged; f64 reports failed after2 iterations/156 evaluations, with position
  NaN and finite recorded objective. These are literal observed TFP states,
  not desirable training outcomes or a new bitwise requirement.

API and pending tests
---------------------

The uncollected `fixtures/lbfgs_reference_phase0/test_reference_lbfgs_contract.py.txt`
is frozen for activation with EXACTLY the same bytes only after implementation
authorization. It is not imported or run now; no skip/xfail was added.
`ReferenceLBFGS` uses `max_iter` ADDITIONAL iterations per step and cumulative
runtime result counters. Generic `lbfgs_minimize` retains cumulative
`max_iterations`, passing only the remaining budget to the class. Other options
are explicit in the recipe: num_correction_pairs, tolerance, x_tolerance,
f_relative_tolerance, f_absolute_tolerance, max_line_search_iterations.

Checkpoint tests use torch.save and torch.load(weights_only=True), as T1 does.
Saved optimizer state must contain only tensors/primitives, with schema/dtype/
shape validated before base-loader casting and original torchLBFGS state
rejected. Same-backend checkpoint replay equality tests state restoration; they
do not demand cross-framework trajectory equality. A trial closure exception
must restore the last accepted model parameters and propagate the exception.
Algorithmic line-search failure is different: the reference updates to the
returned LEFT endpoint even on a new line-search failure, per
TFP bfgs_utils.py:218-239. It cannot be universally rolled back.

Source provenance and limits
----------------------------

TFP bfgs_utils.py:373-391 applies independent gradient, displacement, relative-f
(signed current_objective), and absolute-f tests. Initial gradient comparison
is strict `<` (get_initial_state_args); later comparison is `<=`. History queues
update only when the next result is neither converged nor failed. Hager-Zhang
uses the actual installed public implementation; its raw hash is preserved.

Mathematical references: Hager and Zhang, SIAM J. Optimization16(1),170-192
(2005), DOI10.1137/030601880; Algorithm851, ACM TOMS32(1),113-137(2006).
These references do not establish termination or bit identity with TFP.
No third-party implementation was copied into product in this phase.

Only small CPU analytic objectives were evaluated. No NIF model imports,
notebook scientific cells, GPU, full training, installs, or scientific quality
gates were used. Backend arithmetic/overflow can differ. No full-training
cause or successful full scientific parity follows from these records.

Development evidence remains in ignored
dev_tests/artifacts/lbfgs-reference-phase0-5d81e6b3b011. The first oracle invocation
omitted TFP's required explicit initial_position=None on resume; that recipe
error and both intermediate complete outputs remain preserved, not product RED.
