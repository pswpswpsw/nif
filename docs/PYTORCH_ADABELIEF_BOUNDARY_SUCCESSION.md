# AdaBelief empty-gradient boundary and public test succession

This follow-up preserves the parameter-dtype scalar arithmetic correction.
After closure evaluation, native AdaBelief rejects a call with no gradient
tensor across any group by raising `ValueError`. The guard runs before gradient
preparation, schedule evaluation, counters, slots or parameters can mutate.

A real all-zero gradient tensor counts as an update. Partial missing gradients
leave the corresponding parameters/slots unchanged, while all fixed existing
groups advance their existing per-group clocks. Late-added groups retain their
native independent clock and start at step one. This is not a global-clock
redesign. Dense/sparse arithmetic, scalar caching, optimizer options and other
optimizers remain unchanged.

The earlier scalar phase reported two public-test conflicts. The coordinator
explicitly authorized their succession; the original sources, original expected
JSON and raw two-failure log remain verbatim under
`dev_tests/fixtures/adabelief_boundary_v3/` with SHA256 provenance.

- The collected `test_current_initializers_rng_and_historical_adabelief_trajectories_v2`
  checks current initializer/RNG and model initialization against every original
  expected hash, then replays historical training with the frozen PRE-scalar
  optimizer. The original callable name remains a non-collected compatibility
  wrapper for the existing build-portability check; all its assertions run in
  the explicitly named collected v2 successor. No expected values changed.
- Current corrected optimizer math remains covered by the unchanged, frozen
  source-derived `test_adabelief_scalar_dtype.py` regressions. Historical model
  fingerprints are not substituted for those current-math checks.
- The phase-v3 scope guard reconstructs only exact approved edits from archived
  sources: the atomic guard/documentation, historical test binding/versioning,
  the diagnostic's archived-source read, and explicit UTF-8 helper reads. It
  retains all other notebook, source, data, frozen numerical expectations and
  `.gitattributes` fingerprints. Negative controls reject unrelated model and
  optimizer-math edits.

Both `tutorial1_gc_scope.py` and `test_tutorial1_gc_scope.py` now read their
JSON/source/log text explicitly as UTF-8. Controlled GBK-default-reader tests
reproduced the previous decoding failures. Archive bytes were not rewritten.
The new phase helper also passes that simulation; no native Windows execution
is claimed.

Evidence and changed-file hashes are recorded in
[the candidate report](PYTORCH_CANDIDATE_REPORT.md) and
[the follow-up manifest](../dev_tests/adabelief_boundary_candidate_manifest.json).
The full PUBLIC suite passes; private evaluation and full scientific training
criteria are separate and unchanged. No GPU or cross-framework parity claim is
made.
