# Experiment entry points

The portable self-scaled comparison lives in
[examples/self_scaled](../examples/self_scaled/README.md), with
[Tutorial 9](../tutorial/9_self_scaled_optimizers_pytorch.ipynb) as its Colab
entry point. The [experiment guide](../docs/SELF_SCALED_OPTIMIZERS.md) is the
shared source for methods, recorded results and interpretation.

## Historical local runs

Some maintainer checkouts retain `tutorial1_self_scaled`,
`tutorial2_self_scaled`, `tutorial3_self_scaled`, `tutorial5_self_scaled`
and `tutorial6_self_scaled` below this directory. Those local sources,
checkpoints and reports are frozen records, with distinct workloads and
cross-folder dependencies. They are outside the published experiment API.

The guide links their public result summary and source hashes. Read that
summary for the comparison; use the original local folders to inspect a
particular recorded run. Their cached-result notebooks and the portable
runner have different defaults.

## New experiments

Reuse the portable runner's parameter packing, objective recorder and verified
author-source loader. Keep case-specific data, normalization, objective,
training budget and plotting definitions explicit. Save each run to a new
directory and generate tables and figures from its saved arrays and JSON.

Add new result summaries to the guide with links to the run's provenance;
retain earlier records under their original identifiers.
