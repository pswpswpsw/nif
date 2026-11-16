# Development checks

Run the current PyTorch regression suite from the repository root:

```bash
python -m pytest dev_tests -q
```

The suite covers current model and optimizer behavior, derivatives, checkpoint
continuation, pruning, tutorial structure, documentation and release metadata.
Short notebook executions are separate from the full scientific comparison
records.

[Historical integrity audits](provenance/README.md) preserve the migration-era
scope guards, source snapshots and RED/GREEN evidence in a versioned archive.
They run against their frozen source tree rather than constraining future
implementation changes. The original TensorFlow checks have their own
[environment and instructions](../tests/README.md).
