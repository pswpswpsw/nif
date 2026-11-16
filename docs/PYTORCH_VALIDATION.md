# Native PyTorch validation

The native PyTorch implementation is released as **v2.0.0**. The user completed
reviews of all eight released tutorials on 2026-10-25–27. Their current
[comparison summary](PYTORCH_TENSORFLOW_PARITY.md) is the authoritative index of
acceptance decisions, measured errors and automatic criteria that remain unmet.

## Evidence by scope

- **Package behavior:** the [API guide](PYTORCH_API.md) defines ordering, derivatives, regularization, optimizer arithmetic and artifact support. Current public regression checks run through [GitHub Actions](https://github.com/pswpswpsw/nif/actions).
- **Tutorial execution:** the [dated execution snapshot](archive/2026-11-15/PYTORCH_VALIDATION.md#full-tutorial-execution) preserves the original full workloads, checkpoints, package-acceptance counts and visual inspection records.
- **Numerical comparisons and hosted runs:** the [comparison summary](PYTORCH_TENSORFLOW_PARITY.md) links each tutorial's measured results and complete history.
- **Distributed training and monitoring:** the [training example record](PYTORCH_TRAINING_EXAMPLE.md) states the exercised configurations and export/checkpoint coverage.
- **Experimental self-scaled optimizers:** the [experiment guide](SELF_SCALED_OPTIMIZERS.md) separates historical measurements, fresh FULL runs and shortened execution checks.

Full execution means the declared workload and its output checks completed.
Numerical comparisons quantify the agreement of outputs, derivatives or trained
fields under their recorded controls. User acceptance records the review of
those results. Each report keeps these scopes and its source revision explicit.

## Dated records

The original 2026-10-22 report, including its then-current branches and open work,
is retained byte-for-byte in the [archive](archive/2026-11-15/PYTORCH_VALIDATION.md).
Later results are indexed by the comparison summary above. The
[archive manifest](archive/2026-11-15/manifest.json) records the original source
commit and checksums; historical test counts apply to the versions named there.
The [provenance audit guide](../dev_tests/provenance/README.md) describes how to
reconstruct the frozen historical test sources and run their original guards.
