# Agent A development checks

These are implementation-owned tests derived from the public contract and
baseline mathematics. They are separate from the frozen blind acceptance suite;
they are not Agent B's tests and contain no independent fixtures.

```bash
python -m pip install -e '.[dev]'
python -m pytest dev_tests -q
python dev_tests/run_native_notebooks.py
python dev_tests/run_native_notebooks.py --full --numbers 4,5
```

The default notebook runner explicitly enables `NIF_SMOKE=1`, starts a clean
kernel for each notebook, and keeps execution outputs under
`dev_tests/artifacts/smoke/`. `--full` disables smoke and stores outputs separately
under `dev_tests/artifacts/full/`. The notebooks themselves default to FULL.
Large binary artifacts and generated TFRecord data are ignored by git. The
candidate report and validation summary record the runs actually completed.

`model_red.log` records the initial native import failure before implementation.
Other RED checks were run in the implementation session before targeted fixes:
normalization zero ranges; missing native notebooks; mixed-precision subnet
reconstruction; dtype-preserving config restore; GlorotNormal and toy Dense
initialization; Lion inverse-time decay; and ordinary optimizer clipnorm axes.
The meaningful numerical checks cover model layout/residual math, parameter
sharing, finite training, initialization distributions, differentiable selected
Jacobians/Hessians, explicit regularization reductions, optimizer update formulas
and resume, genuine TFRecord interoperability and CRCs, and native L-BFGS.

CUDA checks are skipped when CUDA is unavailable. Smoke convergence is not an
acceptance criterion, and full GPU scientific workloads remain Agent B's job.
