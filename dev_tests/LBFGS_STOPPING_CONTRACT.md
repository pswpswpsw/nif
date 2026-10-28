# T1 L-BFGS stopping-contract reproducer (RED; no implementation change)

The new test reads the actual full-mode settings and optimizer constructor from tutorial/1_simple_1d_wave_pytorch.ipynb. It executes only that setup and constructor, supplies one CPU float64 parameter through parameters(), and minimizes f(x)=0.5*scale*x^2 from x=1. No NIF model/data, scientific notebook cell or training run is used. It requires a strict decrease on this chosen analytic objective, without requiring matching solver trajectories, iteration counts or parameter bits.

| scale | Native actual T1 constructor | Original TFP oracle |
|---|---|---|
| 1 | x=0, f=0, gradient=0; 2 closure evaluations | x=0, f=0, gradient=0; converged; 1 iteration, 3 evaluations |
| 1e-8 | x=1, f=5e-9, gradient=1e-8; 1 closure evaluation; no step | x=1.1102230246251565e-16, f=6.162975822039155e-41, gradient=1.1102230246251566e-24; converged, not failed; 1 iteration, 15 evaluations |

Native RED: 1 failed, 1 passed. The small-scale gradient remains above the shared 1e-15 gradient threshold. In installed torch/optim/lbfgs.py:374-375 the first direction is -g; :438-442 stops before line search when g dot d > -tolerance_change. Here g dot d is approximately -1e-16 and tolerance_change=1e-15. This is an observed additional stopping rule relative to the original TFP contract on this objective. It is not evidence about the cause of full T1 training outcomes.

The TFP process reads keyword settings from the preserved public ad9113df14590189b12068b6b3fa7c8c8d9187d4:nif/optimizers/lbfgs.py:105-119. It uses the actual installed tfp.optimizer.lbfgs_minimize, not a hand-written solver. The public wrapper requests 20 correction pairs, gradient/x/relative-f tolerances 1e-15, max_iterations=1000, max_line_search_iterations=100, parallel_iterations=1; default f_absolute_tolerance=0. Installed TFP bfgs_utils.py:373-391 uses gradient, actual displacement, relative objective and absolute objective conditions; its Hager-Zhang line search is a different algorithm from native strong Wolfe. This experiment is one fresh call, not all 200 outer rounds.

Versions: Python 3.13.11; Torch 2.14.1+cpu, CUDA build None, GPU count 0; TF 2.21.0 and TFP 0.25.0 with physical/logical GPU lists empty. Both independent processes use CUDA_VISIBLE_DEVICES=-1 and one CPU thread; TF_USE_LEGACY_KERAS=1 is set before TF imports. No installations, network, scientific source edits, threshold changes or solver-option experiments were performed.

Reproduce the native RED with the existing native interpreter: python -B -m pytest dev_tests/test_lbfgs_stopping_contract.py -q -s -p no:cacheprovider, with CUDA_VISIBLE_DEVICES=-1, OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1. The independent oracle recipe is preserved verbatim as dev_tests/fixtures/lbfgs_stopping_contract/tfp_oracle.py.txt and can be executed with the existing TF interpreter plus an output JSON path. It reads only the public repo and installed libraries. The measured oracle is lbfgs_stopping_contract_oracle.json; source locations, constructor/settings hashes and serialization lineage are in lbfgs_stopping_contract_source.json. The new public RED log is canonical LF; its original CRLF bytes are preserved under this run's ignored evidence directory.

Status: reproducer and oracle only. Product, notebooks, existing tests, thresholds and prior archives remain frozen. Any implementation work requires a separate instruction; this test should remain unchanged across that work.
