# Public Git checkout EOL correction

Base: `5dcba14b36829376ed7e1276a5e533b87d4820b8`.

Four exact `.gitattributes` declarations keep `tutorial/README.md` at LF and
preserve three historical patch records byte for byte. No package, notebook,
data, numerical assertion, historical hash manifest, or existing evidence is
changed. The attributes guard strips only that exact four-rule suffix and
compares the remaining bytes with the original recorded hash. The Windows scope
helper permits only the corresponding exact moment-scope transformation.

The prior Windows helper is archived once under
`fixtures/publication_checkout_v3/`; its digest matches the unchanged v2
manifest. `publication_checkout_v3_scope.md` records the complete three-file
patch. `publication_checkout_v3_manifest.json` records before/after hashes and
evidence hashes. The new successor test freezes the exact helper and checks
original/extra/missing attribute-rule rejection, substantive README rejection,
and all three historical patch hashes.

## Verification

- **RED:** `publication_checkout_v3_red_native.log`: the new Git regression
  failed, showing four checkout byte deltas and all three originally reported
  scope failures; its archive check passed. No fix was present.
- **GREEN:** `publication_checkout_v3_green.log`: the same unchanged regression
  file passed both tests after the fix.
- **Scope controls:** `publication_checkout_v3_successor_green.log`: 8 passed.
- **Full actual Windows checkout:** `publication_checkout_v3_full.log`:
  **424 passed, 4 skipped**, 576 warnings, 63.88 seconds. The skips are two
  recorded-Linux-build RNG fingerprints and two unavailable CUDA checks.
- **LF checkout policy:** `publication_checkout_v3_lf_green.log`: 11 passed
  with `core.autocrlf=false` on Windows. This checks LF checkout behavior; it
  does not claim a native Linux runtime run.
- `git diff --check` passed for the three modified source files.
  `git diff HEAD --name-only -- nif tutorial tests` was empty. The recorded
  unified-diff document retains its original literal blank context lines.

Native interpreter:
`C:\Users\pswpe\.cache\nif-pytorch-evaluation\env\Scripts\python.exe`.
The reproducer is `python -m pytest dev_tests/test_publication_checkout_eol.py -q
--basetemp <fresh-path>`. The full run used `python -m pytest dev_tests -q -rs
--basetemp <fresh-path>` from the checkout recorded in
`publication_checkout_v3_full_checkout.json`.

These are real local Git clones with `core.autocrlf=true`, followed by
`read-tree HEAD`, `hash-object --no-filters`, and `update-index --cacheinfo` for
the six explicit candidate files, then `checkout-index --all --force`.
This exercises Git's actual index-to-worktree EOL conversion without committing
or modifying the main repository's Git metadata. Fresh clones remain under the
ignored `dev_tests/artifacts/` namespace. The full suite includes the new Git
regression, which creates another fresh clone and runs the three original gates.

The initial `publication_checkout_v3_red.log` records a sandbox-only MSYS access
failure before cloning. Native execution was then authorized; the substantive
RED, GREEN, and full results above used that execution mode. No Docker, WSL,
GPU, commit, push, or private evaluator material was used.

## Publication text serialization

Seven new log/JSON outputs initially contained CRLF bytes. The existing Git LF
attributes normalized them when staged, so their published bytes differed from
the first A9 manifest. An isolated Git-index reproducer confirmed 8 matching
hashes and exactly 7 mismatches before correction. The seven published copies
now use LF; each is exactly its raw counterpart with `CRLF` replaced by `LF`.
Their text and recorded test results are unchanged. The same isolated index
check then passed all 15 manifest hashes; all 16 staged files matched disk bytes.

The original raw copies, original hashes, and the prior A9 manifest/note remain
under `dev_tests/artifacts/checkout-v3-serialization-x59lzp01/raw/`. The current
manifest records each raw-to-canonical hash, byte count, CRLF count, and the
isolated Git serialization check. The source/helper/test hashes and the raw
unified-diff proof are unchanged. This serialization correction does not rerun
or revise the scientific test results above.
