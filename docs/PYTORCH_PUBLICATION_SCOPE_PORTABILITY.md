# Public scope portability — publication v1

The AdaBelief v4 scope helper previously required 439 ignored local outputs under
**exactly `dev_tests/artifacts/`**. A clean public projection therefore failed on
`dev_tests/artifacts/adabelief_boundary/public_suite.junit.xml`. These generated
notebooks, images, checkpoints, TFRecords, wheel and JUnit files are not public
prerequisites and are not added for publication.

The only preexisting implementation file changed is
`dev_tests/adabelief_moment_scope.py`: two docstring lines and an exact-prefix
exclusion in its historical fingerprint loop. All existing public-file hash
checks and previously authorized exact source transformations remain unchanged.
There is no generic missing-file fallback, extension filter or ignore-rule lookup
in the runtime scope helper. The original helper is archived verbatim with its
SHA-256 under `dev_tests/fixtures/publication_scope_v1/`; the new exact-transition
test ties that archive to the unchanged moment-phase candidate manifest.

`dev_tests/publication_projection.py` supplies the new test-only projection
builder and a separate **optional local** integrity audit. The latter checks the
original hashes of present files in that one namespace and reports absences.
Here, all 439 local historical outputs still match. In the clean projection,
all 439 are reported absent. Neither their original hashes nor any old manifest,
fixture, numeric golden, raw historical log or scientific result was rewritten.

## Evidence and clean projection

- True BEFORE-fix RED: **6 failed, 5 passed**, retained in
  `dev_tests/publication_v1_scope_red.log`. The missing JUnit file also masked
  several preservation negative checks. Tests, inventory, projection helper and
  logs were frozen in `dev_tests/publication_v1_red_fingerprints.json` before the
  helper edit.
- Same tests GREEN: **11 passed**, `dev_tests/publication_v1_green.log`.
  Missing public fixtures/provenance, changed model code, changed historical
  goldens/archives, and changed present local artifacts are rejected.
- Full public suite in a fresh public-only tree: **401 passed, 2 skipped**
  (CUDA unavailable), **571 warnings**, `dev_tests/publication_v1_clean_full.log`.
  This is regression-test execution, not a notebook or scientific training run.
- An initial invocation failed during temporary-directory setup (**10 errors**),
  before exercising the helper. Its unmodified log is retained as
  `dev_tests/publication_v1_red.log`; it is not counted as the bug reproduction.

The full-suite projection contains **352 files**, copied exactly from a
read-only `git ls-files --cached --others --exclude-standard -z` inventory. It
includes all **297 non-artifact paths** from the unchanged 736-entry historical
manifest. Composition and per-file SHA-256 values are recorded in
`dev_tests/publication_v1_projection.json`. No `.git` or `dev_tests/artifacts`
tree was copied. Git was used only to construct this outer projection. Runtime
checks use ordinary files and an explicit inventory; no Git executable is
required. The subprocess used absolute Python with `-I`, an empty `PATH`, CPU
only, and verified that `nif`, the scope helper and projection helper were
imported from the projected tree. Pytest ran the complete `dev_tests` collection
with its cache disabled and temporary/JUnit output directed outside that tree.

All generated runs use `dev_tests/artifacts/publication_v1/`; older generated
outputs remain untouched. The frozen 348-entry inventory used by the focused
regressions is a runtime-independent public input list. The later 352-file
full-suite inventory also includes evidence created after RED. Final reporting
files were written after execution and are not runtime dependencies.

## Preservation and limitations

Package code, notebooks/generator, original TensorFlow artifacts, data,
dependencies, `.gitattributes`, scientific tests and numeric expectations retain
their prior bytes. A concurrent coordinator edit to
`docs/PYTORCH_TENSORFLOW_PARITY.md` was observed and left untouched. New files are
limited to public portability tests/helpers, their archived helper, inventories,
raw logs, this report and the source manifest.

`dev_tests/publication_v1_manifest.json` records exact before/after hashes,
new public files, optional generated evidence and preservation results. This
establishes public-test portability in this CPU environment. It does not establish
full scientific parity, completion of all tutorials or independent acceptance.
No notebook workloads, GPU, TensorFlow runtime, dependency installation, Git
mutation or GitHub sync were performed.
