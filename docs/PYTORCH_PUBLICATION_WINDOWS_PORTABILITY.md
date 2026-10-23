# Public-test Windows portability — publication v2

This phase addresses the six sanitized native-Windows test failures supplied by
the coordinator. No evaluator files were accessed. Local reproduction used
Windows path semantics and a simulated GBK default reader on Linux; **native
Windows GREEN remains a separate independent check**.

Three narrowly scoped test fixes:

- `test_pruning_adam_state.py`: use `relative_to(root).as_posix()` for immutable
  POSIX fingerprint keys. The original hashes and notebook checks are unchanged.
- `test_tutorial1_gc_behavior.py`: explicitly decode the unchanged generator as
  UTF-8. Both primary-default and optional-GC source consumers are covered.
- `test_publication_scope.py`: serialize `FileNotFoundError.filename` as a
  root-relative POSIX identity, separately from its display string. Require that
  identity to equal the exact requested public filename; the exception-type
  assertion remains. A different filename is rejected.

Five existing test/helper files additionally receive bounded scope/projection
wiring: `adabelief_moment_scope.py`, `publication_projection.py`,
`test_adabelief_boundary_succession.py`, `test_publication_transition.py`, and
`test_tutorial1_gc_scope.py`. The original eight sources are archived verbatim in
`dev_tests/fixtures/publication_windows_v2/`. Their hashes are anchored to the
unchanged publication-v1 manifests. The new `publication_windows_scope.py`
permits only the exact documented substitutions. V1's original helper transition
still executes against its archived source; the current successor is checked
separately. The complete before/after diff is
`dev_tests/publication_windows_v2_scope.diff`.

The new projection inventory is versioned; v1's inventory, raw RED/GREEN and
setup-error logs, manifests, fixture bytes and numeric expectations are intact.
Runtime checking still needs no Git. The sole optional generated-output namespace
remains exactly `dev_tests/artifacts/`; no required public fixture/provenance
became optional. Added negative controls reject widened namespace exclusions,
weakened filename assertions, altered numerical assertions, and a missing new
public archive.

## Recorded evidence

| Check | Local result | Raw log |
|---|---|---|
| BEFORE-fix Windows-path/GBK reproduction | **6 failed, 3 passed** | `publication_windows_v2_red.log` |
| Same frozen regressions after three fixes | **9 passed** | `publication_windows_v2_green.log` |
| Old scope guards before explicit succession | **4 failed, 2 passed, 4 deselected** | `publication_windows_v2_legacy_scope_red.log` |
| Focused portability, scope and preservation checks | **40 passed** | `publication_windows_v2_scope_green.log` |
| Entire public suite in fresh projection | **416 passed, 2 skipped**, 571 warnings | `publication_windows_v2_clean_full.log` |

All logs are under `dev_tests/`. The reproduction invokes the actual fingerprint
guard, generator reader, missing-file assertion and child reporting script. Its
source and original RED log remain at their BEFORE-fix hashes. No numerical
expectation or tolerance was changed.

The full-suite projection contains **377 public files**, including all 297
non-artifact paths from the unchanged historical manifest. It contains no `.git`
or historical `dev_tests/artifacts` tree. A read-only Git inventory constructs the
projection; absolute Python `-I` with an empty `PATH` runs the suite. Import checks
verify that package, helper and target-test modules resolve inside the copied
tree. Composition/hashes and test import origins are recorded in
`publication_windows_v2_projection.json` and
`publication_windows_v2_import_origins.json`. New generated outputs use only
`dev_tests/artifacts/publication_windows_v2/`; all 439 older local artifact hashes
still match. Later report/manifest files are evidence, not runtime inputs.

Package/scientific source, notebooks/generator, original TF artifacts, data,
`.gitattributes`, dependencies and recorded scientific results are unchanged.
A concurrent coordinator parity-document edit was observed and left untouched.
No notebook/scientific training workloads, GPU, TensorFlow runtime, installation,
Git mutation or publication were performed. The CPU regression suite is the
only execution workload. No full-training parity or independent acceptance is
claimed. `dev_tests/publication_windows_v2_manifest.json` records exact source
hashes, allowed transformations, provenance and these limitations.
