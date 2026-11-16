# Historical integrity audit

Current numerical, model, optimizer, tutorial and packaging regressions live in
`dev_tests/test_*.py` and run with:

```bash
python -m pytest dev_tests -q
```

The migration-era scope guards and their mutation controls run separately:

```bash
python -m dev_tests.provenance audit --output dev_tests/artifacts/history-001
```

Use a new output directory for each audit. The command reconstructs the public
tree at commit `73b2ffcf71d9f0d019bd86d62cbf80989e9f7eac`, then executes the
original historical tests in an isolated process. It needs the test dependencies,
but no Git executable, network access, TensorFlow runtime or private evaluator
files. Its output includes the full pytest logs, JUnit report and `report.json`.
Tests requiring an actual Git checkout retain their original skip condition in
this no-Git view. The legacy RNG fingerprint control also retains its recorded
platform/build requirement. The report counts these skips separately from
executed passes; on the Windows validation run, 285 tests passed and these two
controls were skipped.

## What is retained

`baseline-73b2ffc.json` maps all 606 original public paths to SHA256 digests.
`baseline-73b2ffc.zip` contains one compressed object per distinct source or
evidence payload. This stores 558 logical files as 545 objects and removes
247,886 bytes of duplicate uncompressed evidence. The compressed bundle is about
2.6 MB. Its manifest and object hashes are verified before historical source is
used.

The other 48 entries are the unchanged original notebooks, figures, fonts and
datasets already present in the repository. They are hash-checked when building
the isolated view, avoiding a second copy of the large original outputs.
Original notebook text accepts only its exact LF or uniform CRLF checkout form;
mixed line endings and content changes are rejected. Live files are never
rewritten by this reader.

The 16 historical TensorFlow notebooks now live under
`tutorial/legacy/tensorflow/`. A fixed relocation map reads those exact backing
files while retaining their original `tutorial/<filename>.ipynb` logical paths
inside the frozen projection. The baseline manifest, ZIP objects, content
digests and original line-ending rules remain unchanged. Other backing assets
keep their recorded locations.

`migration.json` indexes the flat historical paths moved into the bundle. They
include the old scope helpers, scope-only tests, logs, manifests, duplicate
fixtures, editor state and Sphinx caches. The original source, numerical
expectations, error tolerances, failure logs and manifests remain recoverable at
their original logical paths. Extract selected files for inspection with:

```bash
python -m dev_tests.provenance extract --output dev_tests/artifacts/old-sources \
  dev_tests/adabelief_moment_scope.py dev_tests/adabelief_moment_red.log
```

Current tests retain the original initializer/model fingerprints, the uncached
ReferenceAdam control, and the historical AdaBelief control needed by current
RNG checks. Three formerly mixed test modules were split by purpose: their
current scientific and UTF-8 checks remain live, while source-scope replay uses
the original complete modules in the frozen view.

## Rebuilding the archive

`python dev_tests/provenance/build.py` reads raw Git blobs from the fixed commit.
It is a maintenance tool; ordinary audits never invoke Git. The generated
manifest digest is pinned in the reader, so changing the archive requires an
explicit reviewed update. New current-code changes do not extend a chain of
historical source substitutions.
