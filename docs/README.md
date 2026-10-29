# NIF v2.0.0 documentation

The [online documentation](https://www.shaowupan.com/nif/) describes the native
PyTorch implementation. Sphinx imports the actual package to generate its API
reference. The build derives the quickstart and selected guides from the
repository README and existing PyTorch documentation, keeping their code examples
in one source.

Use Python 3.11+ for the documentation builder. Install the package and
documentation tools into your documentation environment:

```bash
python -m pip install -e '.[tensorboard]'
python -m pip install -r docs/requirements.txt
python scripts/build_docs.py --output dev_tests/artifacts/docs_site/html
```

The script stages generated sources beside the output directory and runs Sphinx
with warnings treated as errors. Preview the result with:

```bash
python -m http.server --directory dev_tests/artifacts/docs_site/html 8000
```

The GitHub Actions documentation workflow builds pull requests for review and
deploys successful `main` builds to GitHub Pages. Deployment uses the
`github-pages` environment; repository Pages settings must select GitHub Actions.

The original generated HTML, images, and notebooks remain in the repository.
Each new build includes the old documentation under `legacy/tensorflow/`, with
its assets and relative links. The current API reference and eight native
tutorials are linked separately from that historical archive.

[PYTORCH_API.md](PYTORCH_API.md) records the native definitions;
[PYTORCH_PORT_CONTRACT.md](PYTORCH_PORT_CONTRACT.md) is the migration contract.
