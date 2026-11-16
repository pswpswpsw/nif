# Archived TensorFlow validation

This directory retains the TensorFlow/legacy-Keras tutorial compatibility
checks. Their original and Colab-compatible notebooks live in
[`tutorial/legacy/tensorflow/`](../tutorial/legacy/tensorflow/README.md).
Colab is the execution platform; these archived notebooks still use TensorFlow.
The current PyTorch implementation uses `dev_tests/` for default
pytest discovery; the independent blind acceptance suite is kept outside the
repository.

Run these archived checks explicitly with a separate TensorFlow environment:

```bash
python -m unittest discover -s tests
```

Several archived tests execute the legacy notebook bootstrap, which installs
its pinned TensorFlow dependencies in the running interpreter. Use an isolated
environment rather than the native PyTorch training environment. The tests
and original TensorFlow notebooks remain available as historical references.
