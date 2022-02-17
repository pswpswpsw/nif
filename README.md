# NIF: mesh-agnostic dimensionality reduction

NIF is a mesh-agnostic dimensionality reduction paradigm for parametric spatial temporal fields. For decades, dimensionality reduction (e.g., proper orthogonal decomposition, convolutional autoencoders) has been the very first step in reduced-order modeling of any large-scale spatial-temporal dynamics. Unfortunately, these frameworks are either not extendable to realistic industry scenario, e.g., adaptive mesh refinement, or cannot preceed nonlinear operations without resorting to lossy interpolation on a uniform grid. 

NIF currently only supports Tensorflow 2.x. 

**Documentation** 

**Papers**

## Features

- distributed learning: data parallelism across multiple GPUs on a single node.
- flexible training schedule: e.g., first Adam then fine-tunning with L-BFGS
- performance monitoring: model checkpoints and restoration

## How to cite

If you find NIF is helpful to you, you can cite the following

```
some citation
```

## License

[LGPL-2.1 License](https://github.com/pswpswpsw/nif/blob/master/LICENSE)
