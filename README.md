# NIF: mesh-agnostic dimensionality reduction

NIF is a mesh-agnostic dimensionality reduction paradigm for parametric spatial temporal fields. For decades, dimensionality reduction (e.g., proper orthogonal decomposition, convolutional autoencoders) has been the very first step in reduced-order modeling of any large-scale spatial-temporal dynamics. Unfortunately, these frameworks are either not extendable to realistic industry scenario, e.g., adaptive mesh refinement, or cannot preceed nonlinear operations without resorting to lossy interpolation on a uniform grid. 

NIF is built on top of Keras. 

**Documentation** 

**Papers**

## Features

- built on top of tensorflow 2 with Keras, hassle-free for many up-to-date advanced concepts and features
- distributed learning: data parallelism across multiple GPUs on a single node
- flexible training schedule: e.g., first Adam then fine-tunning with L-BFGS
- performance monitoring: model weights checkpoints and restoration

## How to cite

If you find NIF is helpful to you, you can cite our [paper](https://arxiv.org/abs/2204.03216) in the following bibtex format

```
@misc{pan2022neural,
      title={Neural Implicit Flow: a mesh-agnostic dimensionality reduction paradigm of spatio-temporal data}, 
      author={Shaowu Pan and Steven L. Brunton and J. Nathan Kutz},
      year={2022},
      eprint={2204.03216},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## License

[LGPL-2.1 License](https://github.com/pswpswpsw/nif/blob/master/LICENSE)
