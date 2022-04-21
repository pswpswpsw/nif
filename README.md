# Neural Implicit Flow (NIF): mesh-agnostic dimensionality reduction

- NIF is a mesh-agnostic dimensionality reduction paradigm for parametric spatial temporal fields. For decades, dimensionality reduction (e.g., proper orthogonal decomposition, convolutional autoencoders) has been the very first step in reduced-order modeling of any large-scale spatial-temporal dynamics. 

- Unfortunately, these frameworks are either not extendable to realistic industry scenario, e.g., adaptive mesh refinement, or cannot preceed nonlinear operations without resorting to lossy interpolation on a uniform grid. Details can be found in our [paper](https://arxiv.org/pdf/2204.03216.pdf).

- NIF is built on top of Keras, in order to minimize user's efforts in using the code and maximize the existing functionality in Keras. 

## Features

- built on top of tensorflow 2 with Keras, hassle-free for many up-to-date advanced concepts and features
- distributed learning: data parallelism across multiple GPUs on a single node
- flexible training schedule: e.g., first Adam then fine-tunning with L-BFGS
- performance monitoring: model weights checkpoints and restoration

## Google Colab Tutorial

1. [Hello world! A simple fitting on 1D travelling wave](https://colab.research.google.com/drive/1odreAOvjlkLVJsHXDMOjqc_CTaXil_ha?usp=sharing)
	- learn how to use class `nif.NIF`
	- model checkpoints/restoration
	- mixed precision training
	- L-BFGS fine tuning

2. [Tackling multi-scale data](https://colab.research.google.com/drive/1GtbgfmfkwyZnNJFrz-zf_qeY-25-eKMY?usp=sharing)
    - learn how to use class `nif.NIFMultiScale`
    - demonstrate the effectiveness of learning high frequency data

3. [Learning linear representation](https://colab.research.google.com/drive/19TTaMhddNpFIA2f-ogosLtE0l0wvDKk3?usp=sharing)
	- learn how ot use class `nif.NIFMultiScaleLastLayerParameterized`
	- demonstrate on a (shortened) flow over a cylinder case from AMR solver

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
