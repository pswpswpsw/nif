# Neural Implicit Flow (NIF): mesh-agnostic dimensionality reduction

<p align="center">
  <img src="./misc/myimage.gif" alt="animated" />
</p>

- NIF is a mesh-agnostic dimensionality reduction paradigm for parametric spatial temporal fields. For decades, dimensionality reduction (e.g., proper orthogonal decomposition, convolutional autoencoders) has been the very first step in reduced-order modeling of any large-scale spatial-temporal dynamics. 

- Unfortunately, these frameworks are either not extendable to realistic industry scenario, e.g., adaptive mesh refinement, or cannot preceed nonlinear operations without resorting to lossy interpolation on a uniform grid. Details can be found in our [paper](https://arxiv.org/pdf/2204.03216.pdf).

- NIF is built on top of Keras, in order to minimize user's efforts in using the code and maximize the existing functionality in Keras. 


## Highlights

- Built on top of **Tensorflow 2.x** with **Keras model subclassing**, hassle-free for many up-to-date advanced concepts and features

    ```python
    from nif import NIF
    
    # set up the configurations, loading dataset, etc...
    
    model_ori = nif.NIF(...)
    model_opt = model_ori.build()
    
    model_opt.compile(optimizer, loss='mse')
    model_opt.fit(...)
    
    model_opt.predict(...)
    ```

- **Distributed learning**: data parallelism across multiple GPUs on a single node

    ```python
    enable_multi_gpu = True
    cm = tf.distribute.MirroredStrategy().scope() if enable_multi_gpu else contextlib.nullcontext()
    with cm:
        
        # ...
        model.fit(...)
        # ...
    ```

- Flexible training schedule: e.g., first some standard optimizer (e.g., Adam) then **fine-tunning with L-BFGS**
  
    ```python
    from nif.optimizers import TFPLBFGS
    
    # load previous model
    new_model_ori = nif.NIF(cfg_shape_net, cfg_parameter_net, mixed_policy)
    new_model.load_weights(...)
    
    # prepare the dataset
    data_feature = ... #
    data_label = ... # 
    
    # fine tune with L-BFGS
    loss_fun = tf.keras.losses.MeanSquaredError()
    fine_tuner = TFPLBFGS(new_model, loss_fun, data_feature, data_label, display_epoch=10)
    fine_tuner.minimize(rounds=200, max_iter=1000)
    new_model.save_weights("./fine-tuned/ckpt")
    ```
  
- Templates for many useful customized callbacks

   ```python
   # setting up the model
   # ...

   # - tensorboard
   tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./tb-logs", update_freq='epoch')

   # - printing, model save checkpoints etc.
   class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
   # ....

   # - learning rate schedule
   def scheduler(epoch, lr):
       if epoch < 1000:
           return lr
       else:
           return 1e-4
   scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

   # - collecting callbacks into model.fit(...)

   callbacks = [tensorboard_callback, LossAndErrorPrintingCallback(), scheduler_callback]
   model_opt.fit(train_dataset, epochs=nepoch, batch_size=batch_size,
             shuffle=False, verbose=0, callbacks=callbacks)
   ```
   
- Simple extraction of subnetworks

    ```python
    model_ori = nif.NIF(...)
    
    # ....
    
    # extract latent space encoder network
    model_p_to_lr = model_ori.model_p_to_lr()
    lr_pred = model_p_to_lr.predict(...)
    
    # extract latent-to-weight network: from latent representation to weights and biase of shapenet
    model_lr_to_w = model_ori.model_lr_to_w()
    w_pred = model_lr_to_w.predict(...)
    
    # extract shapenet: inputs are weights and spatial coordinates, output is the field of interests
    model_x_to_u_given_w = model_ori.model_x_to_u_given_w()
    u_pred = model_x_to_u_given_w.predict(...)
    ```

- Get input-output Jacobian or Hessian.
    ```python
    model = ... # your keras.Model
    x = ... # your dataset
    # define both the indices of target and source 
    
    x_index = [0,1,2,3]
    y_index = [0,1,2,3,4]
    
    # wrap up keras.Model using JacobianLayer 
    from nif.layers import JacobianLayer
    y_and_dydx_layer = JacobianLayer(model, y_index, x_index)
    
    y, dydx = y_and_dydx_layer(x)
    
    model_with_jacobian = Model([x], [y, dydx])
    
    # wrap up keras.Model using HessianLayer
    from nif.layers import HessianLayer
    y_and_dydx_and_dy2dx2_layer = HessianLayer(model, y_index, x_index)
    
    y, dydx, dy2dx2 = y_and_dydx_and_dy2dx2_layer(x)
    
    model_with_jacobian_and_hessian = Model([x], [y, dydx, dy2dx2])
    
    ```

- Data normalization for multi-scale problem

    - just simply feed `n_para`: number of parameters, `n_x`: input dimension of shapenet, `n_target`: output dimension of shapenet, and `raw_data`: numpy array with shape = `(number of pointwise data points, number of features, target, coordinates, etc.)`

    ```python
    from nif.data import PointWiseData
    data_n, mean, std = PointWiseData.minmax_normalize(raw_data=data, n_para=1, n_x=3, n_target=1) 
    ```

## Google Colab Tutorial

1. **Hello world! A simple fitting on 1D travelling wave** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pswpswpsw/nif/blob/master/tutorial/1_simple_1d_wave.ipynb)
	- learn how to use class `nif.NIF`
	- model checkpoints/restoration
	- mixed precision training
	- L-BFGS fine tuning

2. **Tackling multi-scale data** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pswpswpsw/nif/blob/master/tutorial/2_multi_scale_NIF.ipynb)

    - learn how to use class `nif.NIFMultiScale`
    - demonstrate the effectiveness of learning high frequency data

3. **Learning linear representation** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pswpswpsw/nif/blob/master/tutorial/3_multi_scale_linear_NIF.ipynb)
	- learn how to use class `nif.NIFMultiScaleLastLayerParameterized`
	- demonstrate on a (shortened) flow over a cylinder case from an AMR solver

4. **Getting input-output derivatives is super easy** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pswpswpsw/nif/blob/master/tutorial/4_get_gradients_by_wrapping_model_with_layer.ipynb)
	- learn how to use `nif.layers.JacobianLayer`, `nif.layers.HessianLayer`

## How to cite

If you find NIF is helpful to you, you can cite our [paper](https://arxiv.org/abs/2204.03216) in the following bibtex format

   ```
@article{pan2022neural,
  title={Neural Implicit Flow: a mesh-agnostic dimensionality reduction paradigm of spatio-temporal data},
  author={Pan, Shaowu and Brunton, Steven L and Kutz, J Nathan},
  journal={arXiv preprint arXiv:2204.03216},
  year={2022}
}
   ```

## License

[LGPL-2.1 License](https://github.com/pswpswpsw/nif/blob/master/LICENSE)
