Eight scientific tutorials
==========================

Each native notebook contains the scientific case, training loop, evaluation,
and figures. Open the version 2.0.0 notebook in GitHub or run it in Google Colab.
The full scientific workloads are the default. ``NIF_SMOKE=1`` selects a short
development check and is separate from full training.

.. list-table:: Native PyTorch tutorials
   :header-rows: 1
   :widths: 12 29 39 20
   :class: nif-tutorial-table

   * - Tutorial
     - Scientific case
     - Main implementation concepts
     - Open
   * - 1
     - One-dimensional traveling wave
     - Configure NIF, train, save and restore weights, refine with L-BFGS.
     - `Notebook <https://github.com/pswpswpsw/nif/blob/v2.0.0/tutorial/1_simple_1d_wave_pytorch.ipynb>`__ / `Colab <https://colab.research.google.com/github/pswpswpsw/nif/blob/v2.0.0/tutorial/1_simple_1d_wave_pytorch.ipynb>`__
   * - 2
     - Multiscale wave
     - Use NIFMultiScale to represent high-frequency spatial structure.
     - `Notebook <https://github.com/pswpswpsw/nif/blob/v2.0.0/tutorial/2_multi_scale_NIF_pytorch.ipynb>`__ / `Colab <https://colab.research.google.com/github/pswpswpsw/nif/blob/v2.0.0/tutorial/2_multi_scale_NIF_pytorch.ipynb>`__
   * - 3
     - Cylinder-flow data from an adaptive mesh
     - Use NIFMultiScaleLastLayerParameterized to learn linear spatial bases.
     - `Notebook <https://github.com/pswpswpsw/nif/blob/v2.0.0/tutorial/3_multi_scale_linear_NIF_pytorch.ipynb>`__ / `Colab <https://colab.research.google.com/github/pswpswpsw/nif/blob/v2.0.0/tutorial/3_multi_scale_linear_NIF_pytorch.ipynb>`__
   * - 4
     - Input-output derivatives
     - Wrap a field model with JacobianLayer and HessianLayer.
     - `Notebook <https://github.com/pswpswpsw/nif/blob/v2.0.0/tutorial/4_get_gradients_by_wrapping_model_with_layer_pytorch.ipynb>`__ / `Colab <https://colab.research.google.com/github/pswpswpsw/nif/blob/v2.0.0/tutorial/4_get_gradients_by_wrapping_model_with_layer_pytorch.ipynb>`__
   * - 5
     - Large-scale record data
     - Create shards and stream batches from a one-million-row dataset.
     - `Notebook <https://github.com/pswpswpsw/nif/blob/v2.0.0/tutorial/5_large_scale_training_on_tensorflow_record_data_pytorch.ipynb>`__ / `Colab <https://colab.research.google.com/github/pswpswpsw/nif/blob/v2.0.0/tutorial/5_large_scale_training_on_tensorflow_record_data_pytorch.ipynb>`__
   * - 6
     - Multiscale fields with L1/L2 regularization
     - Penalize ParameterNet parameters and inspect a refined evaluation grid.
     - `Notebook <https://github.com/pswpswpsw/nif/blob/v2.0.0/tutorial/6_revisit_multi_scale_NIF_with_L1_L2_regularization_pytorch.ipynb>`__ / `Colab <https://colab.research.google.com/github/pswpswpsw/nif/blob/v2.0.0/tutorial/6_revisit_multi_scale_NIF_with_L1_L2_regularization_pytorch.ipynb>`__
   * - 7
     - Model compression
     - Magnitude pruning, dynamic int8 CPU inference, and inference export.
     - `Notebook <https://github.com/pswpswpsw/nif/blob/v2.0.0/tutorial/7_model_pruning_and_quantization_pytorch.ipynb>`__ / `Colab <https://colab.research.google.com/github/pswpswpsw/nif/blob/v2.0.0/tutorial/7_model_pruning_and_quantization_pytorch.ipynb>`__
   * - 8
     - Sobolev training of a multiscale field
     - Supervise spatial derivatives and evaluate field and derivative errors.
     - `Notebook <https://github.com/pswpswpsw/nif/blob/v2.0.0/tutorial/8_revisit_multi_scale_NIF_with_sobolov_training_pytorch.ipynb>`__ / `Colab <https://colab.research.google.com/github/pswpswpsw/nif/blob/v2.0.0/tutorial/8_revisit_multi_scale_NIF_with_sobolov_training_pytorch.ipynb>`__

The :doc:`validation` and :doc:`comparison_records` pages retain the recorded
error measurements and acceptance scope. Training points and additional spatial
or temporal evaluation points are distinguished in those comparisons.

Original cases and compatibility notebooks
------------------------------------------

The original notebooks preserve the scientific definitions and historical
outputs. The compatibility copies contain the pinned setup used for the
historical backend. The following links preserve access to both sets.

.. list-table:: Historical references
   :header-rows: 1
   :class: nif-tutorial-table

   * - Tutorial
     - Original notebook
     - Compatibility notebook
   * - 1
     - `Original <https://github.com/pswpswpsw/nif/blob/v2.0.0/tutorial/1_simple_1d_wave.ipynb>`__
     - `Compatibility <https://github.com/pswpswpsw/nif/blob/v2.0.0/tutorial/1_simple_1d_wave_colab.ipynb>`__
   * - 2
     - `Original <https://github.com/pswpswpsw/nif/blob/v2.0.0/tutorial/2_multi_scale_NIF.ipynb>`__
     - `Compatibility <https://github.com/pswpswpsw/nif/blob/v2.0.0/tutorial/2_multi_scale_NIF_colab.ipynb>`__
   * - 3
     - `Original <https://github.com/pswpswpsw/nif/blob/v2.0.0/tutorial/3_multi_scale_linear_NIF.ipynb>`__
     - `Compatibility <https://github.com/pswpswpsw/nif/blob/v2.0.0/tutorial/3_multi_scale_linear_NIF_colab.ipynb>`__
   * - 4
     - `Original <https://github.com/pswpswpsw/nif/blob/v2.0.0/tutorial/4_get_gradients_by_wrapping_model_with_layer.ipynb>`__
     - `Compatibility <https://github.com/pswpswpsw/nif/blob/v2.0.0/tutorial/4_get_gradients_by_wrapping_model_with_layer_colab.ipynb>`__
   * - 5
     - `Original <https://github.com/pswpswpsw/nif/blob/v2.0.0/tutorial/5_large_scale_training_on_tensorflow_record_data.ipynb>`__
     - `Compatibility <https://github.com/pswpswpsw/nif/blob/v2.0.0/tutorial/5_large_scale_training_on_tensorflow_record_data_colab.ipynb>`__
   * - 6
     - `Original <https://github.com/pswpswpsw/nif/blob/v2.0.0/tutorial/6_revisit_multi_scale_NIF_with_L1_L2_regularization.ipynb>`__
     - `Compatibility <https://github.com/pswpswpsw/nif/blob/v2.0.0/tutorial/6_revisit_multi_scale_NIF_with_L1_L2_regularization_colab.ipynb>`__
   * - 7
     - `Original <https://github.com/pswpswpsw/nif/blob/v2.0.0/tutorial/7_model_pruning_and_quantization.ipynb>`__
     - `Compatibility <https://github.com/pswpswpsw/nif/blob/v2.0.0/tutorial/7_model_pruning_and_quantization_colab.ipynb>`__
   * - 8
     - `Original <https://github.com/pswpswpsw/nif/blob/v2.0.0/tutorial/8_revisit_multi_scale_NIF_with_sobolov_training.ipynb>`__
     - `Compatibility <https://github.com/pswpswpsw/nif/blob/v2.0.0/tutorial/8_revisit_multi_scale_NIF_with_sobolov_training_colab.ipynb>`__

See the `original API documentation <legacy/tensorflow/index.html>`_ for that
historical workflow. Use the native PyTorch links above for version 2.0.0.
