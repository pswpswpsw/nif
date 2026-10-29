Data and normalization
======================

Point-wise arrays preserve the column order
``[parameters, spatial coordinates, targets, optional sample weight]``.
Normalization returns the data and transformation statistics; use the output
column slices of those statistics to restore predictions to physical units.

.. code-block:: python

   from nif.data import PointWiseData
   from nif.demo import TravelingWave

   raw = TravelingWave().data_raw
   normalized, mean, scale = PointWiseData.minmax_normalize(
       raw, n_para=1, n_x=1, n_target=1
   )
   restored_targets = PointWiseData.inverse_normalize(
       normalized[:, 2:3], mean[2:3], scale[2:3]
   )

Point-wise data
---------------

.. autoclass:: nif.data.PointWiseData
   :members: parameter, x, u, standard_normalize, minmax_normalize, inverse_normalize
   :undoc-members:

Both normalization functions apply the column-wise transform
``normalized = (raw - offset) / scale``. The returned array named ``mean``
contains the offset and the returned array named ``std`` contains the scale.
Their definitions depend on the method and column group:

.. list-table:: Normalization offsets and scales
   :header-rows: 1
   :widths: 30 35 35

   * - Method and columns
     - Offset
     - Scale
   * - Standard: parameters, coordinates and targets
     - Raw column mean
     - Raw population standard deviation (``ddof=0``)
   * - Min-max: parameters and coordinates
     - ``(raw_min + raw_max) / 2``
     - ``(raw_max - raw_min) / 2``
   * - Min-max: targets
     - Raw column mean
     - ``max(abs(raw_column))``
   * - Either method: final weight column with ``area_weighted=True``
     - ``0``
     - Raw mean weight

In particular, min-max target scaling uses the maximum absolute *raw* target,
while its offset remains the raw target mean. It does not use the target
half-range or recompute a scale after centering.

Without area weights, the normalization functions return
``(normalized_data, mean, scale)``. With ``area_weighted=True``, they return
``(normalized_data_without_weights, mean, scale, normalized_weights)``.
Feature and target columns retain their order in both cases. The instance
properties ``parameter``, ``x`` and ``u`` read slices of ``self.data``; populate
that normalized array before accessing the properties.

The offset and scale vectors include the final weight column when weights are
enabled, even though the normalized weights are returned separately from the
data matrix. Zero or non-finite scales raise ``ValueError``. For predictions,
pass only the target-column slices to ``inverse_normalize``; the inverse is
``physical_prediction = normalized_prediction * scale + offset``.

Streamed record datasets
------------------------

``TFRDataset`` reads and writes column-list Example records with native PyTorch
iteration. A meta-dataset entry contains one tensor per stored column, shaped
``[1, rows]``. Convert that entry to a ``DataLoader`` with
``gen_dataset_from_batch_file`` to obtain ``(features, targets)`` batches, or
``(features, targets, weights)`` when area weighting is enabled.

.. autoclass:: nif.data.TFRDataset
   :members: create_from_npz, get_tfr_meta_dataset, gen_dataset_from_batch_file
   :undoc-members:

This small example writes shards in a temporary directory and reads a batch.

.. code-block:: python

   from pathlib import Path
   from tempfile import TemporaryDirectory
   import numpy as np
   from nif.data import TFRDataset
   from nif.demo import TravelingWave

   with TemporaryDirectory() as directory:
       root = Path(directory)
       np.savez(root / "wave.npz", data=TravelingWave().data)
       records = TFRDataset(n_feature=2, n_target=1)
       records.create_from_npz(1000, root / "wave.npz", "data", root / "shards", "wave")
       first_shard = next(iter(records.get_tfr_meta_dataset(root / "shards", epoch=1)))
       loader = records.gen_dataset_from_batch_file(first_shard, batch_size=128)
       features, targets = next(iter(loader))
