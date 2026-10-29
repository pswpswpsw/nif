Packaged datasets
=================

The ``nif.demo`` classes load datasets bundled with NIF and inherit the
point-wise data interface. ``data_raw`` contains the source values; ``data``
contains the normalized feature and target columns. ``mean`` and ``std`` store
the normalization offset and scale used by each loader.

- ``data_raw``: the source parameter, coordinate, target and optional weight
  columns in physical units.
- ``data``: the normalized parameter, coordinate and target columns.
- ``mean`` and ``std``: per-column offset and scale for the transform
  ``normalized = (raw - mean) / std``. They are derived from the raw data, not
  statistics recomputed from the normalized array.
- ``parameter``, ``x`` and ``u``: views into the corresponding normalized column
  groups. ``n_p``, ``n_x`` and ``n_o`` give their feature counts.
- ``sample_weight``: the separated normalized weights for ``CylinderFlow``;
  ``None`` for the two traveling-wave loaders.

.. code-block:: python

   from nif.demo import TravelingWave

   wave = TravelingWave()
   features = wave.data[:, :2]  # [time, spatial coordinate]
   targets = wave.data[:, 2:3]
   physical_targets = targets * wave.std[2:3] + wave.mean[2:3]

Traveling wave
--------------

.. autoclass:: nif.demo.TravelingWave
   :class-doc-from: init

This loader uses standard normalization and has one parameter, one coordinate
and one field component.

High-frequency traveling wave
-----------------------------

.. autoclass:: nif.demo.TravelingWaveHighFreq

This loader uses ``PointWiseData.minmax_normalize`` and has one parameter,
one coordinate and one field component.

Cylinder flow
-------------

.. autoclass:: nif.demo.CylinderFlow
   :class-doc-from: init

The column groups are time, two spatial coordinates, two velocity components
and sample weight. The loader separates normalized weights into
``sample_weight`` and uses min-max normalization for the field data.
