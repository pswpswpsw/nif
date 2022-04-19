from .__about__ import __version__

import tensorflow as tf
from tensorflow.keras import mixed_precision

from .model import NIFMultiScale, NIF

# tf.keras.backend.set_floatx("float64")

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_physical_devices('GPU')
    # logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

__all__ = [
    "tf",
    "NIFMultiScale",
    "NIF",
    "mixed_precision",
    "optimizers"
]