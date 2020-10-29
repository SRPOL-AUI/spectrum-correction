"""Models used in this experiment"""

from tensorflow.python import keras
from tensorflow.python.keras import layers


def basic(num_classes, input_shape, hparams):
    return keras.Sequential([
        layers.Conv2D(16, 3, input_shape=input_shape, activation='relu'),
        layers.BatchNormalization(center=False, scale=False),

        layers.Conv2D(32, 3, 2, activation='relu'),
        layers.BatchNormalization(center=False, scale=False),

        layers.Conv2D(32, 3, 1, activation='relu'),
        layers.BatchNormalization(center=False, scale=False),

        layers.Conv2D(64, 3, 2, activation='relu'),
        layers.BatchNormalization(center=False, scale=False),

        layers.Conv2D(64, 3, 1, activation='relu'),
        layers.BatchNormalization(center=False, scale=False),

        layers.GlobalAveragePooling2D(),

        layers.Dense(num_classes, activation='softmax', name='label'),
    ])


def build_model(name, num_classes, input_shape, hparams):
    return {
        'basic': basic,
    }[name](num_classes, input_shape, hparams)
