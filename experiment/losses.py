"""Losses."""

import tensorflow as tf

from tensorflow.python.keras import backend as K


def categorical_focal_loss(target, predictions, power=1.0, weights=None, axis=-1):
    """Categorical focal loss for Keras.

    Args:
        target: ground truth categories with at least one dimension
        predictions: batched outputs of the network, same shape as `target`
        power (float): power parameter
        weights (floats): weights
        axis (int): axis containing categories

    Returns:
        score for givent pairs of predictions and targets
    """
    power = tf.convert_to_tensor(power, tf.float32)
    eps = tf.convert_to_tensor(K.epsilon(), predictions.dtype.base_dtype)

    target.shape.assert_same_rank(predictions.shape)
    power.shape.assert_has_rank(0)

    predictions = predictions / tf.reduce_sum(predictions, axis, True)
    predictions = tf.clip_by_value(predictions, eps, 1.0 - eps)

    p = tf.reduce_sum(target * predictions, axis)

    if weights is not None:
        weights = tf.convert_to_tensor(weights)
        weights.shape.assert_has_rank(1)
        a = tf.tensordot(target, weights, [[axis], [0]])
    else:
        a = tf.reduce_sum(target, axis)

    return -a * tf.pow(1 - p, power) * tf.math.log(p)


def sparse_categorical_focal_loss(target, predictions, power=1.0, weights=None, axis=-1):
    """Sparse categorical focal loss for Keras.

    Args:
        target: ground truth categories with at least one dimension
        predictions: batched outputs of the network, same shape as `target`
        power (float): power parameter
        weights (floats): weights
        axis (int): axis containing categories

    Returns:
        score for givent pairs of predictions and targets
    """
    target = tf.one_hot(tf.squeeze(tf.cast(target, tf.int32)), predictions.shape[axis])
    return categorical_focal_loss(target, predictions, power, weights, axis)
