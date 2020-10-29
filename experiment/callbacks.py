"""Better callbacks for Keras"""

import time

import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorboard.plugins.hparams import api as hp
from tensorboard.plugins.hparams import summary_v2 as hp


class Validate(keras.callbacks.Callback):
    """Validation callback

    Args:
        x: features used for model evaluation
        y: labels used for model evaluation
        steps: number of steps to perform
        suffix: suffix for the metrics
        batch_size: size of the mini-batch
        verbose: print progress
        every_epoch (True/int): every which epoch to validate
    """

    def __init__(
        self,
        x,
        y=None,
        steps=None,
        suffix=None,
        batch_size=None,
        verbose=1,
        every_epoch=True
    ):
        super().__init__()
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.verbose = verbose
        self.suffix = suffix
        self.steps = steps
        self.every_epoch = every_epoch

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

        if self.every_epoch is True or epoch % self.every_epoch == 0:
            results = self.model.evaluate(
                self.x,
                self.y,
                batch_size=self.batch_size,
                verbose=self.verbose,
                steps=self.steps,
            )

            for label, result in zip(self.model.metrics_names, results):
                logs[self._format_label(label)] = result

    def _format_label(self, label):
        if self.suffix:
            return label + '/' + self.suffix
        else:
            return label


class Timer(keras.callbacks.Callback):
    """Adds number of epochs and time per epoch to the logs."""

    def __init__(self):
        super().__init__()
        self._last_batch = None

    def on_train_begin(self, logs=None):
        self._last_batch = time.time()

    def on_epoch_end(self, epoch, logs=None):
        now = time.time()
        logs['time/per_epoch'] = now - self._last_batch
        logs['time/num_epochs'] = epoch + 1
        self._last_batch = now


class MacroAverage(keras.callbacks.Callback):
    """Computes macro average from the specified metrics.

    Args:
        name: name for the macro average
        metrics: which metric to average
    """

    def __init__(self, name, metrics):
        super().__init__()
        self._metrics = metrics
        self._name = name

    def _push_average(self, logs):
        logs[self._name] = np.nanmean([logs.get(key, np.nan) for key in self._metrics])

    def on_epoch_end(self, epoch, logs=None):
        self._push_average(logs)


class TensorBoard(keras.callbacks.Callback):
    """An alternative TensorBoard callback that utilizes TB's groups.

    Args:
        log_dir: where to save the logs
        update_freq: how often to save the logs
        hparams: parameters of the experiment
    """

    def __init__(self, log_dir='logs', update_freq='batch', hparams=None):
        super().__init__()

        self.log_dir = log_dir
        self._hparams = hparams

        if update_freq == 'batch':
            self.update_freq = 1
        else:
            self.update_freq = update_freq

        self._samples_seen = 0
        self._samples_seen_at_last_write = 0
        self._total_batches_seen = 0

        self._writer = tf.summary.create_file_writer(self.log_dir)

    def on_train_begin(self, logs=None):
        if self._hparams:
            with self._writer.as_default(), tf.summary.record_if(True):
                hp.hparams(self._hparams)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        self._samples_seen += logs.get('size', 1)
        samples_seen_since = self._samples_seen - self._samples_seen_at_last_write
        if self.update_freq != 'epoch' and samples_seen_since >= self.update_freq:
            self._log_metrics(logs, suffix='batch')
            self._samples_seen_at_last_write = self._samples_seen
        self._total_batches_seen += 1

    def on_epoch_end(self, epoch, logs=None):
        self._log_metrics(logs, suffix='epoch')

    def on_train_end(self, logs=None):
        self._writer.close()

    def _format_tag(self, name, time_scale):
        return f'{name}/{time_scale}'

    def _log_metrics(self, logs, suffix):
        logs = logs or {}

        with self._writer.as_default(), tf.summary.record_if(True):
            for (name, value) in logs.items():
                if name in ('batch', 'size'):
                    continue

                tf.summary.scalar(
                    self._format_tag(name, suffix),
                    value,
                    step=self._samples_seen,
                )

            self._writer.flush()
