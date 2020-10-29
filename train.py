#!/usr/bin/env python

import re
import json
import warnings
import argparse

warnings.filterwarnings('ignore', category=FutureWarning, module='tensor.*dtypes')

import h5py
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.optimizer_v2.adadelta import Adadelta

from experiment import callbacks, utils
from experiment.models import build_model
from experiment.sequences import HDF5Batch, Sparse2OneHot, Mixup


def _train(hparams):
    log_dir, start_time = utils.setup_experiment(
        hparams['logs'],
        hparams['name'],
        hparams['seed'],
        save_diff=False,  # set to True to export uncommitted git changes
    )
    hparams['time'] = str(start_time)

    with h5py.File(hparams['data'], 'r') as h5:
        classes = json.loads(h5.attrs['classes'])
        devices = json.loads(h5.attrs['devices'])
        input_shape = json.loads(h5.attrs['input_shape'])
        has_eval = 'eval' in h5
        has_dev = 'dev' in h5

    train_sequence = HDF5Batch(hparams['batch_size'], hparams['data'], 'train/all/features', 'train/all/label')
    train_sequence = Sparse2OneHot(train_sequence, len(classes))
    if hparams['mixup']:
        train_sequence = Mixup(train_sequence, hparams['mixup'], hparams['mixup_exp'])

    hooks = []
    losses = ['categorical_crossentropy']
    metrics = ['categorical_accuracy']
    averages = {
        'abc': ('categorical_accuracy', 'a|b|c'),
        'bc': ('categorical_accuracy', 'b|c'),
    }
    target_metric = 'categorical_accuracy/bc/dev'

    if has_dev:
        hooks += _validation_hooks('dev', devices, len(classes), hparams, averages)
    if has_eval:
        hooks += _validation_hooks('eval', devices, len(classes), hparams, averages)

    hooks += [
        callbacks.Timer(),
        keras.callbacks.ModelCheckpoint(
            str(log_dir / 'weights.hdf5'),
            monitor=target_metric,
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=target_metric,
            factor=hparams['reduction_factor'],
            patience=hparams['reduction_patience'],
            min_delta=hparams['reduction_delta'],
            verbose=1,
        ),
        callbacks.TensorBoard(str(log_dir), update_freq='batch', hparams=hparams),
        keras.callbacks.TerminateOnNaN(),
    ]

    model = build_model(hparams['model'], len(classes), input_shape, hparams)
    model.compile(
        Adadelta(hparams['lr'], rho=hparams['rho']),
        loss=losses,
        loss_weights=None,
        metrics=metrics,
        run_eagerly=hparams['eager'],
    )

    if hparams['load']:
        model.load_weights(hparams['load'])

    model.fit_generator(
        train_sequence,
        shuffle=True,
        epochs=hparams['epochs'],
        max_queue_size=hparams['queue'],
        callbacks=hooks,
        verbose=1,
    )

    print('DONE')


def _validation_hooks(subset, devices, depth, hparams, averages, every_epoch=True):
    h5 = h5py.File(hparams['data'], mode='r')
    hooks = []

    for device in devices:
        if f'{subset}/{device}' not in h5:
            continue

        sequence = Sparse2OneHot(
            HDF5Batch(
                hparams['batch_size'],
                h5,
                f'{subset}/{device}/features',
                f'{subset}/{device}/label',
            ),
            depth,
        )

        hooks.append(callbacks.Validate(
            sequence,
            suffix=f'{device}/{subset}',
            batch_size=hparams['batch_size'],
            every_epoch=every_epoch),
        )

    for name, (metric, pattern) in averages.items():
        hooks.append(callbacks.MacroAverage(
            f'{metric}/{name}/{subset}',
            [
                f'{metric}/{device}/{subset}'
                for device in devices
                if re.match(pattern, device)
            ],
        ))

    return hooks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64,
                        help='size of the batch for training and validation')
    parser.add_argument('--seed', type=int, default=42,
                        help='seed for the random number generators')
    parser.add_argument('--epochs', type=int, default=150,
                        help='number of training epochs')
    parser.add_argument('--queue', type=int, default=16,
                        help='queue size for Keras')
    parser.add_argument('--logs', type=str, default='./logs',
                        help='directory to store outputs')
    parser.add_argument('--reduction', action='store_true',
                        help='reduce learning rate on plateau')
    parser.add_argument('--reduction-factor', type=float, default=0.5,
                        help='learning rate reduction factor')
    parser.add_argument('--reduction-patience', type=float, default=16,
                        help='reduction patience in epochs')
    parser.add_argument('--reduction-delta', type=float, default=1e-3,
                        help='minimal considered improvement')
    parser.add_argument('--lr', type=float, default=0.5,
                        help='learning rate')
    parser.add_argument('--rho', type=float, default=0.95,
                        help='rho parameter for Adadelta')
    parser.add_argument('--model', type=str, default='basic',
                        help='name of the selected model')
    parser.add_argument('--eager', action='store_true',
                        help='run in eager mode')
    parser.add_argument('--mixup', type=float, default=0,
                        help='alpha parameter for mixup. Misup is disabled if set to zero')
    parser.add_argument('--mixup-exp', action='store_true',
                        help='perform mixup without log scaling')
    parser.add_argument('--reproducible', action='store_true',
                        help='switch to reproducible mode')
    parser.add_argument('--load', type=str, default='',
                        help='model weights to load')
    parser.add_argument('data', type=str,
                        help='path to the preprocessed dataset')
    parser.add_argument('name', type=str,
                        help='name for the experiment')

    hparams = parser.parse_args().__dict__

    if hparams['reproducible']:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

    _train(hparams)
