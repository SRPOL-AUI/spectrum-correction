#!/usr/bin/env python

"""DCASE 2019 Mobile preprocessing script."""

import os
import sys
import json
import random
import logging
import argparse
import warnings
import pathlib

warnings.filterwarnings('ignore', category=FutureWarning, module='tensor.*dtypes')

import tqdm
import h5py
import numpy as np
import pandas as pd

from experiment import dcase
from experiment import preprocessing
from experiment import correction as sc

# Tensorflow 2.0.0b1 invades the root logger
# This is a workaround.
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
logging.root.addHandler(logging.StreamHandler(sys.stdout))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


_DCASE_FOLDER = 'TAU-urban-acoustic-scenes-2019-mobile-development'


def _main(args):
    random.seed(args.seed)
    np.random.seed(args.seed + 1)

    assert args.no_correction or not args.grouped

    if not (args.data_dir / _DCASE_FOLDER).exists():
        logger.info('Downloading the DCASE 2019 dataset.')
        args.data_dir.mkdir(parents=True, exist_ok=True)
        dcase.download(args.data_dir)

    logger.info('Loading metadata and preparing the split.')
    train_frame, dev_frame = dcase.get_split(
        args.data_dir / _DCASE_FOLDER,
        args.split,
        args.validation_fraction,
        args.reuse,
        args.holdout_cities,
        retries=16,
        full_paths=True,
    )

    _print_stats(train_frame, dev_frame)
    if args.no_correction:
        correction = None
    elif args.aligned:
        correction = get_aligned_correction(args, train_frame)
    else:
        correction = get_unaligned_correction(args, train_frame)

    input_shape = _preprocess_features(train_frame.iloc[0], correction, args).shape
    classes = train_frame.label.cat.categories.tolist()
    devices = train_frame.device.cat.categories.tolist()
    class_weights = [1.0] * len(train_frame.label.cat.categories)

    try:
        with h5py.File(args.output, 'w') as h5:
            h5.attrs['name'] = 'DCASE 2019 Mobile'

            h5.attrs['input_shape'] = json.dumps(input_shape)
            h5.attrs['classes'] = json.dumps(classes)
            h5.attrs['devices'] = json.dumps(devices)
            h5.attrs['class_weights'] = json.dumps(class_weights)

            logger.info('Processing training files.')
            _load_subset(h5, 'train/all', train_frame, input_shape, correction, args)
            for device in devices:
                current_frame = train_frame[train_frame.device == device]
                _load_subset(h5, f'train/{device}', current_frame, input_shape, correction, args)

            logger.info('Processing development files.')
            for device in devices:
                current_frame = dev_frame[dev_frame.device == device]
                _load_subset(h5, f'dev/{device}', current_frame, input_shape, correction, args)

            if not args.no_standardization:
                if args.grouped:
                    standardize_grouped(h5, devices)
                else:
                    standardize(h5, devices)

    except:
        logger.info('Deleting the dataset.')
        os.remove(args.output)
        raise

    logger.info('Done.')


def _load_subset(h5, name, frame, input_shape, correction, args):
    h5[f'{name}/label'] = frame.label.cat.codes.tolist()
    h5[f'{name}/device'] = frame.device.cat.codes.tolist()
    dataset = h5.create_dataset(
        f'{name}/features',
        shape=[len(frame)] + list(input_shape),
        dtype=np.float32,
    )

    for i, (_, row) in enumerate(tqdm.tqdm(frame.iterrows(), total=len(frame), unit='file')):
        dataset[i] = _preprocess_features(row, correction, args)


def _preprocess_features(row, correction, args):
    features = preprocessing.spectrogram(
        row,
        correction,
        args.num_fft,
        args.hop_length,
        args.power,
        args.num_mels,
        args.htk,
        not args.no_norm,
    )
    return np.expand_dims(features, -1)


def _proportions(training, validation, column):
    training_counts = training[column].value_counts()
    validation_counts = validation[column].value_counts()

    props = pd.concat([
        training_counts,
        validation_counts,
        training_counts / len(training),
        validation_counts / len(validation),
    ], axis=1)

    props = props.fillna(0)
    props.columns = pd.MultiIndex.from_product([
        ['count', 'fraction'],
        ['training', 'validation']
    ])
    return props


def _print_stats(training, validation):
    training_size = len(training)
    validation_size = len(validation)
    total_size = len(validation) + len(training)
    labels = _proportions(training, validation, 'label')
    devices = _proportions(training, validation, 'device')
    cities = _proportions(training, validation, 'city')

    print(
        f'Training size = {training_size}, validation size = {validation_size}, total = {total_size}',
        'labels stats:',
        labels,
        '',
        'devices stats:',
        devices,
        '',
        'cities stats:',
        cities,
        sep='\n',
    )


def get_aligned_correction(args, train_frame):
    if args.fir:
        correction = sc.AlignedSpectrumCorrectionFIR(args.num_taps, args.num_fft, args.hop_length)
        logger.info('Computing correction coefficients for FIR filters (aligned).')
    else:
        correction = sc.AlignedSpectrumCorrection(args.num_fft, args.hop_length)
        logger.info('Computing correction coefficients. (aligned)')

    aligned_recordings = dcase.get_aligned_recordings(train_frame, args.num_samples)
    # generator of dicts (load on the fly)
    aligned_recordings = (
        {
            device: preprocessing.read_wave(path)[0]
            for device, path in row.items()
        } for row in aligned_recordings
    )
    correction.fit(aligned_recordings, args.reference)

    return correction


def get_unaligned_correction(args, train_frame):
    if args.fir:
        correction = sc.UnalignedSpectrumCorrectionFIR(args.num_taps, args.num_fft, args.hop_length)
        logger.info('Computing correction coefficients for FIR filters (unaligned).')
    else:
        correction = sc.UnalignedSpectrumCorrection(args.num_fft, args.hop_length)
        logger.info('Computing correction coefficients. (unaligned)')

    shuffled_recordings = dcase.get_unaligned_recordings(train_frame, args.num_samples)
    # dict of generators/iterables (load on the fly)
    shuffled_recordings = {
        device: (preprocessing.read_wave(path)[0] for path in paths)
        for device, paths in shuffled_recordings.items()
    }
    correction.fit(shuffled_recordings, args.reference)

    return correction


def standardize(h5, devices):
    logger.info('Computing standardization.')
    standardization = preprocessing.compute_standardization(
        h5['train/all/features'],
        axes=[0, 2],
    )

    logger.info('Applying standardization to training features.')
    preprocessing.apply_standardization(h5['train/all/features'], standardization)
    for device in devices:
        logger.info('Applying standardization to training features (device %s).', device)
        preprocessing.apply_standardization(h5['train'][device]['features'], standardization)
        logger.info('Applying standardization to developement features (device %s).', device)
        preprocessing.apply_standardization(h5['dev'][device]['features'], standardization)


def standardize_grouped(h5, devices):
    standardization = [
        preprocessing.compute_standardization(
            h5['train'][device]['features'],
            axes=[0, 2],
        ) for device in devices
    ]
    logger.info('Applying standardization to training features.')
    preprocessing.apply_grouped_standardization(h5['train/all/features'], h5['train/all/device'],
                                                standardization)
    for device in devices:
        logger.info('Applying standardization to training features (device %s).', device)
        preprocessing.apply_grouped_standardization(h5['train'][device]['features'],
                                                    h5['train'][device]['device'],
                                                    standardization)
        logger.info('Applying standardization to developement features (device %s).', device)
        preprocessing.apply_grouped_standardization(h5['dev'][device]['features'],
                                                    h5['dev'][device]['device'], standardization)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=pathlib.Path, default='./data/dcase',
                        help='directory where datasets is stored')
    parser.add_argument('--seed', type=int, default=42,
                        help='seed for the random number generators')
    parser.add_argument('--num-fft', type=int, default=2048,
                        help='number of frequency bins in STFT')
    parser.add_argument('--hop-length', type=int, default=512,
                        help='hop length for the STFT')
    parser.add_argument('--num-mels', type=int, default=256,
                        help='number of bins after conversion to mel-scale')
    parser.add_argument('--power', type=int, default=1,
                        help='power for the spectrogram')
    parser.add_argument('--htk', action='store_true',
                        help='use HTK formula instead of Slaney')
    parser.add_argument('--no-norm', action='store_true',
                        help='disable normalization of the mel bands')
    parser.add_argument('--no-correction', action='store_true',
                        help='disable spectrum correction')
    parser.add_argument('--no-standardization', action='store_true',
                        help='disable standardization of the features')
    parser.add_argument('--reference', default=None,
                        help='reference devices for spectrum correction')
    parser.add_argument('--aligned', action='store_true',
                        help='use aligned variant of the spectrum correction')
    parser.add_argument('--fir', action='store_true',
                        help='use FIR implementation of the spectrum correction')
    parser.add_argument('--grouped', action='store_true',
                        help='use STD implementation of the spectrum correction')
    parser.add_argument('--num-taps', type=int, default=1025,
                        help='number of taps for the FIR')
    parser.add_argument('--split', default='official',
                        help='train-test split, use "official" for the official split or any intiger for randomized validation')
    parser.add_argument('--reuse', action='store_true',
                        help='use examples from the developement set in the new cross-validation split')
    parser.add_argument('--validation-fraction', type=float, default=0.4,
                        help='percentage of the examples to use for validation in case of randomized validation')
    parser.add_argument('--holdout-cities', nargs='+', choices=dcase.TRAINING_CITIES,
                        help='which cities to include only in the validation set')
    parser.add_argument('num_samples', type=int, help='number of samples used for computing correction coefficients')
    parser.add_argument('output', type=str, help='where to save the output file')
    _main(parser.parse_args())
