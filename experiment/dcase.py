"""Utils for DCASE 2019 Mobile dataset"""

import pathlib
import numpy as np
import pandas as pd

from tensorflow.python import keras


CLASSES = [
    'airport',
    'shopping_mall',
    'metro_station',
    'street_pedestrian',
    'public_square',
    'street_traffic',
    'tram',
    'bus',
    'metro',
    'park',
]
DEVICES = ['a', 'b', 'c']
CITIES = [
    'barcelona',
    'helsinki',
    'lisbon',
    'london',
    'lyon',
    'milan',
    'paris',
    'prague',
    'stockholm',
    'vienna',
]
TRAINING_CITIES = [city for city in CITIES if city != 'milan']

_URL = 'https://zenodo.org/record/2589332/files/{file}?download=1'
_FILES = [
    'TAU-urban-acoustic-scenes-2019-mobile-development.audio.1.zip',
    'TAU-urban-acoustic-scenes-2019-mobile-development.audio.2.zip',
    'TAU-urban-acoustic-scenes-2019-mobile-development.audio.3.zip',
    'TAU-urban-acoustic-scenes-2019-mobile-development.audio.4.zip',
    'TAU-urban-acoustic-scenes-2019-mobile-development.audio.5.zip',
    'TAU-urban-acoustic-scenes-2019-mobile-development.audio.6.zip',
    'TAU-urban-acoustic-scenes-2019-mobile-development.audio.7.zip',
    'TAU-urban-acoustic-scenes-2019-mobile-development.audio.8.zip',
    'TAU-urban-acoustic-scenes-2019-mobile-development.audio.9.zip',
    'TAU-urban-acoustic-scenes-2019-mobile-development.audio.10.zip',
    'TAU-urban-acoustic-scenes-2019-mobile-development.audio.11.zip',
    'TAU-urban-acoustic-scenes-2019-mobile-development.meta.zip',
]

_DEV_PATH = 'evaluation_setup/fold1_evaluate.csv'
_TRAIN_PATH = 'evaluation_setup/fold1_train.csv'

_SEP = '\t'


def _parse_path(path):
    splits = str(pathlib.Path(path).stem).split('-')
    assert len(splits) == 5
    return splits


def _expand_frame(files):
    labels, cities, locations, segments, ds = zip(*[
        _parse_path(path) for path in files
    ])
    return pd.DataFrame({
        'file': files,
        'label': pd.Categorical(labels, categories=CLASSES),
        'device': pd.Categorical(ds, categories=DEVICES),
        'city': pd.Categorical(cities),
        'location': locations,
        'segment': segments,
    })


def download(data_dir):
    """Download the dataset to the specified directory.

    Args:
        data_dir: output directory
    """
    for file in _FILES:
        keras.utils.get_file(
            file,
            _URL.format(file=file),
            extract=True,
            cache_subdir='',
            cache_dir=str(data_dir),
        )


def get_correction_recordings(dcase_frame, num_samples, random_state=None):
    """Aggregates a list of recordings.

    Args:
        dcase_frame (pd.DataFrame): dataframe with metadata for the dataset
        num_samples (int): number of examples to select, use all if evaluates to False
        random_state: random state for the sampling

    Returns:
        a list of dictionaries where each device is represented by a different key
        and value is the path to the recording
    """
    recordings = dcase_frame.set_index(['label', 'city', 'location', 'segment', 'device'])\
        .unstack('device')\
        .dropna()
    recordings.columns = recordings.columns.droplevel()

    if num_samples:
        recordings = recordings.sample(
            n=num_samples,
            replace=False,
            random_state=random_state
        )
    return recordings


def get_aligned_recordings(dcase_frame, num_samples, random_state=None):
    """Aggregates a list of aligned recordings.

    Args:
        dcase_frame (pd.DataFrame): dataframe with metadata for the dataset
        num_samples (int): number of examples to select, use all if evaluates to False
        random_state: random state for the sampling

    Returns:
        a list of dictionaries where each device is represented by a different key
        and value is the path to the recording
    """
    recordings = get_correction_recordings(dcase_frame, num_samples, random_state)
    return recordings.to_dict('records')


def get_unaligned_recordings(dcase_frame, num_samples, random_state=None):
    """Aggregates a list of unaligned recordings.

    Args:
        dcase_frame (pd.DataFrame): dataframe with metadata for the dataset
        num_samples (int): number of examples to select, use all if evaluates to False
        random_state: random state for the sampling

    Returns:
        a list of dictionaries where each device is represented by a different key
        and value is the path to the recording
    """
    recordings = get_correction_recordings(dcase_frame, num_samples, random_state)
    return {
        device: column.sample(len(column), random_state=random_state).values
        for device, column in recordings.iteritems()
    }


def _read_protocol(dataset_path, protocol):
    return pd.read_csv(pathlib.Path(dataset_path) / protocol, sep=_SEP)


def _get_dev_frame(dataset_path):
    protocol = _read_protocol(dataset_path, _DEV_PATH)
    return _expand_frame(protocol.filename)


def _get_training_frame(dataset_path):
    protocol = _read_protocol(dataset_path, _TRAIN_PATH)
    return _expand_frame(protocol.filename)


def get_split(
    data_dir,
    split='official',
    validation_fraction=0.3,
    reuse=False,
    holdout_cities=None,
    retries=32,
    full_paths=True,
):
    """Loads the official dataset split or creates a randomly selected train/validation split.

    Args:
        data_dir: where the dataset is stored
        split ('official'/int): use 'official' to select the official split
            or int for a randomly generated one
        validation_fraction: fraction of examples used for the validation subset
        reuse: use examples from the developement set
        holdout_cities: which cities to used only in the validation subset
        retries: number of retries
        full_paths: use full paths to the audio files

    Returns:
        pd.DataFrame: training examples
        pd.DataFrame: validation/testing examples
    """

    training = _get_training_frame(data_dir)
    validation = _get_dev_frame(data_dir)

    if split != 'official':
        if reuse:
            training = pd.concat([training, validation])
        training, validation = _split(
            training,
            validation_fraction,
            holdout_cities or [],
            int(split),
            retries
        )

    if full_paths:
        training.file = training.file.map(lambda f: str(data_dir / f))
        validation.file = validation.file.map(lambda f: str(data_dir / f))

    training = training.sample(n=len(training), replace=False)
    validation = validation.sample(n=len(validation), replace=False)

    return training, validation


def __split(dataset, fraction, maintain, separate, random_state):
    training = []
    validation = []

    for _, group in dataset.groupby(maintain):
        per_chunk = group.groupby(separate, as_index=False).size()
        per_chunk = per_chunk.sample(per_chunk.shape[0], random_state=random_state)

        target = fraction * len(group)
        num_selected = per_chunk.cumsum()
        last_smaller = num_selected < target
        first_larger = np.roll(last_smaller, 1)
        first_larger[0] = True
        if np.abs(per_chunk[last_smaller].sum() - target) \
                < np.abs(per_chunk[first_larger].sum() - target):
            selection = per_chunk.index[last_smaller]
        else:
            selection = per_chunk.index[first_larger]

        split = pd.merge(
            group,
            selection.to_frame().reset_index(drop=True),
            on=separate,
            how='outer',
            indicator=True,
        )

        validation.append(split[split['_merge'] == 'both'].drop(columns='_merge'))
        training.append(split[split['_merge'] == 'left_only'].drop(columns='_merge'))

    return pd.concat(training, axis=0), pd.concat(validation, axis=0)


def _split(examples, fraction, holdout_cities, seed, retries):
    assert len(set(examples['file'])) == len(examples)
    retries = max(1, retries)

    if isinstance(seed, np.random.RandomState):
        random_state = seed
    else:
        random_state = np.random.RandomState(seed)

    best_split = None
    best_error = None
    for _ in range(retries):
        holdout = examples[np.isin(examples['city'], holdout_cities)]
        splitable = examples[~np.isin(examples['city'], holdout_cities)]

        # Official split would discard examples from holdout_cities
        validation_size = max(fraction * len(examples) - len(holdout), 0)
        training, validation = __split(
            splitable,
            validation_size / len(splitable),
            maintain=['label', 'city'],
            separate=['city', 'location'],
            random_state=random_state,
        )

        validation = pd.concat([validation, holdout], axis=0)

        error = _split_error(training, validation, ['label', 'city', 'device'])
        if best_error is None or best_error > error:
            best_split = training, validation
            best_error = error

    return best_split


def _split_error(training, validation, columns):
    error = 0.0
    for column in columns:
        training_ = training[column].value_counts() / len(training)
        validation_ = validation[column].value_counts() / len(validation)
        error = max(error, np.max(np.abs(training_ - validation_)))
    return error
