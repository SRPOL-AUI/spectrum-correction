"""Dataset independent data preprocessing."""

from math import floor, ceil

import librosa
import numpy as np
import soundfile as sf
import dask.array as da
from scipy import signal


def read_wave(path, start=0, stop=None):
    """Reads wave file form `start` to `stop`.

    Args:
        path (str): path to the wave file
        start: start in seconds
        stop: stop in seconds

    Returns:
        np.ndarray: mono audio sequence
        int: sample rate
    """
    file = sf.SoundFile(path)

    start = floor(start * file.samplerate) if start else 0
    frames = ceil(stop * file.samplerate) - start if stop else -1

    file.seek(start)
    audio = file.read(frames, always_2d=True)

    if audio.shape[-1] > 1:
        audio = np.mean(audio, axis=-1)
    else:
        audio = np.squeeze(audio)

    return audio, file.samplerate


def normalize_amplitude(audio, eps=1e-7):
    """Normalize amplitude using RMS

    Args:
        audio (np.array): audio sequence
        eps (float): epsilon for numerical stability

    Returns:
        np.array: normalized audio sequence
    """
    assert audio.ndim == 1
    audio -= np.mean(audio)
    audio /= np.std(audio) + eps
    return audio


def wave(example, correction, normalize, log_scale):
    """Read and preprocess an audio file to its raw waveform.

    Args:
        example: row representing the example
        correction: spectrum correction object or None
        normalize (bool): normalize the waveform
        log_scale (bool): use log scaling

    Returns:
        np.array: audio sequence
    """
    audio, sample_rate = read_wave(
        example.file,
        getattr(example, 'start', 0),
        getattr(example, 'stop', None)
    )

    if normalize:
        audio = normalize_amplitude(audio)

    if correction:
        audio = correction.transform_wave(audio, example.device)

    if log_scale:
        audio = np.sign(audio) * np.log(np.abs(audio) + 1)

    return audio


def spectrogram(example, correction, num_fft, hop_length, power, num_mels, htk, norm, normalize=False, eps=1e-7):
    """Read and preprocess an audio file to a log-melspectrogram.

    Args:
        example: row representing the example
        correction: spectrum correction object or None
        num_fft (int): number of frequency bins in STFT
        hop_length (int): length of the hop for STFT
        power (float): power of the spectrogram
        num_mels (int): number of mel bins in the final melspectrogram
        htk (bool): use HTK formula instead of Slaney
        norm (bool): normalization of the mel bands
        normalize (bool): normalize the waveform
        eps (float): epsilon for numerical stability

    Returns:
        np.array [shape=(num_mels, t)]: log-melspectrogram
    """
    audio, sample_rate = read_wave(
        example.file,
        getattr(example, 'start', 0),
        getattr(example, 'stop', None)
    )

    if normalize:
        audio = normalize_amplitude(audio)

    if correction:
        assert correction.num_fft == num_fft
        stft = correction.transform_wave(audio, example.device, return_stft=True)
    else:
        stft = librosa.core.stft(audio, n_fft=num_fft, hop_length=hop_length)

    spec = np.abs(stft)
    spec = librosa.feature.melspectrogram(
        S=spec ** power, sr=sample_rate, n_mels=num_mels, htk=htk, norm=1 if norm else None
    )

    spec = np.log(spec + eps)

    return spec


def compute_standardization(dataset, axes, chunk=128):
    """Compute standardization for the entire dataset.

    Args:
        dataset: array like
        axes (int/sequence of ints): which axes to standardize
        chunk (int): size of the chunk for processing

    Returns:
        np.array: mean
        np.array: standard deviation
    """
    chunks = list(dataset.shape)
    chunks[0] = chunk
    array = da.from_array(dataset, chunks=chunks)

    return (
        array.mean(axes, keepdims=True).compute(),
        array.std(axes, keepdims=True).compute(),
    )


def apply_standardization(dataset, standardization, chunk=128):
    """Standardize the entire dataset using previously computed mean and variance.

    Args:
        dataset: array like
        standardization: mean and variance
        chunk (int): size of the chunk for processing
    """
    mean, std = standardization
    chunks = list(dataset.shape)
    chunks[0] = chunk
    array = da.from_array(dataset, chunks=chunks)
    array = (array - mean) / std
    array.store(dataset)


def apply_grouped_standardization(dataset, groups, standardization, chunk=128):
    """Standardize the entire dataset using previously computed mean and variance.

    Args:
        dataset: array like
        standardization: mean and variance for each group
        chunk (int): size of the chunk for processing
    """
    chunks = list(dataset.shape)
    chunks[0] = 100
    array = da.from_array(dataset, chunks=chunks)
    groups = da.from_array(groups, chunks=chunk)

    # workaround for piecewise
    for key, (mean, std) in enumerate(standardization):
        selection = (groups == key).reshape([-1, 1, 1, 1])
        mean = da.where(selection, mean, 0)
        std = da.where(selection, std, 1)
        array = (array - mean) / std

    array.store(dataset)
