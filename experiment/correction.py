"""Spectrum correction"""

import re
from abc import ABC, abstractmethod

import librosa
import numpy as np

from scipy import signal
from scipy.stats import gmean


class _SpectrumCorrection(ABC):
    """Base class for spectrum correction.

    Args:
        num_fft (int): FFT window size
        hop_length (int): Number audio of frames between STFT columns.
            If unspecified, defaults `num_fft / 4`.
        sub_mean (bool): If True subtracts mean from every recoring.
    """

    def __init__(self, num_fft, hop_length, sub_mean=True):
        self._num_fft = num_fft
        self._hop_length = hop_length
        self._sub_mean = sub_mean
        self.coefficients = None

    @property
    def num_fft(self):
        return self._num_fft

    @property
    def hop_length(self):
        return self._hop_length

    @property
    def sub_mean(self):
        return self._sub_mean

    def _stft(self, audio):
        if self._sub_mean:
            audio -= np.mean(audio)
        return librosa.stft(audio, n_fft=self._num_fft, hop_length=self._hop_length)

    def _correction(self, spec_a, spec_b):
        ratio = spec_a / spec_b
        return gmean(ratio, axis=-1)

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    def transform_stft(self, stft, device, frequency_axis=0):
        """Apply spectrum correction to a STFT.

        Args:
            stft (np.array): STFT of the signal
            device: which device produced the recording
            frequency_axis: which is the frequency axis

        Returns:
            np.array: transformed STFT
        """
        shape = np.ones(stft.ndim, int)
        shape[frequency_axis] = -1
        return stft * self.coefficients[device].reshape(shape)

    def transform_wave(self, recording, device, return_stft=False):
        """Apply spectrum correction to a raw waveform.

        Args:
            recording (np.array): raw audio sequence
            device: which device produced the recording
            return_stft: return the STFT of the signal instead of the waveform

        Returns:
            np.array: transformed waveform or STFT if selected
        """
        stft = self._stft(recording)
        stft = self.transform_stft(stft, device, frequency_axis=0)

        if return_stft:
            return stft
        else:
            return librosa.istft(stft, hop_length=self._hop_length, length=len(recording))


class AlignedSpectrumCorrection(_SpectrumCorrection):
    """Spectrum correction using aligned recordings
    and transformation in frequency domain (STFT).
    """

    def fit(self, aligned_segments, reference):
        """Computes coefficients for spectrum correction.

        Args:
            aligned_segments (iterable): Each element is a dictionary of aligned in time recordings,
                keys correspond to device names and values are raw waveforms.
            reference (str): Reference device.
        """
        coefficients = {}

        for segment in aligned_segments:
            stfts = {
                device: np.abs(self._stft(audio))
                for device, audio in segment.items()
                if not np.isnan(audio).any()
            }

            ref_stft = stfts[reference]

            for device, spec in stfts.items():
                # TODO possible div by zero + use less memory
                correction = self._correction(ref_stft, spec)
                coefficients.setdefault(device, []).append(correction)

        self.coefficients = {
            device: gmean(spectra, axis=0)
            for device, spectra in coefficients.items()
        }


class UnalignedSpectrumCorrection(_SpectrumCorrection):
    """Spectrum correction using unaligned recordings
    and transformation in frequency domain (STFT).
    """

    def fit(self, segments, reference=None):
        """Computes coefficients for spectrum correction.

        Args:
            segments (dict): A dictionary of iterables of recordings,
                keys correspond to device names.
        """
        self.coefficients = {
            device: gmean(np.array([
                self._correction(1, np.abs(self._stft(recording)))
                for recording in recordings
            ]), axis=0)
            for device, recordings in segments.items()
        }
        if reference:
            ref = self.coefficients[reference]
            self.coefficients = {
                device: coeffs / ref
                for device, coeffs in self.coefficients.items()
            }


class _FIRMixin(_SpectrumCorrection):
    """This is an alternative implementation that utilizes a FIR filter
    to transform waveforms directly in the time domain.

    Args:
        num_taps (int): number of taps in FIR filter
        num_fft (int): FFT window size
        hop_length (int): Number audio of frames between STFT columns.
            If unspecified, defaults `num_fft / 4`.
        sub_mean (bool): If True subtracts mean from every recoring.
    """

    def __init__(self, num_taps, num_fft, hop_length, sub_mean=True):
        super().__init__(num_fft, hop_length, sub_mean)
        self.num_taps = num_taps
        self.firs = None

    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)

        freqs = librosa.core.fft_frequencies(sr=2, n_fft=self.num_fft)
        self.firs = {
            device: signal.firls(self.num_taps, _bands(freqs), _bands(coefficients), fs=2)
            for device, coefficients in self.coefficients.items()
        }

    def transform_wave(self, recording, device, return_stft=False):
        recording = signal.lfilter(self.firs[device], 1, recording)
        if return_stft:
            return librosa.stft(recording, n_fft=self._num_fft, hop_length=self._hop_length)
        return recording


class AlignedSpectrumCorrectionFIR(_FIRMixin, AlignedSpectrumCorrection):
    pass


class UnalignedSpectrumCorrectionFIR(_FIRMixin, UnalignedSpectrumCorrection):
    pass


class DCASESpectrumCorrection:
    """Spectrum correction using spectrum computed from the averaged STFT.
    Rewrite of the implementation we used during DCASE 2019.

    Args:
        num_fft (int): FFT window size
        hop_length (int): Number audio of frames between STFT columns.
            If unspecified, defaults `num_fft / 4`.
        sub_mean (bool): If True subtracts mean from every recoring.
    """

    def __init__(self, num_fft, hop_length, sub_mean=True):
        self._num_fft = num_fft
        self._hop_length = hop_length
        self._sub_mean = sub_mean
        self.coefficients = None

    @property
    def num_fft(self):
        return self._num_fft

    @property
    def hop_length(self):
        return self._hop_length

    @property
    def sub_mean(self):
        return self._sub_mean

    def _spectrum(self, audio):
        if self._sub_mean:
            audio -= np.mean(audio)
        stft = librosa.stft(audio, n_fft=self._num_fft, hop_length=self._hop_length)
        return np.mean(np.abs(stft), axis=-1)

    def _reduce(self, spectra, use_median, axis):
        if use_median:
            return np.median(spectra, axis=axis)
        else:
            return np.mean(spectra, axis=axis)

    def fit(self, aligned_segments, reference, regex=False, use_median=False):
        """Computes coefficients for spectrum correction.

        Args:
            aligned_segments (iterable): Each element is a dictionary of aligned in time recordings,
                keys correspond to device names and values are raw waveforms.
                Waves for non-reference devices can be missing.
            reference (str/list/None): Reference device or list of reference devices.
                When `None` all devices are used as reference.
            regex (bool): If `True` assume `reference` is a regex.
            use_median (bool): If `True` use median instead of mean.
        """
        coefficients = {}

        for segment in aligned_segments:
            spectra = {
                device: self._spectrum(audio)
                for device, audio in segment.items()
                if not np.isnan(audio).any()
            }

            base_spec = self._reference_spectrum(reference, spectra, regex=regex)

            for device, spec in spectra.items():
                # TODO possible div by zero + use less memory
                coefficients.setdefault(device, []).append(base_spec / spec)

        self.coefficients = {
            device: self._reduce(spectra, use_median, axis=0)
            for device, spectra in coefficients.items()
        }

    def _reference_spectrum(self, reference, spectra, regex=False):
        if regex:
            spectra = [spectra[key] for key in spectra if re.match(reference, key)]
        else:
            spectra = [spectra[key] for key in spectra if reference == key]

        if not spectra:
            raise RuntimeError('Unamble to find reference spectra.')

        return np.mean(spectra, axis=0)

    def transform_stft(self, stft, device, frequency_axis=0):
        """Apply spectrum correction to a STFT.

        Args:
            stft (np.array): STFT of the signal
            device: which device produced the recording
            frequency_axis: which is the frequency axis

        Returns:
            np.array: transformed STFT
        """
        shape = np.ones(stft.ndim, int)
        shape[frequency_axis] = -1
        return stft * self.coefficients[device].reshape(shape)

    def transform_wave(self, recording, device, return_stft=False):
        """Apply spectrum correction to a raw waveform.

        Args:
            recording (np.array): raw audio sequence
            device: which device produced the recording
            return_stft: return the STFT of the signal instead of the waveform

        Returns:
            np.array: transformed waveform or STFT if selected
        """
        if self._sub_mean:
            recording = recording - np.mean(recording)

        stft = librosa.stft(recording, n_fft=self._num_fft, hop_length=self._hop_length)
        stft = self.transform_stft(stft, device, frequency_axis=0)

        if return_stft:
            return stft
        return librosa.istft(stft, hop_length=self._hop_length, length=len(recording))


def _bands(freqs):
    return np.stack([freqs[:-1], freqs[1:]], axis=-1)
