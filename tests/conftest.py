"""Shared fixtures for pyfcwt tests."""

import numpy as np
import pytest

from pyfcwt import Wavelet
from pyfcwt.frequencies import Frequencies


# ---------- common parameters ----------
FS = 1000.0          # sample rate (Hz)
DURATION = 1.0       # seconds
N_SAMPLES = int(FS * DURATION)
F_LOW = 1.0          # lowest analysis frequency
F_HIGH = 100.0       # highest analysis frequency
N_FREQS = 50         # number of frequency bins


# ---------- fixtures ----------

@pytest.fixture
def fs():
    return FS


@pytest.fixture
def n_samples():
    return N_SAMPLES


@pytest.fixture
def wavelet():
    """Default Morlet wavelet (imaginary, zero-mean)."""
    return Wavelet(fs=FS, n_cycles=7.0, gauss_sd=5.0, sigma=-1, zero_mean=True, imaginary=True)


@pytest.fixture
def wavelet_real():
    """Real-valued Morlet wavelet."""
    return Wavelet(fs=FS, n_cycles=7.0, gauss_sd=5.0, sigma=-1, zero_mean=True, imaginary=False)


@pytest.fixture
def wavelet_fixed_sigma():
    """Wavelet with a fixed sigma (not frequency-adaptive)."""
    return Wavelet(fs=FS, n_cycles=7.0, gauss_sd=5.0, sigma=10, zero_mean=True, imaginary=True)


@pytest.fixture
def frequencies(wavelet):
    """Log-spaced Frequencies object."""
    return Frequencies(wavelet=wavelet, f0=F_LOW, f1=F_HIGH, fn=N_FREQS, fs=FS, scaling="log")


@pytest.fixture
def frequencies_linear(wavelet):
    """Linearly-spaced Frequencies object."""
    return Frequencies(wavelet=wavelet, f0=F_LOW, f1=F_HIGH, fn=N_FREQS, fs=FS, scaling="linear")


@pytest.fixture
def sine_signal():
    """Pure 20 Hz sine wave."""
    t = np.arange(N_SAMPLES) / FS
    return np.sin(2 * np.pi * 20.0 * t)


@pytest.fixture
def multi_sine_signal():
    """Sum of 10 Hz and 50 Hz sine waves."""
    t = np.arange(N_SAMPLES) / FS
    return np.sin(2 * np.pi * 10.0 * t) + np.sin(2 * np.pi * 50.0 * t)


@pytest.fixture
def impulse_signal():
    """Unit impulse at the centre of the signal."""
    sig = np.zeros(N_SAMPLES)
    sig[N_SAMPLES // 2] = 1.0
    return sig
