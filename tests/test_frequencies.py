"""Tests for pyfcwt.frequencies.Frequencies class."""

import numpy as np
import pytest

from pyfcwt.frequencies import Frequencies
from pyfcwt.wavelets import Wavelet


FS = 1000.0


@pytest.fixture
def wavelet():
    return Wavelet(fs=FS)


class TestFrequenciesInit:
    def test_log_scaling_length(self, wavelet):
        freq = Frequencies(wavelet=wavelet, f0=1.0, f1=100.0, fn=50, fs=FS, scaling="log")
        assert freq.f.size == 50
        assert freq.s.size == 50

    def test_linear_scaling_length(self, wavelet):
        freq = Frequencies(wavelet=wavelet, f0=1.0, f1=100.0, fn=50, fs=FS, scaling="linear")
        assert freq.f.size == 50

    def test_stores_parameters(self, wavelet):
        freq = Frequencies(wavelet=wavelet, f0=1.0, f1=100.0, fn=50, fs=FS, scaling="log")
        assert freq.f0 == 1.0
        assert freq.f1 == 100.0
        assert freq.fn == 50
        assert freq.fs == FS
        assert freq.scaling == "log"


class TestFrequencyValues:
    def test_log_endpoints(self, wavelet):
        freq = Frequencies(wavelet=wavelet, f0=1.0, f1=100.0, fn=50, fs=FS, scaling="log")
        # Reversed, so first element is highest frequency
        assert pytest.approx(freq.f[0]) == 100.0
        assert pytest.approx(freq.f[-1]) == 1.0

    def test_linear_endpoints(self, wavelet):
        freq = Frequencies(wavelet=wavelet, f0=1.0, f1=100.0, fn=50, fs=FS, scaling="linear")
        assert pytest.approx(freq.f[0]) == 100.0
        assert pytest.approx(freq.f[-1]) == 1.0

    def test_descending_order(self, wavelet):
        """Frequencies should be stored highest-first (descending)."""
        freq = Frequencies(wavelet=wavelet, f0=1.0, f1=100.0, fn=50, fs=FS, scaling="log")
        assert np.all(np.diff(freq.f) < 0)

    def test_scales_ascending(self, wavelet):
        """Scales should be ascending (since freqs are descending)."""
        freq = Frequencies(wavelet=wavelet, f0=1.0, f1=100.0, fn=50, fs=FS, scaling="log")
        assert np.all(np.diff(freq.s) > 0)

    def test_all_positive(self, wavelet):
        freq = Frequencies(wavelet=wavelet, f0=1.0, f1=100.0, fn=50, fs=FS, scaling="log")
        assert np.all(freq.f > 0)
        assert np.all(freq.s > 0)


class TestFrequencyScaleConsistency:
    def test_f_and_s_correspond(self, wavelet):
        """Each scale should be f_to_s of the corresponding frequency."""
        from pyfcwt.wavelet_utils import f_to_s

        freq = Frequencies(wavelet=wavelet, f0=1.0, f1=100.0, fn=50, fs=FS, scaling="log")
        expected_s = f_to_s(freq.f, FS, wavelet.n_cycles)
        np.testing.assert_allclose(freq.s, expected_s)
