"""Tests for pyfcwt.wavelets.Wavelet class."""

import numpy as np
import pytest

from pyfcwt.wavelets import Wavelet


FS = 1000.0


class TestWaveletInit:
    def test_default_parameters(self):
        w = Wavelet(fs=FS)
        assert w.fs == FS
        assert w.n_cycles == 7.0
        assert w.gauss_sd == 5.0
        assert w.sigma == -1
        assert w.zero_mean is True
        assert w.imaginary is True


class TestWaveletScale:
    def test_scale_positive(self):
        w = Wavelet(fs=FS)
        assert w.scale(20.0) > 0

    def test_scale_inverse_freq(self):
        w = Wavelet(fs=FS)
        assert w.scale(50.0) < w.scale(10.0)


class TestWaveletTime:
    @pytest.fixture
    def wavelet(self):
        return Wavelet(fs=FS, n_cycles=7.0)

    def test_output_is_complex(self, wavelet):
        wt = wavelet.time(20.0)
        assert np.iscomplexobj(wt)

    def test_length_is_odd(self, wavelet):
        wt = wavelet.time(20.0)
        assert wt.size % 2 == 1

    def test_length_matches_length_method(self, wavelet):
        fc = 20.0
        wt = wavelet.time(fc)
        assert wt.size == wavelet.length(fc)

    def test_unit_energy(self, wavelet):
        """After normalisation the L2 norm should be sqrt(2)."""
        wt = wavelet.time(20.0)
        norm = np.linalg.norm(wt, ord=2)
        assert pytest.approx(norm, rel=1e-6) == np.sqrt(2)

    def test_zero_mean_removes_dc(self, wavelet):
        """With zero_mean=True the real part should integrate to ~0."""
        wt = wavelet.time(20.0)
        dc = np.abs(np.sum(wt.real))
        assert dc < 1e-3  # nearly zero

    def test_non_zero_mean_wavelet(self):
        w = Wavelet(fs=FS, zero_mean=False)
        wt = w.time(20.0)
        # Still valid output
        assert wt.size == w.length(20.0)

    def test_fixed_sigma(self):
        """Fixed-sigma wavelet should still produce valid output."""
        w = Wavelet(fs=FS, sigma=10)
        wt = w.time(20.0)
        assert np.iscomplexobj(wt)
        assert wt.size > 0

    def test_different_freqs_different_lengths(self, wavelet):
        len_low = wavelet.time(5.0).size
        len_high = wavelet.time(50.0).size
        assert len_low > len_high


class TestWaveletFrequency:
    @pytest.fixture
    def wavelet(self):
        return Wavelet(fs=FS)

    def test_output_length(self, wavelet):
        size = 1024
        wf = wavelet.frequency(20.0, size)
        assert wf.size == size

    def test_output_is_complex_for_imaginary_wavelet(self, wavelet):
        wf = wavelet.frequency(20.0, 1024)
        assert np.iscomplexobj(wf)

    def test_output_is_real_magnitude_for_real_wavelet(self):
        w = Wavelet(fs=FS, imaginary=False)
        wf = w.frequency(20.0, 1024)
        # Should be real (abs was applied)
        np.testing.assert_allclose(wf.imag, 0.0, atol=1e-15)

    def test_peak_near_centre_frequency(self, wavelet):
        """The FFT of the wavelet should peak near the centre frequency."""
        size = 4096
        wf = wavelet.frequency(20.0, size, dtype="complex128")
        freqs = np.fft.fftfreq(size, d=1.0 / FS)
        mag = np.abs(wf)
        peak_idx = np.argmax(mag)
        peak_freq = np.abs(freqs[peak_idx])
        # Should be within 2 Hz of the requested 20 Hz
        assert abs(peak_freq - 20.0) < 2.0

    def test_dtype_complex64(self, wavelet):
        wf = wavelet.frequency(20.0, 1024, dtype="complex64")
        assert wf.dtype == np.complex64


class TestWaveletLength:
    def test_positive(self):
        w = Wavelet(fs=FS)
        assert w.length(20.0) > 0

    def test_odd(self):
        w = Wavelet(fs=FS)
        assert w.length(20.0) % 2 == 1

    def test_consistent_with_scale(self):
        w = Wavelet(fs=FS)
        # Lower freq → larger scale → longer wavelet
        assert w.length(5.0) > w.length(50.0)
