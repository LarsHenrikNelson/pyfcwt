"""Tests for pyfcwt.wavelet_utils helper functions."""

import numpy as np
import pytest

from pyfcwt.wavelet_utils import (
    f_to_s,
    s_to_f,
    fwhm_sigma,
    fwhm_freq,
    fwhm_to_cycles,
    gen_freqs,
    get_support,
    get_wavelet_length,
    get_amp_scale,
)


# ------------------------------------------------------------------ #
# f_to_s / s_to_f round-trip
# ------------------------------------------------------------------ #
class TestScaleFreqConversion:
    def test_scalar_round_trip(self):
        fc, fs, n = 10.0, 1000.0, 7.0
        s = f_to_s(fc, fs, n)
        fc_back = s_to_f(s, fs, n)
        assert pytest.approx(fc_back) == fc

    def test_array_round_trip(self):
        fc = np.array([5.0, 10.0, 50.0, 100.0])
        fs, n = 1000.0, 7.0
        s = f_to_s(fc, fs, n)
        fc_back = s_to_f(s, fs, n)
        np.testing.assert_allclose(fc_back, fc)

    def test_higher_freq_gives_smaller_scale(self):
        fs, n = 1000.0, 7.0
        assert f_to_s(100.0, fs, n) < f_to_s(10.0, fs, n)

    def test_default_n_cycles(self):
        """Default n_cycles should be 7."""
        assert f_to_s(10.0, 1000.0) == f_to_s(10.0, 1000.0, 7.0)


# ------------------------------------------------------------------ #
# FWHM helpers
# ------------------------------------------------------------------ #
class TestFWHM:
    def test_fwhm_sigma_positive(self):
        assert fwhm_sigma(1.0) > 0

    def test_fwhm_sigma_value(self):
        """FWHM of a Gaussian with sigma=1 should be ~2.355."""
        expected = 2 * np.sqrt(2 * np.log(2))
        assert pytest.approx(fwhm_sigma(1.0), rel=1e-6) == expected

    def test_fwhm_freq_increases_with_freq(self):
        """Higher centre frequency → narrower time envelope → wider FWHM."""
        assert fwhm_freq(50.0) < fwhm_freq(10.0)

    def test_fwhm_to_cycles_inverse(self):
        """fwhm_to_cycles should be consistent with fwhm_freq."""
        freq = 20.0
        fwhm = fwhm_freq(freq)
        cycles_back = fwhm_to_cycles(fwhm, [freq])
        # fwhm_freq returns sigma_t * fwhm_const; fwhm_to_cycles should
        # recover n_cycles when fed that fwhm value.
        assert pytest.approx(cycles_back[0], rel=1e-6) == 7.0


# ------------------------------------------------------------------ #
# gen_freqs
# ------------------------------------------------------------------ #
class TestGenFreqs:
    def test_linear_endpoints(self):
        freqs = gen_freqs(1.0, 100.0, 50, scaling="linear")
        assert pytest.approx(freqs[0]) == 1.0
        assert pytest.approx(freqs[-1]) == 100.0

    def test_log_endpoints(self):
        freqs = gen_freqs(1.0, 100.0, 50, scaling="log")
        assert pytest.approx(freqs[0]) == 1.0
        assert pytest.approx(freqs[-1]) == 100.0

    def test_length(self):
        freqs = gen_freqs(1.0, 100.0, 50)
        assert freqs.size == 50

    def test_log_spacing_is_not_linear(self):
        freqs = gen_freqs(1.0, 100.0, 50, scaling="log")
        diffs = np.diff(freqs)
        # log-spaced: differences should be increasing
        assert np.all(np.diff(diffs) > 0)

    def test_linear_spacing_is_uniform(self):
        freqs = gen_freqs(1.0, 100.0, 50, scaling="linear")
        diffs = np.diff(freqs)
        np.testing.assert_allclose(diffs, diffs[0], rtol=1e-10)


# ------------------------------------------------------------------ #
# get_support / get_wavelet_length
# ------------------------------------------------------------------ #
class TestSupport:
    def test_get_support_positive(self):
        assert get_support(1.0, 10.0) > 0

    def test_wavelet_length_is_odd(self):
        """Wavelet is centered → length must be odd."""
        length = get_wavelet_length(fc=20.0, fs=1000.0)
        assert length % 2 == 1

    def test_wavelet_length_grows_with_lower_freq(self):
        """Lower frequency → wider wavelet → longer support."""
        l_low = get_wavelet_length(fc=5.0, fs=1000.0)
        l_high = get_wavelet_length(fc=50.0, fs=1000.0)
        assert l_low > l_high


# ------------------------------------------------------------------ #
# get_amp_scale
# ------------------------------------------------------------------ #
class TestAmpScale:
    def test_returns_positive(self):
        assert get_amp_scale(10.0, 100.0) > 0

    def test_mother_equals_f0(self):
        """When f0 == f1 the scale should be ~1 (modulo small constant)."""
        s = get_amp_scale(50.0, 50.0)
        assert pytest.approx(s, rel=1e-3) == 1.0
