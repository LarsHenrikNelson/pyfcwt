"""Tests for pyfcwt.fcwt_cwt – the core CWT engine and spectral helpers."""

import numpy as np
import pytest

from pyfcwt.fcwt_cwt import PyFCWT, daughter_wavelet_multiplication
from pyfcwt.frequencies import Frequencies
from pyfcwt.wavelets import Wavelet


FS = 1000.0
DURATION = 1.0
N_SAMPLES = int(FS * DURATION)


# ---------- helpers ----------

def _make_fcwt(
    fs=FS,
    f0=1.0,
    f1=100.0,
    fn=50,
    n_cycles=7.0,
    dtype="complex128",
    imaginary=True,
    norm=True,
    scaling="log",
):
    w = Wavelet(fs=fs, n_cycles=n_cycles, imaginary=imaginary)
    f = Frequencies(wavelet=w, f0=f0, f1=f1, fn=fn, fs=fs, scaling=scaling)
    return PyFCWT(wavelet=w, frequencies=f, threads=-1, norm=norm, dtype=dtype)


def _sine(freq, fs=FS, n=N_SAMPLES):
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * freq * t)


# ------------------------------------------------------------------ #
# daughter_wavelet_multiplication
# ------------------------------------------------------------------ #
class TestDaughterWaveletMultiplication:
    def test_output_nonzero(self):
        n = 64
        inp = np.ones(n, dtype=np.complex128)
        mother = np.ones(n, dtype=np.complex128)
        out = np.zeros(n, dtype=np.complex128)
        daughter_wavelet_multiplication(inp, out, mother, scale=1.0, mscale=1.0)
        assert np.any(out != 0)

    def test_zero_input_gives_zero_output(self):
        n = 64
        inp = np.zeros(n, dtype=np.complex128)
        mother = np.ones(n, dtype=np.complex128)
        out = np.zeros(n, dtype=np.complex128)
        daughter_wavelet_multiplication(inp, out, mother, scale=1.0, mscale=1.0)
        np.testing.assert_array_equal(out, 0)


# ------------------------------------------------------------------ #
# PyFCWT initialisation
# ------------------------------------------------------------------ #
class TestPyFCWTInit:
    def test_default_threads(self):
        import os
        fcwt = _make_fcwt()
        assert fcwt.threads == os.cpu_count() // 2

    def test_explicit_threads(self):
        w = Wavelet(fs=FS)
        f = Frequencies(wavelet=w, f0=1.0, f1=100.0, fn=10, fs=FS)
        engine = PyFCWT(wavelet=w, frequencies=f, threads=2)
        assert engine.threads == 2

    def test_complex64_dtype(self):
        fcwt = _make_fcwt(dtype="complex64")
        assert fcwt.n_cdtype == np.complex64
        assert fcwt.n_fdtype == np.float32

    def test_complex128_dtype(self):
        fcwt = _make_fcwt(dtype="complex128")
        assert fcwt.n_cdtype == np.complex128
        assert fcwt.n_fdtype == np.float64


# ------------------------------------------------------------------ #
# CWT output shape and dtype
# ------------------------------------------------------------------ #
class TestCWTOutput:
    def test_shape(self):
        fcwt = _make_fcwt(fn=30)
        sig = _sine(20.0)
        result = fcwt.cwt(sig)
        assert result.shape == (30, N_SAMPLES)

    def test_dtype_complex128(self):
        fcwt = _make_fcwt(dtype="complex128")
        result = fcwt.cwt(_sine(20.0))
        assert result.dtype == np.complex128

    def test_dtype_complex64(self):
        fcwt = _make_fcwt(dtype="complex64")
        result = fcwt.cwt(_sine(20.0))
        assert result.dtype == np.complex64

    def test_no_nans(self):
        fcwt = _make_fcwt()
        result = fcwt.cwt(_sine(20.0))
        assert not np.any(np.isnan(result))


# ------------------------------------------------------------------ #
# CWT detects known frequencies
# ------------------------------------------------------------------ #
class TestCWTFrequencyDetection:
    def test_peak_at_20hz(self):
        """CWT power should peak near 20 Hz for a 20 Hz sine."""
        fcwt = _make_fcwt(f0=5.0, f1=80.0, fn=60)
        sig = _sine(20.0)
        result = fcwt.cwt(sig)
        power = np.mean(np.abs(result) ** 2, axis=1)
        peak_idx = np.argmax(power)
        peak_freq = fcwt.frequencies.f[peak_idx]
        assert abs(peak_freq - 20.0) < 3.0

    def test_two_frequencies(self):
        """CWT should show two power peaks for a two-tone signal."""
        fcwt = _make_fcwt(f0=5.0, f1=80.0, fn=80)
        t = np.arange(N_SAMPLES) / FS
        sig = np.sin(2 * np.pi * 15.0 * t) + np.sin(2 * np.pi * 60.0 * t)
        result = fcwt.cwt(sig)
        power = np.mean(np.abs(result) ** 2, axis=1)

        # Normalise power to find prominent peaks
        power_norm = power / power.max()
        freqs = fcwt.frequencies.f
        # There should be bins above 0.3 near both 15 Hz and 60 Hz
        near_15 = np.any((np.abs(freqs - 15.0) < 5.0) & (power_norm > 0.3))
        near_60 = np.any((np.abs(freqs - 60.0) < 5.0) & (power_norm > 0.3))
        assert near_15 and near_60


# ------------------------------------------------------------------ #
# CWT with different options
# ------------------------------------------------------------------ #
class TestCWTOptions:
    def test_unnormalized(self):
        """norm=False should give larger values than norm=True."""
        sig = _sine(20.0)
        fcwt_norm = _make_fcwt(norm=True)
        fcwt_raw = _make_fcwt(norm=False)
        r_norm = fcwt_norm.cwt(sig)
        r_raw = fcwt_raw.cwt(sig)
        assert np.mean(np.abs(r_raw)) > np.mean(np.abs(r_norm))

    def test_real_wavelet(self):
        """CWT with a real (non-imaginary) wavelet should still run."""
        fcwt = _make_fcwt(imaginary=False)
        sig = _sine(20.0)
        result = fcwt.cwt(sig)
        assert result.shape[1] == N_SAMPLES
        assert not np.any(np.isnan(result))

    def test_linear_scaling(self):
        fcwt = _make_fcwt(scaling="linear")
        result = fcwt.cwt(_sine(20.0))
        assert result.shape == (50, N_SAMPLES)

    def test_reuse_engine(self):
        """Calling cwt twice should produce the same result (mother cached)."""
        fcwt = _make_fcwt()
        sig = _sine(20.0)
        r1 = fcwt.cwt(sig)
        r2 = fcwt.cwt(sig)
        np.testing.assert_array_equal(r1, r2)

    def test_input_dtype_cast(self):
        """Engine should accept float32 input without error."""
        fcwt = _make_fcwt(dtype="complex128")
        sig = _sine(20.0).astype(np.float32)
        result = fcwt.cwt(sig)
        assert result.dtype == np.complex128

    def test_short_signal(self):
        """Very short signals should not crash."""
        fcwt = _make_fcwt(f0=50.0, f1=200.0, fn=10)
        sig = _sine(100.0, n=64)
        result = fcwt.cwt(sig)
        assert result.shape == (10, 64)


# ------------------------------------------------------------------ #
# Spectral helpers: amplitude, power, psd, asd, enbw
# ------------------------------------------------------------------ #
class TestSpectralHelpers:
    @pytest.fixture
    def engine_and_cwt(self):
        fcwt = _make_fcwt(fn=20)
        sig = _sine(20.0)
        cwt_result = fcwt.cwt(sig)
        return fcwt, cwt_result

    def test_amplitude_shape(self, engine_and_cwt):
        fcwt, cwt_result = engine_and_cwt
        amp = fcwt.amplitude(cwt_result)
        assert amp.shape == cwt_result.shape

    def test_amplitude_nonneg(self, engine_and_cwt):
        fcwt, cwt_result = engine_and_cwt
        assert np.all(fcwt.amplitude(cwt_result) >= 0)

    def test_power_is_amp_squared(self, engine_and_cwt):
        fcwt, cwt_result = engine_and_cwt
        np.testing.assert_allclose(
            fcwt.power(cwt_result),
            fcwt.amplitude(cwt_result) ** 2,
        )

    def test_power_nonneg(self, engine_and_cwt):
        fcwt, cwt_result = engine_and_cwt
        assert np.all(fcwt.power(cwt_result) >= 0)

    def test_enbw_shape(self, engine_and_cwt):
        fcwt, _ = engine_and_cwt
        enbw = fcwt.enbw()
        assert enbw.shape == (fcwt.frequencies.f.size,)

    def test_enbw_positive(self, engine_and_cwt):
        fcwt, _ = engine_and_cwt
        assert np.all(fcwt.enbw() > 0)

    def test_psd_shape(self, engine_and_cwt):
        fcwt, cwt_result = engine_and_cwt
        psd = fcwt.psd(cwt_result)
        assert psd.shape == cwt_result.shape

    def test_psd_nonneg(self, engine_and_cwt):
        fcwt, cwt_result = engine_and_cwt
        assert np.all(fcwt.psd(cwt_result) >= 0)

    def test_asd_is_sqrt_psd(self, engine_and_cwt):
        fcwt, cwt_result = engine_and_cwt
        np.testing.assert_allclose(
            fcwt.asd(cwt_result),
            np.sqrt(fcwt.psd(cwt_result)),
        )


class TestSigmaF:
    def test_adaptive_sigma(self):
        """Frequency-adaptive: sigma_f = f / n_cycles."""
        fcwt = _make_fcwt(fn=20)
        sf = fcwt._sigma_f()
        expected = fcwt.frequencies.f / fcwt.wavelet.n_cycles
        np.testing.assert_allclose(sf, expected)

    def test_fixed_sigma(self):
        """Fixed sigma: sigma_f should be constant."""
        w = Wavelet(fs=FS, sigma=10)
        f = Frequencies(wavelet=w, f0=1.0, f1=100.0, fn=20, fs=FS)
        fcwt = PyFCWT(wavelet=w, frequencies=f, threads=1)
        sf = fcwt._sigma_f()
        assert np.all(sf == sf[0])
