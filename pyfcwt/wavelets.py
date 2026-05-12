import numpy as np
import pyfftw
from numba import njit

from .wavelet_utils import f_to_s


class Wavelet:
    """Morlet wavelet generator for use with the fast continuous wavelet
    transform.

    The wavelet is a complex sinusoid windowed by a Gaussian envelope.
    It can be evaluated in the time domain via :meth:`time` or in the
    frequency domain via :meth:`frequency`.

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    n_cycles : float, optional
        Number of cycles in the wavelet, controls the
        time-frequency trade-off.  Higher values give better frequency
        resolution at the cost of time resolution.  Default is ``7.0``.
    gauss_sd : float, optional
        Number of standard deviations of the Gaussian envelope to
        include on each side of the wavelet center.  Default is ``5.0``.
    sigma : int, optional
        Fixed bandwidth parameter.  When set to ``-1`` (the default) the
        bandwidth adapts to the center frequency (frequency-adaptive
        mode).  Any positive value fixes the bandwidth across all
        frequencies.
    zero_mean : bool, optional
        If ``True`` (default), subtract a correction term so the real
        part of the wavelet has zero mean (admissibility condition).
    imaginary : bool, optional
        If ``True`` (default), the wavelet is complex-valued
        (analytic).  If ``False``, the frequency-domain representation
        is returned as its magnitude (real-valued).
    """

    def __init__(
        self,
        fs: float,
        n_cycles: float = 7.0,
        gauss_sd: float = 5.0,
        sigma: int = -1,
        zero_mean: bool = True,
        imaginary: bool = True,
    ):
        self.fs = fs
        self.n_cycles = n_cycles
        self.gauss_sd = gauss_sd
        self.sigma = sigma
        self.zero_mean = zero_mean
        self.imaginary = imaginary

    def scale(self, fc: float):
        """Convert a center frequency to the corresponding wavelet scale.

        Parameters
        ----------
        fc : float
            Center frequency in Hz.

        Returns
        -------
        float
            Wavelet scale.
        """
        return f_to_s(fc, self.fs, self.n_cycles)

    def time(self, fc: float):
        """Generate the wavelet in the time domain.

        Constructs a Morlet wavelet centerd at frequency *fc*, sampled
        at ``self.fs``.  The wavelet extends ``gauss_sd`` standard
        deviations on each side, is optionally zero-mean corrected,
        and is normalised to unit energy (L2 norm = ``sqrt(2)``).

        Parameters
        ----------
        fc : float
            Center frequency of the wavelet in Hz.

        Returns
        -------
        np.ndarray
            Complex-valued 1-D array of length :meth:`length(fc) <length>`.
        """
        inv_fs = 1.0 / self.fs

        # Fractional bandwidth in the frequency domain
        if self.sigma == -1:
            sigma_t = self.n_cycles / (2.0 * np.pi * fc)
        else:
            sigma_t = self.n_cycles / (2.0 * np.pi * self.sigma)

        # Go gauss_sd STDEVs out on each side
        num_values = int((self.gauss_sd * sigma_t) // inv_fs)
        t = np.arange(-num_values, num_values + 1) / self.fs
        oscillation = np.exp(2.0 * 1j * np.pi * fc * t)
        if self.zero_mean:
            real_offset = np.exp(-2 * (np.pi * fc * sigma_t) ** 2)
            oscillation -= real_offset
        gaussian_env = np.exp(-(t**2) / (2.0 * sigma_t**2))
        oscillation *= gaussian_env
        oscillation /= np.sqrt(0.5) * np.linalg.norm(oscillation, ord=2)
        return oscillation

    def frequency(self, fc: float, size: int, dtype: str = "complex128"):
        """Generate the wavelet in the frequency domain.

        Computes the FFT of the time-domain wavelet zero-padded to
        *size* samples.  For an imaginary (analytic) wavelet the full
        complex spectrum is returned; for a real wavelet the magnitude
        spectrum is returned instead.

        Parameters
        ----------
        fc : float
            Center frequency of the wavelet in Hz.
        size : int
            FFT length (number of frequency bins).
        dtype : str, optional
            NumPy complex dtype for the FFT buffers.
            Default is ``"complex128"``.

        Returns
        -------
        np.ndarray
            1-D array of length *size* containing the frequency-domain
            wavelet representation.
        """
        wavelet = self.time(fc)
        a = pyfftw.zeros_aligned(size, dtype=dtype)
        b = pyfftw.zeros_aligned(size, dtype=dtype)

        # Plan BEFORE filling a — FFTW_MEASURE destroys buffer contents.
        forward_fft = pyfftw.FFTW(a, b, threads=1)
        a[: wavelet.size] = wavelet
        forward_fft()
        if self.imaginary:
            return b
        else:
            return np.abs(b)

    def length(self, fc: float):
        """Return the number of samples in the time-domain wavelet.

        The length is always odd (the wavelet is symmetric about its
        center sample) and depends on the center frequency, the
        sampling rate, and the ``gauss_sd`` truncation parameter.

        Parameters
        ----------
        fc : float
            Center frequency of the wavelet in Hz.

        Returns
        -------
        int
            Number of samples (always odd).
        """
        inv_fs = 1.0 / self.fs

        # Fractional bandwidth in the frequency domain
        if self.sigma == -1:
            sigma_t = self.n_cycles / (2.0 * np.pi * fc)
        else:
            sigma_t = self.n_cycles / (2.0 * np.pi * self.sigma)

        # Go gauss_sd STDEVs out on each side
        num_values = int((self.gauss_sd * sigma_t) // inv_fs)
        return num_values * 2 + 1
