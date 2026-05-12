import os
from typing import Literal

import numpy as np
import pyfftw
from numba import njit, prange

from .frequencies import Frequencies
from .wavelets import Wavelet


@njit(cache=True, parallel=True)
def daughter_wavelet_multiplication(
    input_fft: np.ndarray,
    output: np.ndarray,
    mother: np.ndarray,
    scale: float,
    mscale: float,
    doublesided: bool = False,
):
    """FFT wavelet convolution from fCWT using numpy. That utilizes
    numpy pass-by assignment as pass-by references/pointer.

    Parameters
    ----------
    input_fft : np.ndarray
        FFT of the signal of interest, must be a 1D signal
    output : np.ndarray
        Pre allocated output array
    mother : np.ndarray
        Mother wavelet
    scale : float
        Scale to convolve input with
    threads : int, optional
        Number of threads to use, by default 1
    doublesided : bool, optional
        Whether the wavelet is has a doubleside FFT (not in use), by default False
    """
    isize = input_fft.size
    isizef = float(isize)
    endpointf = min(isizef / mscale, (isizef * mscale) / scale)
    step = scale / mscale
    endpoint = int(endpointf)
    batchsize = endpoint
    maximum = isizef - 1.0

    for q1 in prange(0, int(batchsize)):
        tmp = min(maximum, step * q1)

        output[q1] = input_fft[q1] * mother[int(tmp)]
    # if doublesided:
    #     for q1 in prange(0, int(batchsize)):
    #         tmp = min(mm, step * q1)

    #         output[s1 - q1] = input_fft[s1 - q1].real * mother[int(tmp)] + input_fft[
    #             s1 - q1
    #         ].imag * mother[int(tmp)] * (1j - 2 * imaginary)


class PyFCWT:
    """Fast continuous wavelet transform engine.

    Performs the CWT by multiplying daughter wavelets with the input
    signal in the frequency domain using FFTW for the forward and
    inverse transforms and Numba for the parallelised wavelet
    convolution.

    The mother wavelet and FFTW plans are cached between calls to
    :meth:`cwt` so that repeated transforms on signals of the same
    length are efficient.

    Parameters
    ----------
    wavelet : Wavelet
        Morlet wavelet instance that defines the mother wavelet.
    frequencies : Frequencies
        Frequency / scale grid over which the CWT is computed.
    threads : int, optional
        Number of threads used by FFTW.  ``-1`` (default) uses half
        of the available CPU cores.
    norm : bool, optional
        If ``True`` (default), normalise the CWT output by the FFT
        length so that amplitudes are comparable across signal sizes.
    dtype : {"complex128", "complex64"}, optional
        Precision of the internal FFT buffers and the output array.
        Default is ``"complex128"``.
    """

    def __init__(
        self,
        wavelet: Wavelet,
        frequencies: Frequencies,
        threads: int = -1,
        norm: bool = True,
        dtype: Literal["complex128", "complex64"] = "complex128",
    ):
        self.frequencies = frequencies
        self.wavelet = wavelet
        self.mother = None
        self.mscale = None
        if threads == -1:
            self.threads: int = int(os.cpu_count()) // 2
        else:
            self.threads = threads
        self.norm = norm
        if dtype == "complex128":
            self.fftw_fdtype = "float64"
            self.fftw_cdtype = "complex128"
            self.n_fdtype = np.float64
            self.n_cdtype = np.complex128
        else:
            self.fftw_fdtype = "float32"
            self.fftw_cdtype = "complex64"
            self.n_fdtype = np.float32
            self.n_cdtype = np.complex64
        self._cached_size: int | None = None
        self.a = None
        self.b = None
        self.Ihat = None
        self.c = None
        self.d = None
        self._forward_fft = None
        self._backward_fft = None

        # Warmup: force Numba JIT compilation and parallel runtime
        # initialization so the first real call doesn't return zeros.
        _warmup = np.zeros(4, dtype=self.n_cdtype)
        daughter_wavelet_multiplication(_warmup, _warmup.copy(), _warmup, 1.0, 1.0)

    def cwt(
        self,
        input_data: np.ndarray,
    ):
        """Compute the continuous wavelet transform of a 1-D signal.

        The input is zero-padded to the next FFTW-friendly length,
        transformed into the frequency domain, convolved with scaled
        daughter wavelets, and transformed back.  The mother wavelet
        and associated scale are cached so that subsequent calls with
        the same signal length reuse the existing plan.

        Parameters
        ----------
        input_data : np.ndarray
            Real-valued 1-D input signal.  If the dtype does not match
            the engine's configured precision it is cast automatically.

        Returns
        -------
        np.ndarray
            Complex CWT coefficient matrix with shape
            ``(n_freqs, n_samples)`` where *n_freqs* is the number of
            frequency bins in ``self.frequencies`` and *n_samples* is
            ``input_data.size``.
        """
        size = input_data.size
        max_w_size = self.wavelet.length(self.frequencies.f[-1])
        newsize = pyfftw.next_fast_len(size + max_w_size - 1)

        if self.mother is None or self.mother.size != newsize:
            self.mother = self.wavelet.frequency(
                self.frequencies.f[0], newsize, dtype=self.fftw_cdtype
            )
            self.mscale = self.frequencies.s[0]

        if input_data.dtype != self.n_fdtype:
            input_data = input_data.astype(self.n_fdtype)

        # Allocate aligned buffers and plan FFTs only when the padded
        # size changes.  FFTW planning (FFTW_MEASURE) is expensive, so
        # reusing plans across calls is critical for performance.
        if self._cached_size != newsize:
            self.a = pyfftw.zeros_aligned(newsize, dtype=self.fftw_fdtype)
            self.b = pyfftw.zeros_aligned(newsize // 2 + 1, dtype=self.fftw_cdtype)
            self.Ihat = pyfftw.zeros_aligned(newsize, dtype=self.fftw_cdtype)
            self.c = pyfftw.zeros_aligned(newsize, dtype=self.fftw_cdtype)
            self.d = pyfftw.zeros_aligned(newsize, dtype=self.fftw_cdtype)

            # Plan BEFORE filling buffers — FFTW_MEASURE destroys contents.
            self._forward_fft = pyfftw.FFTW(
                self.a, self.b, threads=self.threads
            )
            self._backward_fft = pyfftw.FFTW(
                self.c, self.d, direction="FFTW_BACKWARD", threads=self.threads
            )
            self._cached_size = newsize

        self.a[:size] = input_data
        self._forward_fft()

        self.Ihat[: newsize // 2 + 1] = self.b

        if newsize % 2 == 0:
            # Even length: exclude DC and Nyquist components
            self.Ihat[newsize // 2 + 1 :] = np.conjugate(self.b[1 : newsize // 2][::-1])
        else:
            # Odd length: exclude only DC component
            self.Ihat[newsize // 2 + 1 :] = np.conjugate(self.b[1 : newsize // 2 + 1][::-1])

        self.c[:] = 0
        mother_half_len = self.wavelet.length(self.frequencies.f[0]) // 2
        n_scales = self.frequencies.s.size
        cwt = np.zeros((n_scales, size), dtype=self.n_cdtype)
        # Iterate from largest scale (lowest freq, smallest batchsize) to
        # smallest scale (highest freq, largest batchsize) so that each
        # iteration's batchsize >= the previous one, naturally overwriting
        # all prior values in c without needing to zero the buffer.
        for index in range(n_scales - 1, -1, -1):
            s = self.frequencies.s[index]
            daughter_wavelet_multiplication(
                input_fft=self.Ihat,
                output=self.c,
                mother=self.mother,
                scale=s,
                mscale=self.mscale,
            )
            self._backward_fft()

            if not self.wavelet.imaginary:
                cwt[index, :] = self.d[:size]
            else:
                cwt[index, :] = self.d[mother_half_len : size + mother_half_len]

        if self.norm:
            cwt /= newsize

        return cwt

    def _sigma_f(self) -> np.ndarray:
        """Frequency-domain standard deviation of the wavelet at each
        analysis frequency.

        For a Morlet wavelet the time-domain Gaussian has
        ``sigma_t = n_cycles / (2 * pi * fc)`` (frequency-adaptive) or
        ``sigma_t = n_cycles / (2 * pi * sigma)`` (fixed).  The
        corresponding frequency-domain std dev is
        ``sigma_f = 1 / (2 * pi * sigma_t)``.

        Returns
        -------
        np.ndarray
            sigma_f for every frequency in ``self.frequencies.f``.
        """
        n_cycles = self.wavelet.n_cycles
        if self.wavelet.sigma == -1:
            # Frequency-adaptive: sigma_f = fc / n_cycles
            return self.frequencies.f / n_cycles
        else:
            # Fixed sigma: sigma_f = sigma / n_cycles  (constant)
            return np.full_like(self.frequencies.f, self.wavelet.sigma / n_cycles)

    def enbw(self) -> np.ndarray:
        """Equivalent noise bandwidth of the wavelet at each analysis
        frequency.

        For a Gaussian with std dev ``sigma_f`` the ENBW is
        ``sigma_f * sqrt(pi)``.

        Returns
        -------
        np.ndarray
            ENBW in Hz, shape ``(n_freqs,)``.
        """
        return self._sigma_f() * np.sqrt(np.pi)

    def amplitude(self, cwt_result: np.ndarray) -> np.ndarray:
        """Amplitude spectrum from CWT coefficients: ``|W(f, t)|``.

        Parameters
        ----------
        cwt_result : np.ndarray
            CWT output, shape ``(n_freqs, n_times)``.

        Returns
        -------
        np.ndarray
            Amplitude in signal units, same shape as *cwt_result*.
        """
        return np.abs(cwt_result)

    def power(self, cwt_result: np.ndarray) -> np.ndarray:
        """Power spectrum from CWT coefficients: ``|W(f, t)|^2``.

        Parameters
        ----------
        cwt_result : np.ndarray
            CWT output, shape ``(n_freqs, n_times)``.

        Returns
        -------
        np.ndarray
            Power in signal_units^2, same shape as *cwt_result*.
        """
        return np.abs(cwt_result) ** 2

    def psd(self, cwt_result: np.ndarray) -> np.ndarray:
        """Power spectral density from CWT coefficients.

        Normalises the power spectrum by the equivalent noise bandwidth
        (ENBW) of the wavelet at each frequency so the result has
        units of ``signal_units^2 / Hz``.

        Parameters
        ----------
        cwt_result : np.ndarray
            CWT output, shape ``(n_freqs, n_times)``.

        Returns
        -------
        np.ndarray
            PSD in signal_units^2 / Hz, same shape as *cwt_result*.
        """
        return self.power(cwt_result) / self.enbw()[:, np.newaxis]

    def asd(self, cwt_result: np.ndarray) -> np.ndarray:
        """Amplitude spectral density from CWT coefficients.

        Square root of the PSD, equivalent to normalising the
        amplitude by ``sqrt(ENBW)`` at each frequency.  Units are
        ``signal_units / sqrt(Hz)``.

        Parameters
        ----------
        cwt_result : np.ndarray
            CWT output, shape ``(n_freqs, n_times)``.

        Returns
        -------
        np.ndarray
            ASD in signal_units / sqrt(Hz), same shape as *cwt_result*.
        """
        return self.amplitude(cwt_result) / np.sqrt(self.enbw())[:, np.newaxis]
