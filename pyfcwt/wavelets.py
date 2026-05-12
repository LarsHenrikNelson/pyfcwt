import numpy as np
from numba import njit

import pyfftw

from .wavelet_utils import f_to_s


class Wavelet:
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
        return f_to_s(fc, self.fs, self.n_cycles)

    def time(self, fc: float):
        inv_fs = 1.0 / self.fs

        # I think this fraction bandwidth in the freq domain
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
        inv_fs = 1.0 / self.fs

        # I think this fraction bandwidth in the freq domain
        if self.sigma == -1:
            sigma_t = self.n_cycles / (2.0 * np.pi * fc)
        else:
            sigma_t = self.n_cycles / (2.0 * np.pi * self.sigma)

        # Go gauss_sd STDEVs out on each side
        num_values = int((self.gauss_sd * sigma_t) // inv_fs)
        return num_values * 2 + 1