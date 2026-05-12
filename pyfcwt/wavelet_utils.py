"""Utility functions for wavelet scale/frequency conversions, FWHM
calculations, and support-length estimation."""

import numpy as np


def get_support(fb: float, scale: float) -> float:
    """Return the effective support length of a wavelet.

    Parameters
    ----------
    fb : float
        Bandwidth parameter of the wavelet.
    scale : float
        Wavelet scale.

    Returns
    -------
    float
        Support length (``fb * scale * 3``).
    """
    return fb * scale * 3.0


def fwhm_sigma(sigma: float) -> float:
    """Full-width at half-maximum of a Gaussian with standard deviation *sigma*.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian.

    Returns
    -------
    float
        FWHM value (``sigma * 2 * sqrt(2 * ln 2)``).
    """
    return sigma * 2 * np.sqrt(2 * np.log(2))


def fwhm_freq(freq: float, n_cycles: float = 7.0) -> float:
    """FWHM of the Morlet wavelet's Gaussian envelope at a given frequency.

    Parameters
    ----------
    freq : float
        Centre frequency in Hz.
    n_cycles : float, optional
        Number of wavelet cycles.  Default is ``7.0``.

    Returns
    -------
    float
        FWHM in seconds.
    """
    sigma = n_cycles / (2.0 * np.pi * freq)
    return sigma * 2 * np.sqrt(2 * np.log(2))


def fwhm_to_cycles(fwhm: float, freqs) -> np.ndarray:
    """Convert a time-domain FWHM back to the equivalent number of cycles.

    Parameters
    ----------
    fwhm : float
        Full-width at half-maximum in seconds.
    freqs : array_like
        Centre frequencies in Hz.

    Returns
    -------
    np.ndarray
        Number of cycles corresponding to *fwhm* at each frequency.
    """
    return fwhm * np.pi * np.array(freqs) / np.sqrt(2 * np.log(2))


def f_to_s(
    fc: float | np.ndarray, fs: float | np.ndarray, n_cycles: float = 7.0
) -> float | np.ndarray:
    """Convert centre frequency to wavelet scale.

    Parameters
    ----------
    fc : float or np.ndarray
        Centre frequency (or array of frequencies) in Hz.
    fs : float or np.ndarray
        Sampling rate in Hz.
    n_cycles : float, optional
        Number of wavelet cycles.  Default is ``7.0``.

    Returns
    -------
    float or np.ndarray
        Corresponding wavelet scale(s).
    """
    return n_cycles * fs / (2 * fc * np.pi)


def s_to_f(
    s: float | np.ndarray, fs: float | np.ndarray, n_cycles: float = 7.0
) -> float | np.ndarray:
    """Convert wavelet scale to centre frequency.

    Parameters
    ----------
    s : float or np.ndarray
        Wavelet scale (or array of scales).
    fs : float or np.ndarray
        Sampling rate in Hz.
    n_cycles : float, optional
        Number of wavelet cycles.  Default is ``7.0``.

    Returns
    -------
    float or np.ndarray
        Corresponding centre frequency (or frequencies) in Hz.
    """
    return n_cycles * fs / (2 * s * np.pi)


def gen_freqs(
    f0: float, f1: float, num_steps: int, scaling: str = "log"
) -> np.ndarray:
    """Generate a frequency vector.

    Parameters
    ----------
    f0 : float
        Start frequency in Hz.
    f1 : float
        Stop frequency in Hz.
    num_steps : int
        Number of frequency bins.
    scaling : {"linear", "log"}, optional
        Spacing mode.  Default is ``"log"``.

    Returns
    -------
    np.ndarray
        1-D array of *num_steps* frequencies from *f0* to *f1*.
    """
    if scaling == "linear":
        freqs = np.linspace(start=f0, stop=f1, num=num_steps)
    else:
        freqs = np.logspace(start=np.log10(f0), stop=np.log10(f1), num=num_steps)
    return freqs


def get_wavelet_length(
    fc: float, fs: float, n_cycles: float = 7.0, gauss_sd: float = 5.0
) -> int:
    """Compute the sample length of a Morlet wavelet.

    The wavelet extends *gauss_sd* standard deviations on each side of
    its centre, giving an odd total length.

    Parameters
    ----------
    fc : float
        Centre frequency in Hz.
    fs : float
        Sampling rate in Hz.
    n_cycles : float, optional
        Number of wavelet cycles.  Default is ``7.0``.
    gauss_sd : float, optional
        Number of Gaussian standard deviations to include on each
        side.  Default is ``5.0``.

    Returns
    -------
    int
        Number of samples (always odd).
    """
    inv_fs = 1.0 / fs
    sigma_t = n_cycles / (2.0 * np.pi * fc)
    num_values = int((gauss_sd * sigma_t) // inv_fs)
    return num_values * 2 + 1


def get_amp_scale(f0: float, f1: float):
    """Returns the correct amplitude scaling given a frequency
    f0 and a mother frequency f1. f0 < f1.

    To calculate you have to run on the fft of the wavelets:
    Run the regression on the log2 wavelet frequencies and the log2 normalized wavelet
    amplitudes (np.max(np.abs(wavelet_fft))).
    slope is the slope of stats.linregress(np.log(freqs), np.log(x3 / x3[-1]))

    Then run a regression on the log2 wavelet frequencies and the intercepts
    of the above equation with different mother wavelets.
    intercept is the output of stats.linregress(np.log(freqs), intercepts).

    Frequencies can log scale of linear scale.

    Args:
        f0 (float): frequency of interest
        f1 (float): mother frequency

    Returns:
        float: amplitude scaling factor
    """
    return np.exp(-0.5 * np.log(f0) + (0.5 * np.log(f1) + -8.077e-7))
