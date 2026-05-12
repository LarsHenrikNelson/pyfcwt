from ty_extensions import Unknown
from typing import Literal

import numpy as np

from .wavelet_utils import f_to_s
from .wavelets import Wavelet


class Frequencies:
    """Analysis frequency and scale grid for the CWT.

    Generates a set of *fn* center frequencies between *f0* and *f1*
    using either linear or logarithmic spacing, together with the
    corresponding wavelet scales.  Frequencies are stored in
    **descending** order (highest first) so that scales are ascending.

    Parameters
    ----------
    wavelet : Wavelet
        Wavelet instance whose ``n_cycles`` parameter is used for the
        frequency-to-scale conversion.
    f0 : float
        Lowest analysis frequency in Hz.
    f1 : float
        Highest analysis frequency in Hz.
    fn : float
        Number of frequency bins.
    fs : float
        Sampling rate of the signal in Hz.
    scaling : {"linear", "log"}, optional
        Spacing mode for the frequency grid.  Default is ``"log"``.

    Attributes
    ----------
    f : np.ndarray
        1-D array of center frequencies in Hz (descending order).
    s : np.ndarray
        1-D array of wavelet scales corresponding to ``f``
        (ascending order).
    """

    def __init__(
        self,
        wavelet: Wavelet,
        f0: float,
        f1: float,
        fn: float,
        fs: float,
        scaling: Literal["linear", "log"] = "log",
    ):
        self.wavelet = wavelet
        self.f0 = f0
        self.f1 = f1
        self.fn = fn
        self.fs = fs
        self.scaling = scaling

        if scaling == "linear":
            self.f: np.ndarray = np.linspace(start=f0, stop=f1, num=fn)[::-1]
        else:
            self.f: np.ndarray = np.logspace(
                start=np.log10(f0), stop=np.log10(f1), num=fn
            )[::-1]

        self.s: np.ndarray = f_to_s(self.f, self.fs, self.wavelet.n_cycles)
