"""FFTW wisdom persistence utilities.

FFTW *wisdom* captures the optimal FFT plan for a given transform size
and hardware configuration.  The helpers in this module allow wisdom to
be saved to disk and reloaded on subsequent runs, avoiding the overhead
of replanning.
"""

from typing import Literal, NamedTuple

import pyfftw

from . import config

__all__ = [
    "FFTDATA",
    "get_wisdom",
    "import_wisdom",
    "load_wisdom",
    "save_wisdom",
]


class FFTDATA(NamedTuple):
    """Lightweight descriptor for an FFT buffer.

    Attributes
    ----------
    length : int
        Number of elements in the buffer.
    dtype : {"complex128", "float64"}
        NumPy dtype string for the buffer.
    """

    length: int
    dtype: Literal["complex128", "float64"]


def import_wisdom(wisdom) -> None:
    """Import FFTW wisdom into the current session.

    Parameters
    ----------
    wisdom : tuple
        A wisdom tuple as returned by :func:`get_wisdom` or
        ``pyfftw.export_wisdom()``.
    """
    pyfftw.import_wisdom(wisdom)


def create_wisdom(input: FFTDATA, output: FFTDATA) -> None:
    """Create FFTW wisdom by planning a transform.

    Plans either a real-to-complex or complex-to-complex forward
    transform depending on the dtypes of *input* and *output*.  The
    resulting wisdom is stored internally by FFTW and can be exported
    with :func:`get_wisdom`.

    Parameters
    ----------
    input : FFTDATA
        Descriptor for the input buffer.
    output : FFTDATA
        Descriptor for the output buffer.
    """
    if input.dtype == "float64" and output.dtype == "complex128":
        a = pyfftw.empty_aligned(input.length, dtype="float64")
        b = pyfftw.empty_aligned(output.length // 2 + 1, dtype="complex128")

        _ = pyfftw.FFTW(a, b)
    elif input.dtype == "complex128" and output.dtype == "complex128":
        a = pyfftw.empty_aligned(input.length, dtype="complex128")
        b = pyfftw.empty_aligned(output.length, dtype="complex128")

        _ = pyfftw.FFTW(a, b)


def get_wisdom() -> tuple:
    """Export the current FFTW wisdom.

    Returns
    -------
    tuple
        A tuple of bytes objects representing the accumulated wisdom
        for each FFTW precision (single, double, long double).
    """
    return pyfftw.export_wisdom()


def save_wisdom() -> None:
    """Save the current FFTW wisdom to disk.

    Wisdom files are written to the pyfcwt cache directory
    (``~/.pyfcwt/``) as ``0_wisdom``, ``1_wisdom``, etc.  They can
    be restored in a later session with :func:`load_wisdom`.
    """
    wisdom = pyfftw.export_wisdom()
    save_path = config.get_cache_dir()
    for index, i in enumerate(wisdom):
        with open(save_path / f"{index}_wisdom", "wb") as f:
            f.write(i)


def load_wisdom() -> None:
    """Load FFTW wisdom from disk into the current session.

    Reads all ``*_wisdom`` files from the pyfcwt cache directory
    (``~/.pyfcwt/``) and imports them via ``pyfftw.import_wisdom``.
    If no wisdom files are found this is a silent no-op.
    """
    temp_path = list(config.get_cache_dir().glob("*_wisdom"))
    if len(temp_path) > 0:
        data = []
        for i in temp_path:
            with open(i, "rb") as rf:
                data.append(rf.read())
        pyfftw.import_wisdom(tuple(data))
