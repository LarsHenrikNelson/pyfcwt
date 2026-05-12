# pyFCWT

[![Test](https://github.com/larshnelson/pyfcwt/actions/workflows/test.yml/badge.svg)](https://github.com/larshnelson/pyfcwt/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

pyFCWT is a **pure-Python** port of [fCWT](https://github.com/fastlib/fCWT) that relies on [pyFFTW](https://github.com/pyFFTW/pyFFTW), [Numba](https://numba.pydata.org/) and [NumPy](https://numpy.org/). It is faster than other Python CWT packages while remaining limited to Morlet wavelets. Performance is close to the original C++ fCWT with the added benefit of easy installation on any platform — no compiler required.

## Features

- **Fast** — frequency-domain convolution via FFTW with Numba-parallelised daughter-wavelet multiplication.
- **Pure Python** — no C/C++ extensions to compile; installs with `pip` on Linux, macOS and Windows.
- **FFTW wisdom** — save and reload FFTW plans to skip the planning overhead on subsequent runs.
- **Flexible frequency grids** — linear or logarithmic spacing between any two frequencies.
- **Spectral utilities** — built-in methods for amplitude, power, PSD, and ASD directly from CWT coefficients.
- **Single & double precision** — choose `complex64` or `complex128` to trade accuracy for speed/memory.

## Installation

```bash
pip install pyfcwt
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add pyfcwt
```

### conda-forge

```bash
conda install -c conda-forge pyfcwt
```

## Quick Start

```python
import numpy as np
from pyfcwt import PyFCWT, Wavelet, Frequencies

# Parameters
fs = 1000.0            # Sampling rate (Hz)
t = np.arange(0, 1.0, 1 / fs)
signal = np.sin(2 * np.pi * 20 * t) + np.sin(2 * np.pi * 80 * t)

# Configure the transform
wavelet = Wavelet(fs=fs, n_cycles=7.0)
freqs = Frequencies(wavelet, f0=1.0, f1=100.0, fn=200, fs=fs, scaling="log")
fcwt = PyFCWT(wavelet, freqs, threads=-1)

# Run the CWT
result = fcwt.cwt(signal)  # shape: (200, 1000), complex128

# Derive spectral quantities
amplitude = fcwt.amplitude(result)
power = fcwt.power(result)
psd = fcwt.psd(result)
asd = fcwt.asd(result)
```

## API Overview

### `Wavelet`

Defines the Morlet wavelet. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fs` | — | Sampling rate in Hz |
| `n_cycles` | `7.0` | Number of cycles (time–frequency trade-off) |
| `gauss_sd` | `5.0` | Gaussian envelope truncation (std devs each side) |
| `sigma` | `-1` | Fixed bandwidth (`-1` = frequency-adaptive) |
| `zero_mean` | `True` | Enforce zero-mean admissibility condition |
| `imaginary` | `True` | Complex (analytic) wavelet |

### `Frequencies`

Generates the analysis frequency/scale grid.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `wavelet` | — | `Wavelet` instance |
| `f0` | — | Lowest frequency (Hz) |
| `f1` | — | Highest frequency (Hz) |
| `fn` | — | Number of frequency bins |
| `fs` | — | Sampling rate (Hz) |
| `scaling` | `"log"` | `"log"` or `"linear"` spacing |

### `PyFCWT`

The CWT engine.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `wavelet` | — | `Wavelet` instance |
| `frequencies` | — | `Frequencies` instance |
| `threads` | `-1` | FFTW threads (`-1` = half of CPU cores) |
| `norm` | `True` | Normalise output by FFT length |
| `dtype` | `"complex128"` | `"complex128"` or `"complex64"` |

**Methods:**

| Method | Returns |
|--------|---------|
| `cwt(signal)` | Complex CWT coefficients `(n_freqs, n_samples)` |
| `amplitude(cwt)` | `\|W(f,t)\|` |
| `power(cwt)` | `\|W(f,t)\|²` |
| `psd(cwt)` | Power spectral density (signal_units² / Hz) |
| `asd(cwt)` | Amplitude spectral density (signal_units / √Hz) |
| `enbw()` | Equivalent noise bandwidth at each frequency |

### FFTW Wisdom Utilities

FFTW plans can be expensive to create. Save and reload them to avoid repeated planning:

```python
from pyfcwt import save_wisdom, load_wisdom

# After running a transform, persist the plans:
save_wisdom()          # writes to ~/.pyfcwt/

# On the next run, restore them before creating PyFCWT:
load_wisdom()
```

## Speed Comparison with fCWT

Benchmarks comparing pyFCWT against the original C++ [fCWT](https://github.com/fastlib/fCWT) on the same hardware. Columns are formatted as `signal_length-num_frequencies`.

**Setup:** Morlet wavelet, log-spaced frequencies 1–100 Hz, `threads=-1`.

| Implementation    | 10k-300 | 10k-3000 | 100k-300 | 100k-3000 |
|-------------------|--------:|---------:|---------:|----------:|
| fCWT (C++)        | …s      | …s       | …s       | …s        |
| pyFCWT            | …s      | …s       | …s       | …s        |
| PyWavelets        | …s      | …s       | …s       | …s        |
| pyFCWT / fCWT     | …×      | …×       | …×       | …×        |
| PyWavelets / fCWT | …×      | …×       | …×       | …×        |

> **Note:** Fill in the values above with your own measurements.
> Run `python benchmarks/bench_pyfcwt_vs_fcwt.py` to generate the table.
> Times will vary with CPU, thread count, and whether FFTW wisdom is
> pre-loaded. The first pyFCWT call includes Numba JIT compilation
> overhead; subsequent calls are significantly faster.

## Wavelet Utility Functions

The `pyfcwt.wavelet_utils` module provides standalone helpers:

| Function | Description |
|----------|-------------|
| `f_to_s(fc, fs, n_cycles)` | Centre frequency → wavelet scale |
| `s_to_f(s, fs, n_cycles)` | Wavelet scale → centre frequency |
| `fwhm_freq(freq, n_cycles)` | FWHM of the Gaussian envelope (seconds) |
| `fwhm_to_cycles(fwhm, freqs)` | FWHM → equivalent number of cycles |
| `get_wavelet_length(fc, fs, ...)` | Sample length of a Morlet wavelet |
| `gen_freqs(f0, f1, n, scaling)` | Generate a frequency vector |

## Requirements

- Python ≥ 3.10
- NumPy ≥ 2.2.6
- Numba ≥ 0.65.1
- pyFFTW ≥ 0.15.0

## Development

```bash
git clone https://github.com/larshnelson/pyfcwt.git
cd pyfcwt
uv sync --group dev
uv run pytest
```

## License

[MIT](LICENSE) — Lars Henrik Nelson