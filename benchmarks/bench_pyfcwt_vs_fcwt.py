"""Benchmark: pyFCWT vs fCWT (C++) vs PyWavelets.

Replicates the benchmark format from the fCWT paper, testing
combinations of signal lengths and frequency-bin counts. Prints a
Markdown table that can be pasted directly into the README.

The default test matrix matches the fCWT paper:
    signal lengths : 10 000, 100 000
    frequency bins : 300, 3 000

Usage
-----
    # All available libraries
    python benchmarks/bench_pyfcwt_vs_fcwt.py

    # pyFCWT only
    python benchmarks/bench_pyfcwt_vs_fcwt.py --pyfcwt-only

    # Customise iterations / warm-up rounds
    python benchmarks/bench_pyfcwt_vs_fcwt.py --warmup 5 --iterations 20

Requirements
------------
    pip install pyfcwt
    pip install fcwt matplotlib   # optional – fCWT comparison
    pip install PyWavelets         # optional – PyWavelets comparison
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Configuration – matches the fCWT paper benchmark matrix
# ---------------------------------------------------------------------------

SIGNAL_LENGTHS = [10_000, 100_000]
FREQ_BINS = [300, 3_000]
FS = 1000.0
F0 = 1.0
F1 = 100.0
DEFAULT_WARMUP = 3
DEFAULT_ITERATIONS = 10
DEFAULT_THREADS = os.cpu_count() // 2  # physical cores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    """Holds timing results for one (signal_length, fn) combination."""

    length: int
    fn: int
    pyfcwt_ms: float | None = None
    fcwt_ms: float | None = None
    pywt_ms: float | None = None

    @property
    def label(self) -> str:
        length_k = f"{self.length // 1_000}k"
        return f"{length_k}-{self.fn}"

    def _ratio(self, a: float | None, b: float | None) -> str:
        if a is not None and b is not None and b > 0:
            return f"{a / b:.2f}×"
        return "—"

    @property
    def pyfcwt_fcwt_ratio(self) -> str:
        return self._ratio(self.pyfcwt_ms, self.fcwt_ms)

    @property
    def pywt_fcwt_ratio(self) -> str:
        return self._ratio(self.pywt_ms, self.fcwt_ms)

    @staticmethod
    def fmt_ms(ms: float | None) -> str:
        if ms is None:
            return "—"
        return f"{ms / 1_000:.3f}s"

    @property
    def pyfcwt_str(self) -> str:
        return self.fmt_ms(self.pyfcwt_ms)

    @property
    def fcwt_str(self) -> str:
        return self.fmt_ms(self.fcwt_ms)

    @property
    def pywt_str(self) -> str:
        return self.fmt_ms(self.pywt_ms)


def _make_signal(n: int) -> np.ndarray:
    """Deterministic chirp signal of length *n* sampled at *FS*."""
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * ((1 + (20 * t) / (n / FS)) * t)).astype(np.float64)


def _median_time(func, n_iter: int) -> float:
    """Return the median execution time (ms) of *func* over *n_iter* calls."""
    times: list[float] = []
    for _ in range(n_iter):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1_000)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Per-library benchmark runners
# ---------------------------------------------------------------------------


def bench_pyfcwt(
    signal: np.ndarray,
    fn: int,
    threads: int,
    warmup: int,
    iterations: int,
) -> float:
    """Return median CWT time in ms for pyFCWT."""
    from pyfcwt import Frequencies, PyFCWT, Wavelet

    wavelet = Wavelet(fs=FS, n_cycles=7.0)
    freqs = Frequencies(wavelet, f0=F0, f1=F1, fn=fn, fs=FS, scaling="log")
    engine = PyFCWT(wavelet, freqs, threads=threads)

    # Warm-up (includes Numba JIT + FFTW planning)
    for _ in range(warmup):
        engine.cwt(signal)

    return _median_time(lambda: engine.cwt(signal), iterations)


def bench_fcwt(
    signal: np.ndarray,
    fn: int,
    threads: int,
    warmup: int,
    iterations: int,
) -> float:
    """Return median CWT time in ms for the C++ fCWT Python bindings."""
    import fcwt

    sig32 = signal.astype(np.float32)

    # fCWT's class-based API allows explicit thread control.
    morl = fcwt.Morlet(2.0)
    scales = fcwt.Scales(morl, fcwt.FCWT_LOGSCALES, int(FS), F0, F1, fn)
    f = fcwt.FCWT(morl, threads, False, False)

    n = sig32.shape[0]
    output = np.zeros((fn, n), dtype=np.complex64)

    def _run():
        f.cwt(sig32, scales, output)

    # Warm-up
    for _ in range(warmup):
        _run()

    return _median_time(_run, iterations)


def bench_pywt(
    signal: np.ndarray,
    fn: int,
    warmup: int,
    iterations: int,
) -> float:
    """Return median CWT time in ms for PyWavelets (single-threaded)."""
    import pywt

    wavelet = "cmor1.5-1.0"
    freqs = np.logspace(np.log10(F0), np.log10(F1), fn)[::-1]
    central_freq = pywt.central_frequency(wavelet, precision=10)
    scales = central_freq * FS / freqs
    dt = 1.0 / FS

    # Warm-up
    for _ in range(warmup):
        pywt.cwt(signal, scales, wavelet, sampling_period=dt)

    return _median_time(
        lambda: pywt.cwt(signal, scales, wavelet, sampling_period=dt), iterations
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark pyFCWT vs fCWT (C++) and print a Markdown table.",
    )
    parser.add_argument(
        "--pyfcwt-only",
        action="store_true",
        help="Only run pyFCWT benchmarks.",
    )
    parser.add_argument(
        "--fcwt-only",
        action="store_true",
        help="Only run fCWT benchmarks.",
    )
    parser.add_argument(
        "--pywt-only",
        action="store_true",
        help="Only run PyWavelets benchmarks.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Number of warm-up iterations (default {DEFAULT_WARMUP}).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"Number of timed iterations; median is reported (default {DEFAULT_ITERATIONS}).",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=SIGNAL_LENGTHS,
        help=f"Signal lengths to benchmark (default: {SIGNAL_LENGTHS}).",
    )
    parser.add_argument(
        "--freq-bins",
        type=int,
        nargs="+",
        default=FREQ_BINS,
        help=f"Frequency bin counts to benchmark (default: {FREQ_BINS}).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=(
            f"Thread count for all libraries (default: {DEFAULT_THREADS}, "
            f"i.e. cpu_count()//2 = physical cores). "
            f"Controls pyFCWT FFTW threads, Numba parallel threads, "
            f"and fCWT OpenMP threads."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    only_mode = args.pyfcwt_only or args.fcwt_only or args.pywt_only
    run_pyfcwt = args.pyfcwt_only if only_mode else True
    run_fcwt = args.fcwt_only if only_mode else True
    run_pywt = args.pywt_only if only_mode else True

    # Check availability of each library
    if run_fcwt:
        try:
            import fcwt  # noqa: F401
        except (ImportError, OSError) as exc:
            print(
                f"WARNING: fcwt could not be imported – skipping fCWT benchmarks.\n"
                f"         ({exc})\n"
                f"         Install with:  pip install fcwt matplotlib\n"
            )
            run_fcwt = False

    if run_pyfcwt:
        try:
            import pyfcwt  # noqa: F401
        except ImportError as exc:
            print(
                f"WARNING: pyfcwt could not be imported – skipping pyFCWT benchmarks.\n"
                f"         ({exc})\n"
                f"         Install with:  pip install pyfcwt\n"
            )
            run_pyfcwt = False

    if run_pywt:
        try:
            import pywt  # noqa: F401
        except ImportError as exc:
            print(
                f"WARNING: PyWavelets could not be imported – skipping benchmarks.\n"
                f"         ({exc})\n"
                f"         Install with:  pip install PyWavelets\n"
            )
            run_pywt = False

    if not run_pyfcwt and not run_fcwt and not run_pywt:
        print("ERROR: No CWT libraries available. Nothing to benchmark.")
        return

    # Pin Numba thread count so prange uses the same number as FFTW/OpenMP
    threads = args.threads
    import numba
    numba.set_num_threads(threads)

    # Build the test matrix: every (length, fn) combination
    combos = [
        (length, fn)
        for length in args.lengths
        for fn in args.freq_bins
    ]

    print(
        f"Benchmark config: fs={FS}, f0={F0}, f1={F1}, "
        f"warmup={args.warmup}, iterations={args.iterations}"
    )
    print(f"Threads: {threads} (cpu_count={os.cpu_count()}, physical≈{os.cpu_count() // 2})")
    print(f"  pyFCWT  : FFTW={threads}, Numba={numba.get_num_threads()}")
    if run_fcwt:
        print(f"  fCWT    : OpenMP={threads}")
    if run_pywt:
        print(f"  PyWavelets : single-threaded (no thread control)")
    print(f"Test matrix: {len(combos)} combinations")
    print(f"  signal lengths : {args.lengths}")
    print(f"  frequency bins : {args.freq_bins}")
    print()

    results: list[BenchResult] = []

    for length, fn in combos:
        label = f"{length // 1_000}k-{fn}"
        print(f"[{label}] signal={length:,} samples, fn={fn} frequencies")
        signal = _make_signal(length)
        res = BenchResult(length=length, fn=fn)

        if run_fcwt:
            ms = bench_fcwt(signal, fn=fn, threads=threads, warmup=args.warmup, iterations=args.iterations)
            res.fcwt_ms = ms
            print(f"  fCWT (C++)  : {res.fcwt_str:>10}")

        if run_pyfcwt:
            ms = bench_pyfcwt(signal, fn=fn, threads=threads, warmup=args.warmup, iterations=args.iterations)
            res.pyfcwt_ms = ms
            print(f"  pyFCWT      : {res.pyfcwt_str:>10}")

        if run_pywt:
            ms = bench_pywt(signal, fn=fn, warmup=args.warmup, iterations=args.iterations)
            res.pywt_ms = ms
            print(f"  PyWavelets  : {res.pywt_str:>10}")

        results.append(res)
        print()

    # ------------------------------------------------------------------
    # Print Markdown table (column-per-combo, matching fCWT paper style)
    # ------------------------------------------------------------------
    labels = [r.label for r in results]
    header_cols = " | ".join(labels)
    separator_cols = " | ".join("---:" for _ in results)

    print("=" * 72)
    print("Markdown table (paste into README.md):")
    print("=" * 72)
    print()
    print(f"| Implementation | {header_cols} |")
    print(f"|----------------|{separator_cols}|")

    if run_fcwt:
        cols = " | ".join(r.fcwt_str for r in results)
        print(f"| fCWT (C++)     | {cols} |")

    if run_pyfcwt:
        cols = " | ".join(r.pyfcwt_str for r in results)
        print(f"| pyFCWT         | {cols} |")

    if run_pywt:
        cols = " | ".join(r.pywt_str for r in results)
        print(f"| PyWavelets     | {cols} |")

    # Ratio rows (only when fCWT is the baseline)
    if run_fcwt and run_pyfcwt:
        cols = " | ".join(r.pyfcwt_fcwt_ratio for r in results)
        print(f"| pyFCWT / fCWT  | {cols} |")

    if run_fcwt and run_pywt:
        cols = " | ".join(r.pywt_fcwt_ratio for r in results)
        print(f"| PyWavelets / fCWT | {cols} |")

    print()


if __name__ == "__main__":
    main()
