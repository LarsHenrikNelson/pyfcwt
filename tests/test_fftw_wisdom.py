"""Tests for pyfcwt.fftw_wisdom – saving, loading, exporting FFTW wisdom."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from pyfcwt.fftw_wisdom import (
    FFTDATA,
    get_wisdom,
    import_wisdom,
    save_wisdom,
    load_wisdom,
    create_wisdom,
)


class TestFFTDATA:
    def test_named_tuple_fields(self):
        d = FFTDATA(length=1024, dtype="float64")
        assert d.length == 1024
        assert d.dtype == "float64"

    def test_unpacking(self):
        length, dtype = FFTDATA(512, "complex128")
        assert length == 512
        assert dtype == "complex128"


class TestGetWisdom:
    def test_returns_tuple(self):
        wisdom = get_wisdom()
        assert isinstance(wisdom, tuple)


class TestImportWisdom:
    def test_accepts_tuple(self):
        """import_wisdom should not raise with a valid wisdom tuple."""
        current = get_wisdom()
        import_wisdom(current)  # re-import what we already have


class TestCreateWisdom:
    def test_real_to_complex(self):
        """Should plan an r2c transform without error."""
        create_wisdom(
            FFTDATA(length=1024, dtype="float64"),
            FFTDATA(length=1024, dtype="complex128"),
        )

    def test_complex_to_complex(self):
        """Should plan a c2c transform without error."""
        create_wisdom(
            FFTDATA(length=1024, dtype="complex128"),
            FFTDATA(length=1024, dtype="complex128"),
        )


class TestSaveLoadWisdom:
    def test_save_creates_files(self, tmp_path):
        """save_wisdom should write wisdom files into the cache dir."""
        with patch("pyfcwt.fftw_wisdom.config.get_cache_dir", return_value=tmp_path):
            save_wisdom()

        files = list(tmp_path.glob("*_wisdom"))
        assert len(files) > 0

    def test_load_round_trip(self, tmp_path):
        """Saving then loading should not raise."""
        with patch("pyfcwt.fftw_wisdom.config.get_cache_dir", return_value=tmp_path):
            save_wisdom()
            load_wisdom()  # should not raise

    def test_load_empty_dir(self, tmp_path):
        """load_wisdom on a directory with no wisdom files should be a no-op."""
        with patch("pyfcwt.fftw_wisdom.config.get_cache_dir", return_value=tmp_path):
            load_wisdom()  # should not raise
