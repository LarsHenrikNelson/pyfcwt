"""Tests for pyfcwt.config – cache directory management."""

from pathlib import Path

from pyfcwt.config import get_cache_dir, APP_NAME


class TestConfig:
    def test_app_name(self):
        assert APP_NAME == "pyfcwt"

    def test_cache_dir_exists(self):
        cache_dir = get_cache_dir()
        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_cache_dir_is_under_home(self):
        cache_dir = get_cache_dir()
        assert cache_dir.parent == Path.home()
        assert cache_dir.name == f".{APP_NAME}"

    def test_cache_dir_is_cached(self):
        """Repeated calls should return the same object (lru_cache)."""
        assert get_cache_dir() is get_cache_dir()
