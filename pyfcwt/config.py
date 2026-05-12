from functools import lru_cache
from pathlib import Path

APP_NAME = "pyfcwt"


@lru_cache(maxsize=1)
def get_cache_dir() -> Path:
    """Return (and create if needed) the pyfcwt cache directory."""
    cache_dir = Path.home() / f".{APP_NAME}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir