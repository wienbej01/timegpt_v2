"""Tests for forecast cache functionality."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from timegpt_v2.utils.cache import CacheKey, ForecastCache


@pytest.fixture
def temp_cache_dir() -> Path:
    """Create a temporary directory for cache testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_cache_key_hash_stable() -> None:
    """Test that cache key hashing is stable."""
    key1 = CacheKey(
        symbol="AAPL",
        trade_date="2024-01-01",
        snapshot="10:00",
        horizon=10,
        quantiles=(0.25, 0.5, 0.75),
        model="timegpt-1",
    )
    key2 = CacheKey(
        symbol="AAPL",
        trade_date="2024-01-01",
        snapshot="10:00",
        horizon=10,
        quantiles=(0.25, 0.5, 0.75),
        model="timegpt-1",
    )
    assert key1 == key2
    assert hash(key1) == hash(key2)


def test_cache_write_read_roundtrip(temp_cache_dir: Path) -> None:
    """Test cache write and read operations."""
    cache = ForecastCache(root=temp_cache_dir)
    key = CacheKey(
        symbol="TEST",
        trade_date="2024-01-01",
        snapshot="10:00",
        horizon=5,
        quantiles=(0.5,),
    )
    value = {"forecast": [1.0, 2.0, 3.0]}

    # Write
    cache.put(key, value)

    # Read
    retrieved = cache.get(key)
    assert retrieved == value


def test_cache_miss_returns_none(temp_cache_dir: Path) -> None:
    """Test that missing keys return None."""
    cache = ForecastCache(root=temp_cache_dir)
    key = CacheKey(
        symbol="MISSING",
        trade_date="2024-01-01",
        snapshot="10:00",
        horizon=5,
        quantiles=(0.5,),
    )
    assert cache.get(key) is None


def test_cache_memory_only() -> None:
    """Test cache without disk persistence."""
    cache = ForecastCache(root=None)
    key = CacheKey(
        symbol="MEMORY",
        trade_date="2024-01-01",
        snapshot="10:00",
        horizon=5,
        quantiles=(0.5,),
    )
    value = {"data": "in_memory"}

    cache.put(key, value)
    retrieved = cache.get(key)
    assert retrieved == value


def test_cache_file_persistence(temp_cache_dir: Path) -> None:
    """Test that cache persists to files."""
    cache = ForecastCache(root=temp_cache_dir)
    key = CacheKey(
        symbol="PERSIST",
        trade_date="2024-01-01",
        snapshot="10:00",
        horizon=5,
        quantiles=(0.5,),
    )
    value = {"persistent": True}

    cache.put(key, value)

    # Check file exists
    files = list(temp_cache_dir.glob("*.json"))
    assert len(files) == 1

    # Verify file content
    with files[0].open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert data == value


def test_cache_corrupt_file_handling(temp_cache_dir: Path) -> None:
    """Test handling of corrupt cache files."""
    cache = ForecastCache(root=temp_cache_dir)
    key = CacheKey(
        symbol="CORRUPT",
        trade_date="2024-01-01",
        snapshot="10:00",
        horizon=5,
        quantiles=(0.5,),
    )

    # Create corrupt file manually
    path = cache._path_for(key)
    path.write_text("invalid json", encoding="utf-8")

    # Should return None without crashing
    retrieved = cache.get(key)
    assert retrieved is None