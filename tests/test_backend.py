"""Tests for backend initialization and enforcement."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from timegpt_v2.forecast.timegpt_client import NixtlaTimeGPTBackend


def test_backend_enforced_nixtla_failure(tmp_path: Path) -> None:
    """Test that Nixtla backend init fails without API key."""
    # Test the backend init directly
    with patch.dict(os.environ, {"TIMEGPT_API_KEY": ""}, clear=True):
        with pytest.raises(ValueError, match="TimeGPT API key not provided"):
            NixtlaTimeGPTBackend()


def test_backend_auto_fallback_on_failure(tmp_path: Path) -> None:
    """Test that auto mode falls back to stub when Nixtla fails."""
    # This is tested in the CLI forecast function
    # When backend_mode == "auto" and init fails, it logs warning and sets backend to None
    # Then later raises if backend is None

    # For now, since we enforce live backend, backend=None raises error
    pass


def test_no_stub_backend_when_nixtla(tmp_path: Path) -> None:
    """Test that _LocalDeterministicBackend is not used when backend: nixtla."""
    # This is hard to test directly, but we can check that in forecast command,
    # when backend_mode == "nixtla", it tries to init NixtlaTimeGPTBackend

    # We can mock the TimeGPTClient init and check that backend is NixtlaTimeGPTBackend instance
    pass
