"""Tests for environment variable loading and API key handling."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from timegpt_v2.cli import app


@pytest.mark.skip(reason="Environment loading happens at import time, hard to test in isolation")
def test_env_loading(tmp_path: Path) -> None:
    """Test that .env file is loaded and API keys are available."""
    env_file = tmp_path / ".env"
    env_file.write_text("TIMEGPT_API_KEY=test_key_123\nNIXTLA_API_KEY=alt_key_456\n")

    # Change to tmp_path to simulate project root
    original_cwd = os.getcwd()
    original_env = os.environ.copy()
    try:
        os.chdir(tmp_path)

        # Import after changing directory to trigger .env loading
        # The .env loading happens at module import time
        # Clear environment completely for isolation
        os.environ.clear()
        os.environ.update({"PATH": original_env.get("PATH", "")})  # Keep PATH for imports

        # Re-import to trigger loading
        import importlib
        import timegpt_v2.cli
        importlib.reload(timegpt_v2.cli)

        # Check that keys are loaded
        assert os.environ.get("TIMEGPT_API_KEY") == "test_key_123"
        assert os.environ.get("NIXTLA_API_KEY") == "alt_key_456"

    finally:
        os.chdir(original_cwd)


def test_env_fallback_parsing(tmp_path: Path) -> None:
    """Test fallback .env parser when python-dotenv fails."""
    env_file = tmp_path / ".env"
    env_file.write_text("TIMEGPT_API_KEY=fallback_key\n# comment\nINVALID_LINE\n")

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        with patch("dotenv.load_dotenv", side_effect=Exception("dotenv failed")):
            with patch.dict(os.environ, {}, clear=True):
                import importlib
                import timegpt_v2.cli
                importlib.reload(timegpt_v2.cli)

                assert os.environ.get("TIMEGPT_API_KEY") == "fallback_key"

    finally:
        os.chdir(original_cwd)