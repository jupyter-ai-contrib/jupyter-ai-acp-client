"""Tests for the OpenCode ACP persona helper functions."""

from unittest.mock import MagicMock, patch

import pytest
from jupyter_ai_persona_manager import PersonaRequirementsUnmet

# opencode.py has a module-level guard that raises PersonaRequirementsUnmet
# when the opencode CLI is not installed. Mock the guard so we can import
# helpers in CI without the CLI.
_mock_run = MagicMock()
_mock_run.returncode = 0
_mock_run.stdout = "1.0.0"
_mock_run.stderr = ""

with patch("shutil.which", return_value="/usr/bin/opencode"), \
     patch("subprocess.run", return_value=_mock_run):
    from jupyter_ai_acp_client.acp_personas.opencode import (
        _check_opencode,
        _has_user_config,
        _is_auth_error,
    )


class TestIsAuthError:
    """Tests for _is_auth_error() keyword matching."""

    @pytest.mark.parametrize(
        "message",
        [
            "API key not found",
            "OPENAI_API_KEY is not set",
            "Authentication required",
            "Unauthorized access",
            "Invalid credential",
            "Provider not configured",
            "403 Forbidden",
        ],
    )
    def test_detects_auth_errors(self, message):
        error = Exception(message)
        assert _is_auth_error(error) is True

    @pytest.mark.parametrize(
        "message",
        [
            "Connection timed out",
            "Rate limit exceeded",
            "Internal server error",
            "Model not found",
            "Invalid JSON in request body",
            "File not found: /tmp/test.py",
            "Context window exceeded",
        ],
    )
    def test_ignores_non_auth_errors(self, message):
        error = Exception(message)
        assert _is_auth_error(error) is False

    def test_case_insensitive(self):
        assert _is_auth_error(Exception("API KEY MISSING")) is True
        assert _is_auth_error(Exception("AUTHENTICATION FAILED")) is True

    def test_empty_message(self):
        assert _is_auth_error(Exception("")) is False


def _mock_result(stdout="1.0.0", returncode=0, stderr=""):
    m = MagicMock()
    m.stdout = stdout
    m.returncode = returncode
    m.stderr = stderr
    return m


class TestCheckOpencode:
    """Tests for _check_opencode() version guard."""

    def test_not_installed(self):
        with patch("shutil.which", return_value=None):
            with pytest.raises(PersonaRequirementsUnmet, match="requires `opencode`"):
                _check_opencode()

    def test_valid_version(self):
        with patch("shutil.which", return_value="/usr/bin/opencode"), \
             patch("subprocess.run", return_value=_mock_result("opencode v1.2.3")):
            _check_opencode()  # should not raise


class TestHasUserConfig:
    """Tests for _has_user_config() global config detection."""

    def test_no_config_files(self, tmp_path):
        with patch("jupyter_ai_acp_client.acp_personas.opencode.Path.home", return_value=tmp_path):
            assert _has_user_config() is False

    def test_json_config_exists(self, tmp_path):
        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        (config_dir / "opencode.json").write_text("{}")
        with patch("jupyter_ai_acp_client.acp_personas.opencode.Path.home", return_value=tmp_path):
            assert _has_user_config() is True

    def test_jsonc_config_exists(self, tmp_path):
        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        (config_dir / "opencode.jsonc").write_text("{}")
        with patch("jupyter_ai_acp_client.acp_personas.opencode.Path.home", return_value=tmp_path):
            assert _has_user_config() is True
