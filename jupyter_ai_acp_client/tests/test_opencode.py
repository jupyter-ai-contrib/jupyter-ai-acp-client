"""Tests for the OpenCode ACP persona helper functions."""

from unittest.mock import MagicMock, patch

import pytest

# opencode.py has a module-level guard that raises PersonaRequirementsUnmet
# when the opencode CLI is not installed. Mock the guard so we can import
# _is_auth_error in CI without the CLI.
_mock_run = MagicMock()
_mock_run.returncode = 0
_mock_run.stdout = "1.0.0"
_mock_run.stderr = ""

with patch("shutil.which", return_value="/usr/bin/opencode"), \
     patch("subprocess.run", return_value=_mock_run):
    from jupyter_ai_acp_client.acp_personas.opencode import _is_auth_error


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
