"""Tests for the Codex ACP persona helper functions."""

from unittest.mock import patch

import pytest

# codex.py has a module-level guard that raises PersonaRequirementsUnmet
# when codex-acp is not installed. Mock the guard so we can import
# _is_auth_error in CI without the binary.
with patch("shutil.which", return_value="/usr/bin/codex-acp"):
    from jupyter_ai_acp_client.acp_personas.codex import _is_auth_error


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
        ],
    )
    def test_detects_auth_errors(self, message):
        assert _is_auth_error(Exception(message)) is True

    @pytest.mark.parametrize(
        "message",
        [
            "Connection timed out",
            "Rate limit exceeded",
            "Internal server error",
            "Model not found",
        ],
    )
    def test_ignores_non_auth_errors(self, message):
        assert _is_auth_error(Exception(message)) is False

    def test_case_insensitive(self):
        assert _is_auth_error(Exception("API KEY MISSING")) is True

    def test_empty_message(self):
        assert _is_auth_error(Exception("")) is False
