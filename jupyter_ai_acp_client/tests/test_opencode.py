"""Tests for the OpenCode ACP persona helper functions."""

import pytest

from jupyter_ai_acp_client.acp_personas.opencode import _is_auth_error


class TestIsAuthError:
    """Tests for _is_auth_error() keyword matching."""

    @pytest.mark.parametrize(
        "message",
        [
            "API key not found",
            "Invalid api key provided",
            "OPENAI_API_KEY is not set",
            "Authentication required",
            "authentication failed for provider",
            "Unauthorized access",
            "Request unauthorized",
            "Invalid credential",
            "credentials expired",
            "Provider not configured",
            "Model is not configured",
            "403 Forbidden",
            "request forbidden by policy",
            "missing api_key parameter",
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
