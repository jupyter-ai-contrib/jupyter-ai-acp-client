"""Tests for the Claude ACP persona error handling."""

from unittest.mock import patch

from acp.exceptions import RequestError

# claude.py raises PersonaRequirementsUnmet at import time when the
# `claude-agent-acp` adapter is not installed. Mock the guard so the helper
# can be imported in CI without the adapter.
with patch("shutil.which", return_value="/usr/bin/claude-agent-acp"):
    from jupyter_ai_acp_client.acp_personas.claude import _is_auth_error


class TestIsAuthError:
    """claude-agent-acp ends a prompt with the ACP `authRequired` error
    (-32000) when the user is not logged in. Every other error must keep its
    real message, even when the message mentions tokens or authors."""

    def test_auth_required_is_auth_error(self):
        assert _is_auth_error(RequestError.auth_required()) is True

    def test_auth_required_with_reason_is_auth_error(self):
        error = RequestError(-32000, "Authentication required", {"reason": "logged out"})
        assert _is_auth_error(error) is True

    def test_max_output_tokens_is_not_auth_error(self):
        error = RequestError(
            -32603,
            "API Error: Claude's response exceeded the 32000 output token maximum. "
            "To configure this behavior, set the CLAUDE_CODE_MAX_OUTPUT_TOKENS "
            "environment variable.",
            {"errorKind": "max_output_tokens"},
        )
        assert _is_auth_error(error) is False

    def test_message_mentioning_auth_is_not_auth_error(self):
        error = RequestError(-32603, "Tool failed: could not read AUTHORS file")
        assert _is_auth_error(error) is False

    def test_other_request_errors_are_not_auth_errors(self):
        assert _is_auth_error(RequestError(-32601, "Method not found")) is False
        assert _is_auth_error(RequestError(-32002, "Resource not found")) is False

    def test_non_request_error_is_not_auth_error(self):
        assert _is_auth_error(RuntimeError("Authentication required")) is False
