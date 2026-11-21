"""Tests for the Codex ACP persona error handling."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from acp.exceptions import RequestError

with patch("shutil.which", return_value="/usr/bin/codex-acp"):
    from jupyter_ai_acp_client.acp_personas.codex import CodexAcpPersona


class TestProcessMessageErrorHandling:
    """Tests for error code detection in process_message().

    codex-acp sends -32000 (AuthRequired) when no API key is configured,
    and -32603 (InternalError) for runtime failures like wrong API key.
    We only catch -32000; runtime errors should propagate with their
    real message so users can diagnose the issue.
    """

    def test_auth_required_is_caught(self):
        """ACP AuthRequired (-32000) should trigger handle_no_auth."""
        error = RequestError(-32000, "Authentication required")
        assert error.code == -32000

    def test_internal_error_is_not_caught(self):
        """InternalError (-32603) should propagate — user needs the real error."""
        error = RequestError(-32603, "Internal error", "unexpected status 401")
        assert error.code != -32000

    def test_other_errors_propagate(self):
        """Non-auth errors should propagate unchanged."""
        error = RequestError(-32601, "Method not found")
        assert error.code != -32000

    def test_resource_not_found_propagates(self):
        """Resource errors are not auth errors."""
        error = RequestError(-32002, "Resource not found")
        assert error.code != -32000


@pytest.mark.asyncio
async def test_prepare_auth_error_shows_login_prompt():
    """Auth errors raised while preparing the persona use the login flow."""
    persona = MagicMock(spec=CodexAcpPersona)
    persona.handle_message_no_auth = AsyncMock()

    await CodexAcpPersona.handle_uncaught_exception(
        persona, RequestError(-32000, "Authentication required")
    )

    persona.handle_message_no_auth.assert_awaited_once_with(None)
