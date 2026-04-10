"""Tests for the Gemini ACP persona helper behavior."""

from unittest.mock import MagicMock, patch

from acp.exceptions import RequestError


_mock_run = MagicMock()
_mock_run.returncode = 0
_mock_run.stdout = "gemini 0.34.0"
_mock_run.stderr = ""

with patch("shutil.which", return_value="/usr/bin/gemini"), patch(
    "subprocess.run", return_value=_mock_run
):
    from jupyter_ai_acp_client.acp_personas.gemini import GeminiAcpPersona


class TestGeminiLoadSessionRecovery:
    """Tests for Gemini-specific load_session recovery semantics."""

    def test_internal_error_is_recoverable_for_load_session(self):
        """Gemini may return InternalError when previous sessions cannot load."""
        persona = object.__new__(GeminiAcpPersona)
        error = RequestError(-32603, "Internal error")

        assert persona._is_recoverable_load_session_error(error) is True

    def test_resource_not_found_remains_recoverable(self):
        """Gemini still inherits the standard ACP ResourceNotFound recovery."""
        persona = object.__new__(GeminiAcpPersona)
        error = RequestError(-32002, "Resource not found")

        assert persona._is_recoverable_load_session_error(error) is True

    def test_other_errors_are_not_recoverable(self):
        """Gemini does not treat arbitrary load_session errors as recoverable."""
        persona = object.__new__(GeminiAcpPersona)
        error = RequestError(-32601, "Method not found")

        assert persona._is_recoverable_load_session_error(error) is False
