"""Tests for the Goose ACP persona helper functions."""

from unittest.mock import MagicMock, patch

import pytest
from acp.exceptions import RequestError
from jupyter_ai_persona_manager import PersonaRequirementsUnmet

# goose.py has a module-level guard that raises PersonaRequirementsUnmet
# when the goose CLI is not installed. Mock the guard so we can import
# helpers in CI without the CLI.
_mock_run = MagicMock()
_mock_run.returncode = 0
_mock_run.stdout = "goose 1.28.0"
_mock_run.stderr = ""

with patch("shutil.which", return_value="/usr/bin/goose"), \
     patch("subprocess.run", return_value=_mock_run):
    from jupyter_ai_acp_client.acp_personas.goose import (
        GooseAcpPersona,
        _check_goose,
        _get_explicit_user_mode,
        _is_setup_error,
        _parse_goose_mode,
    )


class TestIsSetupError:
    """Tests for _is_setup_error() — source-verified against block/goose server.rs."""

    def test_provider_not_configured(self):
        """server.rs line 870: init_provider fails."""
        error = RequestError(-32603, "Internal error", "Failed to set provider: Configuration value not found: GOOSE_PROVIDER")
        assert _is_setup_error(error) is True

    def test_session_creation_failed(self):
        """server.rs line 855: session_manager.create_session fails."""
        error = RequestError(-32603, "Internal error", "Failed to create session: database error")
        assert _is_setup_error(error) is True

    def test_agent_creation_failed(self):
        """server.rs line 864: create_agent_for_session fails."""
        error = RequestError(-32603, "Internal error", "Failed to create agent: DeveloperClient error")
        assert _is_setup_error(error) is True

    def test_no_data_framework_error(self):
        """Bug fix: sacp framework errors have data=None."""
        error = RequestError(-32603, "Internal error")
        assert _is_setup_error(error) is True

    def test_prompt_reply_error_not_caught(self):
        """server.rs line 1104: agent.reply() failure is NOT a setup error."""
        error = RequestError(-32603, "Internal error", "Error getting agent reply: timeout")
        assert _is_setup_error(error) is False

    def test_stream_error_not_caught(self):
        """server.rs line 1136: stream error is NOT a setup error."""
        error = RequestError(-32603, "Internal error", "Error in agent response stream: broken pipe")
        assert _is_setup_error(error) is False

    def test_auth_required_forward_compat(self):
        """ACP standard -32000 — Goose doesn't send this today but might in future."""
        error = RequestError(-32000, "Authentication required")
        assert _is_setup_error(error) is True

    def test_other_codes_propagate(self):
        error = RequestError(-32601, "Method not found")
        assert _is_setup_error(error) is False

    def test_resource_not_found_propagates(self):
        """server.rs line 925: session not found during prompt."""
        error = RequestError(-32002, "Resource not found", "Session not found: abc123")
        assert _is_setup_error(error) is False

    def test_invalid_api_key(self):
        """Provider configured but API key invalid/expired."""
        error = RequestError(-32603, "Internal error", "Authentication error: Authentication failed. Status: 401 Unauthorized. Response: invalid x-api-key")
        assert _is_setup_error(error) is True

    def test_case_insensitive(self):
        """Defensive: .lower() handles unexpected casing from future Goose versions."""
        error = RequestError(-32603, "Internal error", "FAILED TO SET PROVIDER: ...")
        assert _is_setup_error(error) is True

    def test_plain_exception(self):
        assert _is_setup_error(Exception("provider error")) is False


def _mock_result(stdout="goose 1.28.0", returncode=0, stderr=""):
    m = MagicMock()
    m.stdout = stdout
    m.returncode = returncode
    m.stderr = stderr
    return m


class TestCheckGoose:
    """Tests for _check_goose() version guard."""

    def test_not_installed(self):
        with patch("shutil.which", return_value=None):
            with pytest.raises(PersonaRequirementsUnmet, match="requires the Goose CLI"):
                _check_goose()

    def test_valid_version(self):
        with patch("shutil.which", return_value="/usr/bin/goose"), \
             patch("subprocess.run", return_value=_mock_result("goose 1.28.0")):
            _check_goose()  # should not raise

    def test_old_version(self):
        with patch("shutil.which", return_value="/usr/bin/goose"), \
             patch("subprocess.run", return_value=_mock_result("goose 1.7.0")):
            with pytest.raises(PersonaRequirementsUnmet, match=">=1.8.0"):
                _check_goose()

    def test_exact_minimum_version(self):
        with patch("shutil.which", return_value="/usr/bin/goose"), \
             patch("subprocess.run", return_value=_mock_result("goose 1.8.0")):
            _check_goose()  # should not raise

    def test_major_version_2_rejected(self):
        with patch("shutil.which", return_value="/usr/bin/goose"), \
             patch("subprocess.run", return_value=_mock_result("goose 2.0.0")):
            with pytest.raises(PersonaRequirementsUnmet, match=">=1.8.0,<2"):
                _check_goose()


class TestGooseModeEnv:
    """Tests for Goose subprocess environment overrides."""

    def test_sets_approve_mode_when_unset_and_no_explicit_config_mode(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "jupyter_ai_acp_client.acp_personas.goose._get_explicit_user_mode",
                return_value=None,
            ):
                env = GooseAcpPersona._build_subprocess_env()
        assert env is not None
        assert env["GOOSE_MODE"] == "approve"

    def test_respects_existing_mode(self):
        with patch.dict("os.environ", {"GOOSE_MODE": "auto"}, clear=True):
            env = GooseAcpPersona._build_subprocess_env()
        assert env is None

    def test_respects_explicit_mode_in_user_config(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "jupyter_ai_acp_client.acp_personas.goose._get_explicit_user_mode",
                return_value="smart_approve",
            ):
                env = GooseAcpPersona._build_subprocess_env()
        assert env is None


class TestGooseConfigParsing:
    """Tests for extracting GOOSE_MODE from Goose config."""

    def test_parse_goose_mode_plain(self):
        assert _parse_goose_mode("GOOSE_MODE: approve\n") == "approve"

    def test_parse_goose_mode_quoted(self):
        assert _parse_goose_mode('GOOSE_MODE: "smart_approve"\n') == "smart_approve"

    def test_parse_goose_mode_with_comment(self):
        assert _parse_goose_mode("GOOSE_MODE: approve # require approval\n") == "approve"

    def test_parse_goose_mode_absent(self):
        assert _parse_goose_mode("GOOSE_PROVIDER: openai\n") is None

    def test_get_explicit_user_mode_reads_config(self):
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.read_text.return_value = "GOOSE_PROVIDER: openai\nGOOSE_MODE: approve\n"
        with patch(
            "jupyter_ai_acp_client.acp_personas.goose._get_user_config_path",
            return_value=mock_path,
        ):
            assert _get_explicit_user_mode() == "approve"

    def test_get_explicit_user_mode_ignores_provider_only_config(self):
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.read_text.return_value = "GOOSE_PROVIDER: openai\n"
        with patch(
            "jupyter_ai_acp_client.acp_personas.goose._get_user_config_path",
            return_value=mock_path,
        ):
            assert _get_explicit_user_mode() is None
