"""Tests for the Goose ACP persona helper functions."""

from unittest.mock import MagicMock, patch

import pytest
from acp.exceptions import RequestError
from jupyter_ai_persona_manager import PersonaRequirementsUnmet

# Mock the import-time Goose CLI guard so helpers can be imported in tests.
_mock_run = MagicMock()
_mock_run.returncode = 0
_mock_run.stdout = "goose 1.28.0"
_mock_run.stderr = ""

with patch("shutil.which", return_value="/usr/bin/goose"), \
     patch("subprocess.run", return_value=_mock_run):
    from jupyter_ai_acp_client.acp_personas.goose import (
        _check_goose,
        _get_config_mode,
        _is_setup_error,
        _parse_goose_mode,
        _resolve_mode_decision,
    )


class TestIsSetupError:
    def test_provider_not_configured(self):
        error = RequestError(-32603, "Internal error", "Failed to set provider: Configuration value not found: GOOSE_PROVIDER")
        assert _is_setup_error(error) is True

    def test_session_creation_failed(self):
        error = RequestError(-32603, "Internal error", "Failed to create session: database error")
        assert _is_setup_error(error) is True

    def test_agent_creation_failed(self):
        error = RequestError(-32603, "Internal error", "Failed to create agent: DeveloperClient error")
        assert _is_setup_error(error) is True

    def test_no_data_framework_error(self):
        error = RequestError(-32603, "Internal error")
        assert _is_setup_error(error) is True

    def test_prompt_reply_error_not_caught(self):
        error = RequestError(-32603, "Internal error", "Error getting agent reply: timeout")
        assert _is_setup_error(error) is False

    def test_stream_error_not_caught(self):
        error = RequestError(-32603, "Internal error", "Error in agent response stream: broken pipe")
        assert _is_setup_error(error) is False

    def test_auth_required_forward_compat(self):
        error = RequestError(-32000, "Authentication required")
        assert _is_setup_error(error) is True

    def test_other_codes_propagate(self):
        error = RequestError(-32601, "Method not found")
        assert _is_setup_error(error) is False

    def test_resource_not_found_propagates(self):
        error = RequestError(-32002, "Resource not found", "Session not found: abc123")
        assert _is_setup_error(error) is False

    def test_invalid_api_key(self):
        error = RequestError(-32603, "Internal error", "Authentication error: Authentication failed. Status: 401 Unauthorized. Response: invalid x-api-key")
        assert _is_setup_error(error) is True

    def test_case_insensitive(self):
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
    def test_not_installed(self):
        with patch("shutil.which", return_value=None):
            with pytest.raises(PersonaRequirementsUnmet, match="requires the Goose CLI"):
                _check_goose()

    def test_valid_version(self):
        with patch("shutil.which", return_value="/usr/bin/goose"), \
             patch("subprocess.run", return_value=_mock_result("goose 1.28.0")):
            _check_goose()

    def test_old_version(self):
        with patch("shutil.which", return_value="/usr/bin/goose"), \
             patch("subprocess.run", return_value=_mock_result("goose 1.7.0")):
            with pytest.raises(PersonaRequirementsUnmet, match=">=1.8.0"):
                _check_goose()

    def test_exact_minimum_version(self):
        with patch("shutil.which", return_value="/usr/bin/goose"), \
             patch("subprocess.run", return_value=_mock_result("goose 1.8.0")):
            _check_goose()

    def test_major_version_2_rejected(self):
        with patch("shutil.which", return_value="/usr/bin/goose"), \
             patch("subprocess.run", return_value=_mock_result("goose 2.0.0")):
            with pytest.raises(PersonaRequirementsUnmet, match=">=1.8.0,<2"):
                _check_goose()


class TestGooseConfigParsing:
    """Tests for reading GOOSE_MODE from Goose config."""

    def test_parse_goose_mode_plain(self):
        assert _parse_goose_mode("GOOSE_MODE: approve\n") == "approve"

    def test_parse_goose_mode_quoted(self):
        assert _parse_goose_mode('GOOSE_MODE: "smart_approve"\n') == "smart_approve"

    def test_parse_goose_mode_with_comment(self):
        assert _parse_goose_mode("GOOSE_MODE: approve # require approval\n") == "approve"

    def test_parse_goose_mode_absent(self):
        assert _parse_goose_mode("GOOSE_PROVIDER: openai\n") is None

    def test_get_config_mode_reads_config(self):
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.read_text.return_value = "GOOSE_PROVIDER: openai\nGOOSE_MODE: approve\n"
        with patch(
            "jupyter_ai_acp_client.acp_personas.goose._get_user_config_path",
            return_value=mock_path,
        ):
            assert _get_config_mode() == "approve"

    def test_get_config_mode_ignores_provider_only_config(self):
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.read_text.return_value = "GOOSE_PROVIDER: openai\n"
        with patch(
            "jupyter_ai_acp_client.acp_personas.goose._get_user_config_path",
            return_value=mock_path,
        ):
            assert _get_config_mode() is None


class TestGooseModeDecision:
    """Tests for Goose ACP mode resolution."""

    def test_defaults_to_approve_when_unset(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "jupyter_ai_acp_client.acp_personas.goose._get_config_mode",
                return_value=None,
            ):
                decision = _resolve_mode_decision()
        assert decision.explicit is False
        assert decision.mode == "approve"
        assert decision.env is not None
        assert decision.env["GOOSE_MODE"] == "approve"

    def test_prefers_env_over_config(self):
        with patch.dict("os.environ", {"GOOSE_MODE": "auto"}, clear=True):
            with patch(
                "jupyter_ai_acp_client.acp_personas.goose._get_config_mode",
                return_value="smart_approve",
            ):
                decision = _resolve_mode_decision()
        assert decision.explicit is True
        assert decision.mode == "auto"
        assert decision.env is None

    def test_uses_config_mode_when_present(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "jupyter_ai_acp_client.acp_personas.goose._get_config_mode",
                return_value="smart_approve",
            ):
                decision = _resolve_mode_decision()
        assert decision.explicit is True
        assert decision.mode == "smart_approve"
        assert decision.env is None
