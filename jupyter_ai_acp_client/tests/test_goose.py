"""Tests for the Goose ACP persona helper functions."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from acp.exceptions import RequestError
from jupyter_ai_persona_manager import PersonaRequirementsUnmet

_mock_run = MagicMock()
_mock_run.returncode = 0
_mock_run.stdout = "goose 1.28.0"
_mock_run.stderr = ""

with patch("shutil.which", return_value="/usr/bin/goose"), \
     patch("subprocess.run", return_value=_mock_run):
    from jupyter_ai_acp_client.acp_personas.goose import (
        GooseAcpPersona,
        _check_goose,
        _get_config_mode,
        _get_explicit_provider,
        _get_user_config_path,
        _is_setup_error,
        _parse_goose_config,
        _resolve_mode_decision,
    )


class TestIsSetupError:
    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (RequestError(-32000, "Authentication required"), True),
            (
                RequestError(
                    -32603,
                    "Internal error",
                    "Failed to set provider: Configuration value not found: GOOSE_PROVIDER",
                ),
                True,
            ),
            (
                RequestError(
                    -32603,
                    "Internal error",
                    "Failed to create session: database error",
                ),
                True,
            ),
            (
                RequestError(
                    -32603,
                    "Internal error",
                    "Authentication error: Authentication failed. Status: 401 Unauthorized. Response: invalid x-api-key",
                ),
                True,
            ),
            (RequestError(-32603, "Internal error"), True),
            (
                RequestError(
                    -32603,
                    "Internal error",
                    "Error getting agent reply: timeout",
                ),
                False,
            ),
            (
                RequestError(
                    -32002,
                    "Resource not found",
                    "Session not found: abc123",
                ),
                False,
            ),
            (RequestError(-32601, "Method not found"), False),
        ],
    )
    def test_request_error_classification(self, error, expected):
        assert _is_setup_error(error) is expected

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

    @pytest.mark.parametrize(
        ("version", "error_match"),
        [
            ("1.8.0", None),
            ("1.7.0", ">=1.8.0"),
            ("2.0.0", ">=1.8.0,<2"),
        ],
    )
    def test_version_gate(self, version, error_match):
        with patch("shutil.which", return_value="/usr/bin/goose"), \
             patch("subprocess.run", return_value=_mock_result(f"goose {version}")):
            if error_match is None:
                _check_goose()
            else:
                with pytest.raises(PersonaRequirementsUnmet, match=error_match):
                    _check_goose()


class TestGooseConfigParsing:
    def test_get_user_config_path_prefers_goose_path_root(self):
        with patch.dict(
            "os.environ",
            {"GOOSE_PATH_ROOT": "/tmp/goose-test", "XDG_CONFIG_HOME": "/tmp/xdg"},
            clear=True,
        ):
            assert _get_user_config_path() == (
                Path("/tmp/goose-test") / "config" / "config.yaml"
            )

    def test_parse_goose_config_mapping(self):
        assert _parse_goose_config("GOOSE_MODE: approve\nGOOSE_PROVIDER: openai\n") == {
            "GOOSE_MODE": "approve",
            "GOOSE_PROVIDER": "openai",
        }

    def test_parse_goose_config_invalid_yaml(self):
        assert _parse_goose_config("GOOSE_MODE: [approve\n") is None

    def test_parse_goose_config_non_mapping(self):
        assert _parse_goose_config("- approve\n- openai\n") is None

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

    def test_get_config_mode_ignores_non_string_value(self):
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.read_text.return_value = "GOOSE_MODE: 1\n"
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


class TestGooseProviderDetection:
    def test_prefers_env_provider_over_config(self):
        with patch.dict("os.environ", {"GOOSE_PROVIDER": "anthropic"}, clear=True):
            with patch(
                "jupyter_ai_acp_client.acp_personas.goose._get_config_value",
                return_value="openai",
            ):
                assert _get_explicit_provider() == "anthropic"

    def test_uses_config_provider_when_present(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "jupyter_ai_acp_client.acp_personas.goose._get_config_value",
                return_value="openai",
            ):
                assert _get_explicit_provider() == "openai"

    async def test_is_authed_false_without_provider(self):
        with patch(
            "jupyter_ai_acp_client.acp_personas.goose._get_explicit_provider",
            return_value=None,
        ):
            assert await GooseAcpPersona.is_authed(MagicMock()) is False

    async def test_is_authed_true_with_provider(self):
        with patch(
            "jupyter_ai_acp_client.acp_personas.goose._get_explicit_provider",
            return_value="openai",
        ):
            assert await GooseAcpPersona.is_authed(MagicMock()) is True
