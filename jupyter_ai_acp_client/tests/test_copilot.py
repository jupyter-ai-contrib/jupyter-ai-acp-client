"""Tests for the GitHub Copilot ACP persona and model selection."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from acp.exceptions import RequestError
from acp.schema import SessionConfigOptionSelect, SessionConfigSelectOption
from jupyter_ai_persona_manager import PersonaRequirementsUnmet

with patch("shutil.which", return_value="/usr/local/bin/copilot"):
    from jupyter_ai_acp_client.acp_personas.copilot import (
        DEFAULT_COPILOT_MODELS,
        CopilotAcpPersona,
        _check_copilot,
        _is_auth_error,
    )
from jupyter_ai_acp_client.legacy_models import AcpModelOption, AcpModels


class TestIsAuthError:
    """Tests for _is_auth_error() keyword matching."""

    @pytest.mark.parametrize(
        "message",
        [
            "Authentication required",
            "Please login to continue",
            "Not signed in to GitHub Copilot",
            "Not authenticated",
            "Invalid token provided",
            "Credential check failed",
            "403 Forbidden",
            "Unauthorized request",
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
            "Session expired",
        ],
    )
    def test_ignores_non_auth_errors(self, message):
        error = Exception(message)
        assert _is_auth_error(error) is False


class TestCheckCopilot:
    """Tests for _check_copilot() executable guard."""

    def test_not_installed_raises(self):
        with patch("shutil.which", return_value=None):
            with pytest.raises(PersonaRequirementsUnmet, match="requires the GitHub Copilot CLI"):
                _check_copilot()

    def test_installed_succeeds(self):
        with patch("shutil.which", return_value="/usr/local/bin/copilot"):
            _check_copilot()


def _copilot_persona():
    """Build a CopilotAcpPersona instance for unit testing without full server."""
    with patch("shutil.which", return_value="/usr/local/bin/copilot"), \
         patch.object(CopilotAcpPersona, "emit_once"):
        persona = CopilotAcpPersona.__new__(CopilotAcpPersona)
        persona._acp_modes = []
        persona._acp_current_mode_id = None
        persona._acp_config_options = []
        persona._acp_context_usage = None
        persona._acp_session_usage = None
        persona._legacy_models = AcpModels(
            current_model_id="auto",
            available_models=list(DEFAULT_COPILOT_MODELS),
        )
        persona.chat = MagicMock()
        persona.chat.get_metadata.return_value = {}
        return persona


class TestCopilotPersonaModels:
    """Tests for Copilot ACP persona model picking and awareness building."""

    def test_copilot_awareness_populates_default_models(self):
        persona = _copilot_persona()
        model_config, settings = persona._build_awareness_config()

        assert model_config.current == "auto"
        option_ids = [opt.id for opt in model_config.options]
        assert "auto" in option_ids
        assert "claude-sonnet-5.5" in option_ids
        assert "gpt-5.6-sol" in option_ids
        assert "gemini-3.8-flash" in option_ids
        assert settings == []

    async def test_update_model_calls_set_session_model(self):
        persona = _copilot_persona()
        client = MagicMock()
        client.set_session_model = AsyncMock()
        persona.get_client = AsyncMock(return_value=client)
        persona.get_session_id = AsyncMock(return_value="copilot-sess-1")
        persona.set_acp_config_option = AsyncMock()

        await persona.update_model("claude-sonnet-5.5")

        client.set_session_model.assert_awaited_once_with("claude-sonnet-5.5", "copilot-sess-1")
        assert persona._legacy_models.current_model_id == "claude-sonnet-5.5"
        persona.set_acp_config_option.assert_not_awaited()

    def test_genuine_config_option_wins_over_default_copilot_models(self):
        persona = _copilot_persona()
        persona._acp_config_options = [
            SessionConfigOptionSelect(
                id="model",
                name="Model",
                type="select",
                category="model",
                currentValue="custom-copilot-model",
                options=[
                    SessionConfigSelectOption(value="custom-copilot-model", name="Custom Model"),
                ],
            )
        ]

        model_config, _ = persona._build_awareness_config()
        assert model_config.current == "custom-copilot-model"
        assert [o.id for o in model_config.options] == ["custom-copilot-model"]

    def test_set_legacy_models_updates_copilot_models(self):
        persona = _copilot_persona()
        custom_models = AcpModels(
            current_model_id="gpt-5.3-codex",
            available_models=[
                AcpModelOption(model_id="gpt-5.3-codex", name="GPT-5.3 Codex"),
            ],
        )
        persona.set_legacy_models(custom_models)

        assert persona._legacy_models == custom_models
        model_config, _ = persona._build_awareness_config()
        assert model_config.current == "gpt-5.3-codex"
        assert [o.id for o in model_config.options] == ["gpt-5.3-codex"]

