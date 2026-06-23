# import json

from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import tornado.web
from acp.schema import (
    ModelInfo,
    SessionConfigOptionBoolean,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    SessionMode,
)
from tornado.httpclient import HTTPClientError

from jupyter_ai_acp_client.routes import (
    MODE_CONTROL_ID,
    MODEL_CONTROL_ID,
    AcpSlashCommandsHandler,
    build_controls,
)


async def test_slash_commands_route_no_chat(jp_fetch):
    """
    Expects that the /ai/acp/slash_commands route returns a 400 when no ?chat_path
    URL query argument is given.
    """
    try:
        await jp_fetch("ai", "acp", "slash_commands")
    except HTTPClientError as e:
        assert e.code == 400


def _make_handler_and_serverapp(personas: dict):
    """Create a mock AcpSlashCommandsHandler with controllable personas."""
    app = MagicMock()
    request = MagicMock()
    request.connection = MagicMock()

    handler = object.__new__(AcpSlashCommandsHandler)
    handler.application = app
    handler.request = request
    handler._transforms = []

    # Mock serverapp and settings
    serverapp = MagicMock()
    file_id_manager = MagicMock()
    file_id_manager.get_id.return_value = "file-id-1"

    persona_manager = MagicMock()
    persona_manager.personas = personas
    persona_manager.active_persona = None
    persona_manager.default_persona = None

    serverapp.web_app.settings = {
        "file_id_manager": file_id_manager,
        "jupyter-ai": {
            "persona-managers": {
                "text:chat:file-id-1": persona_manager,
            },
        },
    }

    type(handler).serverapp = PropertyMock(return_value=serverapp)

    return handler, persona_manager


class TestPersonaNotFound:
    """Regression tests for the persona = None initialization bug."""

    def test_persona_not_found_raises_http_404_not_unbound_local_error(self):
        """When persona_mention_name doesn't match any persona, raise 404 (not UnboundLocalError)."""
        mock_persona = MagicMock()
        mock_persona.as_user.return_value.mention_name = "other-bot"
        personas = {"p1": mock_persona}

        handler, _ = _make_handler_and_serverapp(personas)

        with patch.object(handler, "get_argument", return_value="chat.chat"):
            with patch.object(handler, "get_current_user", return_value={"name": "test"}):
                with pytest.raises(tornado.web.HTTPError) as exc_info:
                    handler.get(persona_mention_name="nonexistent")
                assert exc_info.value.status_code == 404
                assert "Persona not found" in str(exc_info.value.log_message)

    def test_persona_not_found_with_non_matching_personas_raises_http_404(self):
        """Multiple personas, none matching the mention name -> 404."""
        p1 = MagicMock()
        p1.as_user.return_value.mention_name = "bot-a"
        p2 = MagicMock()
        p2.as_user.return_value.mention_name = "bot-b"
        personas = {"p1": p1, "p2": p2}

        handler, _ = _make_handler_and_serverapp(personas)

        with patch.object(handler, "get_argument", return_value="chat.chat"):
            with patch.object(handler, "get_current_user", return_value={"name": "test"}):
                with pytest.raises(tornado.web.HTTPError) as exc_info:
                    handler.get(persona_mention_name="bot-c")
                assert exc_info.value.status_code == 404


def _select_option(option_id: str, current: str, values: list[str]) -> SessionConfigOptionSelect:
    return SessionConfigOptionSelect(
        id=option_id,
        name=option_id.title(),
        type="select",
        currentValue=current,
        options=[SessionConfigSelectOption(value=v, name=v.title()) for v in values],
    )


def _persona(*, models=None, current_model=None, modes=None, current_mode=None, config_options=None):
    """A stand-in exposing only the attributes build_controls reads."""
    return SimpleNamespace(
        acp_models=models or [],
        acp_current_model_id=current_model,
        acp_modes=modes or [],
        acp_current_mode_id=current_mode,
        acp_config_options=config_options or [],
    )


class TestBuildControls:
    """Normalization of an ACP persona's model, mode, and config options."""

    def test_model_and_mode_become_selects_in_order(self):
        persona = _persona(
            models=[ModelInfo(modelId="opus", name="Opus", description="big")],
            current_model="opus",
            modes=[SessionMode(id="plan", name="Plan"), SessionMode(id="code", name="Code")],
            current_mode="plan",
        )

        controls = build_controls(persona)

        assert [(c.source, c.id, c.kind) for c in controls] == [
            ("model", MODEL_CONTROL_ID, "select"),
            ("mode", MODE_CONTROL_ID, "select"),
        ]
        model = controls[0]
        assert model.current_value == "opus"
        assert [ch.value for ch in model.choices] == ["opus"]
        assert [ch.value for ch in controls[1].choices] == ["plan", "code"]

    def test_select_and_boolean_config_options_kept_with_their_kind(self):
        persona = _persona(
            config_options=[
                _select_option("effort", "high", ["low", "high"]),
                SessionConfigOptionBoolean(
                    id="allow_all", name="Allow all", type="boolean", currentValue=False
                ),
            ]
        )

        controls = build_controls(persona)

        kinds = {c.id: c.kind for c in controls}
        assert kinds == {"effort": "select", "allow_all": "boolean"}
        boolean = next(c for c in controls if c.id == "allow_all")
        assert boolean.current_value is False
        assert boolean.choices == []

    def test_config_duplicates_of_model_and_mode_are_dropped(self):
        persona = _persona(
            models=[ModelInfo(modelId="opus", name="Opus")],
            current_model="opus",
            modes=[SessionMode(id="plan", name="Plan")],
            current_mode="plan",
            config_options=[
                _select_option("model", "opus", ["opus"]),
                _select_option("mode", "plan", ["plan"]),
                _select_option("effort", "high", ["low", "high"]),
            ],
        )

        controls = build_controls(persona)

        assert [c.id for c in controls] == [MODEL_CONTROL_ID, MODE_CONTROL_ID, "effort"]

    def test_config_model_and_mode_kept_when_no_dedicated_fields(self):
        persona = _persona(
            config_options=[
                _select_option("model", "gpt", ["gpt", "claude"]),
                _select_option("mode", "build", ["build", "plan"]),
            ]
        )

        controls = build_controls(persona)

        assert [(c.source, c.id) for c in controls] == [
            ("config_option", "model"),
            ("config_option", "mode"),
        ]

    def test_persona_with_nothing_advertised_yields_no_controls(self):
        assert build_controls(_persona()) == []
