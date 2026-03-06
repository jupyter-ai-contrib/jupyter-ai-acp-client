# import json

from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import tornado.web
from tornado.httpclient import HTTPClientError

from jupyter_ai_acp_client.routes import AcpSlashCommandsHandler


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
    persona_manager.last_mentioned_persona = None
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
