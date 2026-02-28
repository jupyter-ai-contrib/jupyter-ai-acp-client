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


def _make_handler_and_serverapp(
    chat_path="chat/test.chat",
    file_id="file-id-123",
    personas=None,
):
    """
    Build a minimal AcpSlashCommandsHandler with mocked internals,
    bypassing __init__ to avoid Tornado application setup.

    Returns (handler, mock_serverapp) — the caller must patch
    AcpSlashCommandsHandler.serverapp with PropertyMock(return_value=mock_serverapp)
    before calling handler.get().
    """
    if personas is None:
        personas = {}

    mock_persona_manager = MagicMock()
    mock_persona_manager.personas = personas
    mock_persona_manager.last_mentioned_persona = None
    mock_persona_manager.default_persona = None

    file_id_manager = MagicMock()
    file_id_manager.get_id.return_value = file_id

    room_id = f"text:chat:{file_id}"
    settings = {
        "file_id_manager": file_id_manager,
        "jupyter-ai": {
            "persona-managers": {
                room_id: mock_persona_manager,
            }
        },
    }

    mock_serverapp = MagicMock()
    mock_serverapp.web_app.settings = settings

    handler = object.__new__(AcpSlashCommandsHandler)
    # Set _jupyter_current_user so the @tornado.web.authenticated decorator
    # does not attempt to call get_current_user() (which requires full setup).
    handler._jupyter_current_user = "test-user"
    handler.get_argument = MagicMock(
        side_effect=lambda name, default=None: {
            "chat_path": chat_path,
        }.get(name, default)
    )
    handler.finish = MagicMock()

    return handler, mock_serverapp


def test_persona_not_found_raises_http_404_not_unbound_local_error():
    """
    BE-1 regression test: when persona_mention_name is given but no persona in
    persona_manager.personas matches, the handler must raise HTTPError(404),
    NOT UnboundLocalError.

    Before the fix, `persona` was only assigned inside the for-loop body, so
    an empty personas dict caused `if not persona:` to raise UnboundLocalError.
    """
    handler, mock_serverapp = _make_handler_and_serverapp(personas={})

    with patch.object(
        AcpSlashCommandsHandler,
        "serverapp",
        new_callable=PropertyMock,
        return_value=mock_serverapp,
    ):
        with pytest.raises(tornado.web.HTTPError) as exc_info:
            handler.get(persona_mention_name="nonexistent-bot")

    assert exc_info.value.status_code == 404
    assert "nonexistent-bot" in exc_info.value.log_message


def test_persona_not_found_with_non_matching_personas_raises_http_404():
    """
    BE-1 regression test (variant): when personas exist but none match the
    given mention_name, the handler must raise HTTPError(404), NOT
    UnboundLocalError.
    """
    mock_persona = MagicMock()
    mock_persona.as_user.return_value.mention_name = "other-bot"

    handler, mock_serverapp = _make_handler_and_serverapp(
        personas={"other-bot-id": mock_persona}
    )

    with patch.object(
        AcpSlashCommandsHandler,
        "serverapp",
        new_callable=PropertyMock,
        return_value=mock_serverapp,
    ):
        with pytest.raises(tornado.web.HTTPError) as exc_info:
            handler.get(persona_mention_name="nonexistent-bot")

    assert exc_info.value.status_code == 404
