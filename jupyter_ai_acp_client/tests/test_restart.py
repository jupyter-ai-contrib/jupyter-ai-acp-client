"""Tests for the ACP agent subprocess restart mechanism."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
import tornado.web

from jupyter_ai_acp_client.base_acp_persona import BaseAcpPersona
from jupyter_ai_acp_client.routes import RestartSubprocessHandler


class ConcretePersona(BaseAcpPersona):
    """Concrete subclass for testing (avoids ClassVar sharing with base)."""

    _before_subprocess_future = None
    _subprocess_future = None
    _client_future = None


def _make_restart_persona():
    """Create a mock persona suitable for restart tests."""
    persona = MagicMock(spec=ConcretePersona)
    persona.__class__ = ConcretePersona
    persona.restart = AsyncMock()
    user_mock = MagicMock()
    user_mock.mention_name = "test-bot"
    persona.as_user.return_value = user_mock
    return persona


class TestRestartMethod:
    """Tests for BaseAcpPersona.restart()."""

    async def test_restart_kills_subprocess_and_reinitializes(self):
        """restart() should kill the subprocess and reset class futures."""
        persona = MagicMock()
        persona.__class__ = ConcretePersona
        persona.log = MagicMock()
        persona.event_loop = asyncio.get_event_loop()
        persona._executable = ["echo", "test"]

        # Mock subprocess
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.wait = AsyncMock()

        # Mock client
        mock_client = MagicMock()
        mock_client._personas_by_session = {"sess-1": persona}
        mock_conn = MagicMock()
        mock_conn.close = AsyncMock()
        mock_client.get_connection = AsyncMock(return_value=mock_conn)

        persona.get_client = AsyncMock(return_value=mock_client)
        persona.get_agent_subprocess = AsyncMock(return_value=mock_proc)
        persona.before_agent_subprocess = AsyncMock()
        persona._init_agent_subprocess = AsyncMock(return_value=mock_proc)
        persona._init_client = AsyncMock(return_value=mock_client)
        persona._init_client_session = AsyncMock()

        with patch("os.getpgid", return_value=12345), patch("os.killpg") as mock_killpg:
            await BaseAcpPersona.restart(persona)

        # Verify subprocess was killed
        mock_killpg.assert_called()
        # Verify connection was closed
        mock_conn.close.assert_called_once()

    async def test_restart_reinitializes_sessions_for_siblings(self):
        """restart() should re-init sessions for all personas sharing the subprocess."""
        persona1 = MagicMock()
        persona1.__class__ = ConcretePersona
        persona1.log = MagicMock()
        persona1.event_loop = asyncio.get_event_loop()
        persona1._executable = ["echo"]

        persona2 = MagicMock()
        persona2.__class__ = ConcretePersona
        persona2.log = MagicMock()
        persona2.event_loop = asyncio.get_event_loop()

        mock_proc = MagicMock()
        mock_proc.pid = 100
        mock_proc.wait = AsyncMock()

        mock_client = MagicMock()
        mock_client._personas_by_session = {"s1": persona1, "s2": persona2}
        mock_conn = MagicMock()
        mock_conn.close = AsyncMock()
        mock_client.get_connection = AsyncMock(return_value=mock_conn)

        persona1.get_client = AsyncMock(return_value=mock_client)
        persona1.get_agent_subprocess = AsyncMock(return_value=mock_proc)
        persona1.before_agent_subprocess = AsyncMock()
        persona1._init_agent_subprocess = AsyncMock(return_value=mock_proc)
        persona1._init_client = AsyncMock(return_value=mock_client)
        persona1._init_client_session = AsyncMock()
        persona2._init_client_session = AsyncMock()

        with patch("os.getpgid", return_value=100), patch("os.killpg"):
            await BaseAcpPersona.restart(persona1)

        # Both personas should have their session re-initialized
        persona1._init_client_session.assert_called_once()
        persona2._init_client_session.assert_called_once()

    async def test_restart_handles_already_dead_subprocess(self):
        """restart() should handle ProcessLookupError gracefully."""
        persona = MagicMock()
        persona.__class__ = ConcretePersona
        persona.log = MagicMock()
        persona.event_loop = asyncio.get_event_loop()
        persona._executable = ["echo"]

        mock_proc = MagicMock()
        mock_proc.pid = 999

        mock_client = MagicMock()
        mock_client._personas_by_session = {}
        mock_conn = MagicMock()
        mock_conn.close = AsyncMock()
        mock_client.get_connection = AsyncMock(return_value=mock_conn)

        persona.get_client = AsyncMock(return_value=mock_client)
        persona.get_agent_subprocess = AsyncMock(return_value=mock_proc)
        persona.before_agent_subprocess = AsyncMock()
        persona._init_agent_subprocess = AsyncMock(return_value=mock_proc)
        persona._init_client = AsyncMock(return_value=mock_client)
        persona._init_client_session = AsyncMock()

        with patch("os.getpgid", side_effect=ProcessLookupError):
            # Should not raise
            await BaseAcpPersona.restart(persona)

        # Class futures should still be reset and re-initialized
        assert ConcretePersona._before_subprocess_future is not None


class TestRestartSubprocessHandler:
    """Tests for the REST endpoint handler."""

    def _make_handler(self, personas: dict):
        """Create a mock RestartSubprocessHandler."""
        handler = object.__new__(RestartSubprocessHandler)
        handler.application = MagicMock()
        handler.request = MagicMock()
        handler.request.connection = MagicMock()
        handler._transforms = []

        serverapp = MagicMock()
        file_id_manager = MagicMock()
        file_id_manager.get_id.return_value = "file-id-1"

        persona_manager = MagicMock()
        persona_manager.personas = personas

        serverapp.web_app.settings = {
            "file_id_manager": file_id_manager,
            "jupyter-ai": {
                "persona-managers": {
                    "text:chat:file-id-1": persona_manager,
                },
            },
        }

        type(handler).serverapp = PropertyMock(return_value=serverapp)
        return handler

    async def test_restart_requires_chat_path(self):
        """POST without chat_path should return 400."""
        handler = self._make_handler({})
        with patch.object(handler, "get_argument", return_value=None):
            with patch.object(handler, "get_current_user", return_value={"name": "test"}):
                with pytest.raises(tornado.web.HTTPError) as exc_info:
                    await handler.post()
                assert exc_info.value.status_code == 400

    async def test_restart_chat_not_found(self):
        """POST with unknown chat_path should return 404."""
        handler = self._make_handler({})
        serverapp = handler.serverapp
        serverapp.web_app.settings["file_id_manager"].get_id.return_value = None

        with patch.object(handler, "get_argument", return_value="nonexistent.chat"):
            with patch.object(handler, "get_current_user", return_value={"name": "test"}):
                with pytest.raises(tornado.web.HTTPError) as exc_info:
                    await handler.post()
                assert exc_info.value.status_code == 404

    async def test_restart_calls_persona_restart(self):
        """POST should call restart() on all ACP personas in the chat."""
        persona = _make_restart_persona()
        handler = self._make_handler({"p1": persona})

        with patch.object(handler, "get_argument", return_value="test.chat"):
            with patch.object(handler, "get_current_user", return_value={"name": "test"}):
                with patch.object(handler, "finish") as mock_finish:
                    # isinstance check needs to work with our mock
                    with patch(
                        "jupyter_ai_acp_client.routes.isinstance",
                        side_effect=lambda obj, cls: obj is persona if cls is BaseAcpPersona else isinstance(obj, cls),
                    ):
                        await handler.post()

                    persona.restart.assert_called_once()
                    mock_finish.assert_called_once()
                    result = mock_finish.call_args[0][0]
                    assert result["status"] == "restarted"
                    assert "test-bot" in result["restarted"]

    async def test_restart_skips_non_acp_personas(self):
        """POST should skip personas that are not BaseAcpPersona instances."""
        non_acp = MagicMock()
        non_acp.__class__ = type("OtherPersona", (), {})
        handler = self._make_handler({"p1": non_acp})

        with patch.object(handler, "get_argument", return_value="test.chat"):
            with patch.object(handler, "get_current_user", return_value={"name": "test"}):
                with patch.object(handler, "finish") as mock_finish:
                    await handler.post()
                    result = mock_finish.call_args[0][0]
                    assert result["restarted"] == []
