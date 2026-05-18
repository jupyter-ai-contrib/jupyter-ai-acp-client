"""Tests for KiroAcpPersona.handle_no_auth auth-wait-then-process behavior."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from jupyterlab_chat.models import Message


def _make_message(body: str = "hello @Kiro") -> Message:
    return Message(id="msg-1", body=body, sender="user-1", time=0.0)


class TestHandleNoAuth:
    """Tests that handle_no_auth waits for auth then processes the message."""

    @pytest.mark.asyncio
    async def test_waits_for_auth_then_processes_message(self):
        """After auth completes, handle_no_auth should call super().process_message."""
        with patch(
            "jupyter_ai_acp_client.acp_personas.kiro.shutil.which", return_value="/usr/local/bin/kiro-cli"
        ), patch(
            "jupyter_ai_acp_client.acp_personas.kiro.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="1.25.0", stderr="")

            from jupyter_ai_acp_client.acp_personas.kiro import KiroAcpPersona

        persona = MagicMock(spec=KiroAcpPersona)
        persona._terminal_opened = False
        persona.send_message = MagicMock()
        persona._should_use_device_flow = AsyncMock(return_value=False)
        persona._open_kiro_login_terminal = AsyncMock(return_value=False)

        # Simulate _before_subprocess_future that resolves immediately
        fut = asyncio.get_event_loop().create_future()
        fut.set_result(None)
        persona.__class__ = KiroAcpPersona
        KiroAcpPersona._before_subprocess_future = fut

        # Mock super().process_message
        with patch.object(
            KiroAcpPersona.__mro__[1], "process_message", new_callable=AsyncMock
        ) as mock_process:
            msg = _make_message()
            await KiroAcpPersona.handle_no_auth(persona, msg)

            mock_process.assert_called_once_with(msg)

    @pytest.mark.asyncio
    async def test_waits_for_auth_before_processing(self):
        """handle_no_auth should block until _before_subprocess_future resolves."""
        with patch(
            "jupyter_ai_acp_client.acp_personas.kiro.shutil.which", return_value="/usr/local/bin/kiro-cli"
        ), patch(
            "jupyter_ai_acp_client.acp_personas.kiro.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="1.25.0", stderr="")

            from jupyter_ai_acp_client.acp_personas.kiro import KiroAcpPersona

        persona = MagicMock(spec=KiroAcpPersona)
        persona._terminal_opened = False
        persona.send_message = MagicMock()
        persona._should_use_device_flow = AsyncMock(return_value=False)
        persona._open_kiro_login_terminal = AsyncMock(return_value=False)

        # Simulate _before_subprocess_future that hasn't resolved yet
        fut = asyncio.get_event_loop().create_future()
        persona.__class__ = KiroAcpPersona
        KiroAcpPersona._before_subprocess_future = fut

        process_called = False

        async def mock_process(message):
            nonlocal process_called
            process_called = True

        with patch.object(
            KiroAcpPersona.__mro__[1], "process_message", side_effect=mock_process
        ):
            msg = _make_message()
            task = asyncio.create_task(KiroAcpPersona.handle_no_auth(persona, msg))

            # Give the event loop a chance to run
            await asyncio.sleep(0.01)
            assert not process_called, "process_message should not be called before auth completes"

            # Resolve auth
            fut.set_result(None)
            await task

            assert process_called, "process_message should be called after auth completes"
