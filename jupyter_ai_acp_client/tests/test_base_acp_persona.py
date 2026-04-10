"""Tests for attachment resolution and load-session recovery in BaseAcpPersona."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from acp.exceptions import RequestError
from jupyterlab_chat.models import Message

from jupyter_ai_acp_client.base_acp_persona import BaseAcpPersona


def _make_chat_message(
    id: str, body: str, sender: str, deleted: bool | None = None
) -> Message:
    return Message(id=id, body=body, sender=sender, time=0.0, deleted=deleted)


def _make_persona(attachments_map: dict | None = None):
    """Create a minimal mock BaseAcpPersona for testing process_message."""
    persona = MagicMock()
    persona.get_client = AsyncMock()
    persona.get_session_id = AsyncMock(return_value="sess-1")
    persona.is_authed = AsyncMock(return_value=True)
    persona._pending_session_recovery_context = False

    # as_user() is sync — must return a regular MagicMock
    user_mock = MagicMock()
    user_mock.mention_name = "bot"
    persona.as_user.return_value = user_mock

    # YChat mock
    ychat = MagicMock()
    ychat.get_attachments.return_value = attachments_map or {}
    ychat.get_messages.return_value = []
    ychat.get_users.return_value = {}
    persona.ychat = ychat

    # parent.root_dir
    persona.parent = MagicMock()
    persona.parent.root_dir = "/home/user/notebooks"

    return persona


def _make_client():
    """Create an AsyncMock client with prompt_and_reply explicitly async."""
    client = AsyncMock()
    client.prompt_and_reply = AsyncMock()
    return client


def _make_message(body: str, attachment_ids: list[str] | None = None):
    msg = MagicMock()
    msg.id = "current-msg"
    msg.body = body
    msg.attachments = attachment_ids
    return msg


class _BaseLoadSessionRecoveryPersona:
    """Minimal duck-typed object with the base load-session recovery policy."""

    id = "test-persona"

    def _is_recoverable_load_session_error(self, error: RequestError) -> bool:
        return BaseAcpPersona._is_recoverable_load_session_error(self, error)


class _ExtraLoadSessionRecoveryPersona(_BaseLoadSessionRecoveryPersona):
    """Minimal subclass used to test persona-specific recovery semantics."""

    def _is_recoverable_load_session_error(self, error: RequestError) -> bool:
        return (
            super()._is_recoverable_load_session_error(error)
            or error.code == -32603
        )


def _make_session_init_persona(
    *,
    error: Exception | None = None,
    existing_session_id: str | None = "old-session",
    supports_session_load: bool = True,
    persona_cls: type[_BaseLoadSessionRecoveryPersona] = (
        _BaseLoadSessionRecoveryPersona
    ),
):
    """Create an uninitialized persona wired for _init_client_session tests."""
    persona = persona_cls()
    persona.log = MagicMock()
    persona._pending_session_recovery_context = False

    client = MagicMock()
    capabilities = MagicMock()
    capabilities.load_session = supports_session_load
    client.get_agent_capabilities = AsyncMock(return_value=capabilities)
    persona.get_client = AsyncMock(return_value=client)

    sessions = {}
    if existing_session_id:
        sessions[persona.id] = existing_session_id
    persona._get_existing_sessions = MagicMock(return_value=sessions)
    persona._load_session = AsyncMock()
    persona._create_session = AsyncMock(
        return_value=MagicMock(session_id="new-session")
    )

    if error:
        persona._load_session.side_effect = error
    else:
        persona._load_session.return_value = MagicMock(session_id=existing_session_id)

    return persona, client


class TestProcessMessageAttachments:
    """Tests for how process_message resolves attachments and calls prompt_and_reply."""

    async def test_no_attachments(self):
        """When message has no attachments, prompt_and_reply is called without them."""
        client = _make_client()
        persona = _make_persona()
        persona.get_client.return_value = client
        msg = _make_message("@bot hello")

        await BaseAcpPersona.process_message(persona, msg)

        client.prompt_and_reply.assert_called_once_with(
            session_id="sess-1",
            prompt="hello",
            attachments=None,
            root_dir="/home/user/notebooks",
        )

    async def test_single_attachment(self):
        """A single known attachment ID resolves to a dict."""
        client = _make_client()
        att_map = {
            "att-1": {"value": "file.py", "type": "file", "mimetype": "text/x-python"},
        }
        persona = _make_persona(att_map)
        persona.get_client.return_value = client
        msg = _make_message("@bot check this", ["att-1"])

        await BaseAcpPersona.process_message(persona, msg)

        call_kwargs = client.prompt_and_reply.call_args.kwargs
        assert call_kwargs["attachments"] == [att_map["att-1"]]

    async def test_multiple_attachments(self):
        """Multiple attachment IDs all resolve in order."""
        client = _make_client()
        att_map = {
            "att-1": {"value": "a.py", "type": "file"},
            "att-2": {"value": "b.ipynb", "type": "notebook"},
        }
        persona = _make_persona(att_map)
        persona.get_client.return_value = client
        msg = _make_message("@bot review", ["att-1", "att-2"])

        await BaseAcpPersona.process_message(persona, msg)

        call_kwargs = client.prompt_and_reply.call_args.kwargs
        assert call_kwargs["attachments"] == [att_map["att-1"], att_map["att-2"]]

    async def test_unknown_attachment_id_skipped(self):
        """Unknown attachment IDs are silently skipped with a log warning."""
        client = _make_client()
        persona = _make_persona({})
        persona.get_client.return_value = client
        msg = _make_message("@bot check", ["nonexistent"])

        await BaseAcpPersona.process_message(persona, msg)

        call_kwargs = client.prompt_and_reply.call_args.kwargs
        assert call_kwargs["attachments"] is None

    async def test_partial_resolution(self):
        """Only known IDs are resolved; unknown ones are skipped."""
        client = _make_client()
        att_map = {"att-1": {"value": "good.py", "type": "file"}}
        persona = _make_persona(att_map)
        persona.get_client.return_value = client
        msg = _make_message("@bot check", ["att-1", "missing"])

        await BaseAcpPersona.process_message(persona, msg)

        call_kwargs = client.prompt_and_reply.call_args.kwargs
        assert call_kwargs["attachments"] == [att_map["att-1"]]

    async def test_empty_attachment_list(self):
        """An empty attachment list results in None."""
        client = _make_client()
        persona = _make_persona()
        persona.get_client.return_value = client
        msg = _make_message("@bot hi", [])

        await BaseAcpPersona.process_message(persona, msg)

        call_kwargs = client.prompt_and_reply.call_args.kwargs
        assert call_kwargs["attachments"] is None

    async def test_root_dir_passed(self):
        """root_dir from persona.parent is forwarded to prompt_and_reply."""
        client = _make_client()
        persona = _make_persona()
        persona.parent.root_dir = "/custom/root"
        persona.get_client.return_value = client
        msg = _make_message("@bot hi")

        await BaseAcpPersona.process_message(persona, msg)

        call_kwargs = client.prompt_and_reply.call_args.kwargs
        assert call_kwargs["root_dir"] == "/custom/root"


class TestLoadSessionRecovery:
    """Tests for history injection and flag behavior after load-session recovery."""

    async def test_resource_not_found_load_error_creates_new_session(self):
        """ACP ResourceNotFound from load_session is treated as stale session."""
        persona, client = _make_session_init_persona(
            error=RequestError(-32002, "Resource not found")
        )

        await BaseAcpPersona._init_client_session(persona)

        persona._load_session.assert_awaited_once_with(client, "old-session")
        persona._create_session.assert_awaited_once_with(client)
        assert persona._pending_session_recovery_context is True

    async def test_non_retryable_load_error_propagates(self):
        """Non-stale load_session errors are not masked by a new session."""
        persona, _ = _make_session_init_persona(
            error=RequestError(-32603, "Internal error", "auth failed")
        )

        with pytest.raises(RequestError):
            await BaseAcpPersona._init_client_session(persona)

        persona._create_session.assert_not_awaited()
        assert persona._pending_session_recovery_context is False

    async def test_subclass_recovery_only_applies_to_load_session(self):
        """Persona-specific recovery handles load_session failures only."""
        persona, client = _make_session_init_persona(
            error=RequestError(-32603, "Internal error"),
            persona_cls=_ExtraLoadSessionRecoveryPersona,
        )

        await BaseAcpPersona._init_client_session(persona)

        persona._load_session.assert_awaited_once_with(client, "old-session")
        persona._create_session.assert_awaited_once_with(client)
        assert persona._pending_session_recovery_context is True

    async def test_subclass_recovery_does_not_mask_create_session_error(self):
        """Recoverable load-session errors must not mask create_session()."""
        persona, client = _make_session_init_persona(
            existing_session_id=None,
            persona_cls=_ExtraLoadSessionRecoveryPersona,
        )
        error = RequestError(-32603, "Internal error", "create failed")
        persona._create_session.side_effect = error

        with pytest.raises(RequestError) as exc_info:
            await BaseAcpPersona._init_client_session(persona)

        assert exc_info.value is error
        persona._load_session.assert_not_awaited()
        persona._create_session.assert_awaited_once_with(client)
        assert persona._pending_session_recovery_context is False

    async def test_recovery_flag_resets_after_first_message(self):
        """History is injected only on the first message after recovery."""
        client = _make_client()
        persona = _make_persona()
        persona._pending_session_recovery_context = True
        persona.get_client.return_value = client

        msg1 = _make_message("@bot first")
        await BaseAcpPersona.process_message(persona, msg1)

        # Flag must be cleared — second message must NOT get history prepended
        assert persona._pending_session_recovery_context is False

        msg2 = _make_message("@bot second")
        await BaseAcpPersona.process_message(persona, msg2)

        second_call_prompt = client.prompt_and_reply.call_args_list[1].kwargs["prompt"]
        assert not second_call_prompt.startswith("The previous ACP session")

    def test_build_history_context_excludes_current_message(self):
        """The current message is not included in the injected history."""
        persona = _make_persona()
        persona._MAX_HISTORY_MESSAGES = BaseAcpPersona._MAX_HISTORY_MESSAGES
        msgs = [
            _make_chat_message("msg-1", "hello", "user-1"),
            _make_chat_message("msg-2", "hi there", "bot-1"),
            _make_chat_message("msg-3", "follow up", "user-1"),  # current message
        ]
        persona.ychat.get_messages.return_value = msgs
        persona.ychat.get_users.return_value = {}

        result = BaseAcpPersona._build_history_context(persona, exclude_id="msg-3")

        assert "follow up" not in result
        assert "hello" in result
        assert "hi there" in result

    async def test_recovery_history_injected_into_prompt(self):
        """History is prepended to the prompt on the first message after recovery."""
        client = _make_client()
        persona = _make_persona()
        persona._pending_session_recovery_context = True
        persona._MAX_HISTORY_MESSAGES = BaseAcpPersona._MAX_HISTORY_MESSAGES
        persona.ychat.get_messages.return_value = [
            _make_chat_message("msg-1", "hello world", "user-1"),
        ]
        persona.ychat.get_users.return_value = {}
        persona.get_client.return_value = client
        # Delegate to the real method so history is built from ychat
        persona._build_history_context = (
            lambda **kw: BaseAcpPersona._build_history_context(persona, **kw)
        )
        msg = _make_message("@bot follow up")

        await BaseAcpPersona.process_message(persona, msg)

        prompt = client.prompt_and_reply.call_args.kwargs["prompt"]
        assert prompt.startswith("The previous ACP session")
        assert "hello world" in prompt
        assert "follow up" in prompt

    def test_build_history_context_caps_at_max_messages(self):
        """History is capped at _MAX_HISTORY_MESSAGES."""
        persona = _make_persona()
        cap = BaseAcpPersona._MAX_HISTORY_MESSAGES
        persona._MAX_HISTORY_MESSAGES = cap
        msgs = [
            _make_chat_message(f"msg-{i}", f"message {i}", "user-1")
            for i in range(cap + 10)
        ]
        persona.ychat.get_messages.return_value = msgs
        persona.ychat.get_users.return_value = {}

        result = BaseAcpPersona._build_history_context(persona)

        # Only the last _MAX_HISTORY_MESSAGES messages should appear
        lines = result.splitlines()
        message_lines = [l for l in lines if l.startswith("user-1:")]
        assert len(message_lines) == cap
        # The oldest messages are trimmed, most recent are kept
        assert f"message {cap + 9}" in result
        assert "message 0" not in result
