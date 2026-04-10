"""Tests for attachment resolution and stale session recovery in BaseAcpPersona."""

from unittest.mock import AsyncMock, MagicMock

from jupyterlab_chat.models import Message, User

from jupyter_ai_acp_client.base_acp_persona import BaseAcpPersona


def _make_chat_message(id: str, body: str, sender: str, deleted: bool | None = None) -> Message:
    return Message(id=id, body=body, sender=sender, time=0.0, deleted=deleted)


def _make_persona(attachments_map: dict | None = None):
    """Create a minimal mock BaseAcpPersona for testing process_message."""
    persona = MagicMock()
    persona.get_client = AsyncMock()
    persona.get_session_id = AsyncMock(return_value="sess-1")
    persona.is_authed = AsyncMock(return_value=True)
    persona._recovered_from_stale_session = False

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
    msg.body = body
    msg.attachments = attachment_ids
    return msg


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


class TestStaleSessionRecovery:
    """Tests for history injection and flag behavior after stale session recovery."""

    async def test_recovery_flag_resets_after_first_message(self):
        """History is injected only on the first message after recovery, not subsequent ones."""
        client = _make_client()
        persona = _make_persona()
        persona._recovered_from_stale_session = True
        persona.get_client.return_value = client

        msg1 = _make_message("@bot first")
        await BaseAcpPersona.process_message(persona, msg1)

        # Flag must be cleared — second message must NOT get history prepended
        assert persona._recovered_from_stale_session is False

        msg2 = _make_message("@bot second")
        await BaseAcpPersona.process_message(persona, msg2)

        second_call_prompt = client.prompt_and_reply.call_args_list[1].kwargs["prompt"]
        assert not second_call_prompt.startswith("Here is the conversation history")

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
        persona._recovered_from_stale_session = True
        persona._MAX_HISTORY_MESSAGES = BaseAcpPersona._MAX_HISTORY_MESSAGES
        persona.ychat.get_messages.return_value = [
            _make_chat_message("msg-1", "hello world", "user-1"),
        ]
        persona.ychat.get_users.return_value = {}
        persona.get_client.return_value = client
        # Delegate to the real method so history is built from ychat
        persona._build_history_context = lambda **kw: BaseAcpPersona._build_history_context(persona, **kw)
        msg = _make_message("@bot follow up")

        await BaseAcpPersona.process_message(persona, msg)

        prompt = client.prompt_and_reply.call_args.kwargs["prompt"]
        assert prompt.startswith("Here is the conversation history")
        assert "hello world" in prompt
        assert "follow up" in prompt

    def test_build_history_context_caps_at_max_messages(self):
        """History is capped at _MAX_HISTORY_MESSAGES to prevent context window overflow."""
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
