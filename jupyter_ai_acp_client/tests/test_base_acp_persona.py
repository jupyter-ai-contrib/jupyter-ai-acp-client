"""Tests for attachment resolution in BaseAcpPersona.process_message()."""

from unittest.mock import AsyncMock, MagicMock

from jupyter_ai_acp_client.base_acp_persona import BaseAcpPersona


def _make_persona(attachments_map: dict | None = None):
    """Create a minimal mock BaseAcpPersona for testing process_message."""
    persona = MagicMock()
    persona.get_client = AsyncMock()
    persona.get_session_id = AsyncMock(return_value="sess-1")
    persona.is_authed = AsyncMock(return_value=True)

    # as_user() is sync — must return a regular MagicMock
    user_mock = MagicMock()
    user_mock.mention_name = "bot"
    persona.as_user.return_value = user_mock

    # YChat mock
    ychat = MagicMock()
    ychat.get_attachments.return_value = attachments_map or {}
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
