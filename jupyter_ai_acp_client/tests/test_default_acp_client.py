"""Tests for content block building in JaiAcpClient.prompt_and_reply()."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from acp.schema import ResourceContentBlock, TextContentBlock

from jupyter_ai_acp_client.default_acp_client import JaiAcpClient


SESSION_ID = "sess-1"


def _make_client_and_persona():
    """Create a minimal mock JaiAcpClient with a persona wired for testing."""
    client = object.__new__(JaiAcpClient)
    client._prompt_locks_by_session = {}
    client._cancel_requested = {}
    client._permission_manager = MagicMock()

    # Mock connection
    conn = AsyncMock()
    conn.prompt = AsyncMock(return_value=MagicMock())
    client.get_connection = AsyncMock(return_value=conn)

    # Mock persona
    persona = MagicMock()
    persona.log = MagicMock()
    persona.awareness = MagicMock()
    persona.ychat = MagicMock()
    persona.ychat.get_message.return_value = None

    # Mock tool call manager
    client._tool_call_manager = MagicMock()
    client._tool_call_manager.get_message_id.return_value = None

    client._personas_by_session = {SESSION_ID: persona}

    return client, conn, persona


class TestPromptAndReplyContentBlocks:
    """Tests for how prompt_and_reply builds ACP content blocks."""

    async def test_text_only(self):
        """Without attachments, sends a single TextContentBlock."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(session_id=SESSION_ID, prompt="hello")

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextContentBlock)
        assert blocks[0].text == "hello"

    async def test_file_attachment_produces_resource_block(self):
        """A file attachment produces a ResourceContentBlock with file:// URI."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="check this",
            attachments=[{"value": "src/main.py", "type": "file", "mimetype": "text/x-python"}],
            root_dir="/home/user/notebooks",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert len(blocks) == 2
        assert isinstance(blocks[1], ResourceContentBlock)
        assert blocks[1].uri == Path("/home/user/notebooks/src/main.py").resolve().as_uri()
        assert blocks[1].name == "main.py"
        assert blocks[1].mime_type == "text/x-python"

    async def test_notebook_attachment_default_mime_type(self):
        """Notebook attachments get application/x-ipynb+json when mimetype is None."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="review",
            attachments=[{"value": "analysis.ipynb", "type": "notebook"}],
            root_dir="/home/user",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].mime_type == "application/x-ipynb+json"

    async def test_notebook_explicit_mimetype_preserved(self):
        """When notebook has explicit mimetype, it is preserved."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="review",
            attachments=[{"value": "nb.ipynb", "type": "notebook", "mimetype": "custom/type"}],
            root_dir="/home/user",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].mime_type == "custom/type"

    async def test_multiple_attachments_in_order(self):
        """Multiple attachments produce ResourceContentBlocks in order after text."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="review all",
            attachments=[
                {"value": "a.py", "type": "file"},
                {"value": "b.ipynb", "type": "notebook"},
            ],
            root_dir="/tmp",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert len(blocks) == 3
        assert blocks[0].text == "review all"
        assert blocks[1].name == "a.py"
        assert blocks[2].name == "b.ipynb"
        assert blocks[2].mime_type == "application/x-ipynb+json"

    async def test_none_attachments(self):
        """None attachments produces only the text block."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="hello",
            attachments=None,
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert len(blocks) == 1

    async def test_empty_list_attachments(self):
        """Empty attachment list produces only the text block."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="hello",
            attachments=[],
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert len(blocks) == 1

    async def test_empty_value_fallback_name(self):
        """When attachment value is empty, name falls back to '<attachment>'."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="check",
            attachments=[{"value": "", "type": "file"}],
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].name == "<attachment>"

    async def test_mimetype_none_for_file(self):
        """File attachment with no mimetype gets None mime_type."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="check",
            attachments=[{"value": "data.csv", "type": "file"}],
            root_dir="/tmp",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].mime_type is None

    async def test_no_root_dir_uses_relative_path(self):
        """When root_dir is None, URI is the raw relative path."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="check",
            attachments=[{"value": "subdir/file.py", "type": "file"}],
            root_dir=None,
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].uri == "subdir/file.py"

    async def test_file_uri_format(self):
        """file:// URI has correct RFC 8089 format with three slashes."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="check",
            attachments=[{"value": "test.py", "type": "file"}],
            root_dir="/home/user",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].uri.startswith("file:///")

    async def test_path_traversal_blocked(self):
        """Attachment path escaping root_dir falls back to raw relative path."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="check",
            attachments=[{"value": "../../../etc/passwd", "type": "file"}],
            root_dir="/home/user/notebooks",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].uri == "../../../etc/passwd"
