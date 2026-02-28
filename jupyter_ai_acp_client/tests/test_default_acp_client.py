import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from acp.schema import ResourceContentBlock, TextContentBlock
from jupyterlab_chat.models import FileAttachment, NotebookAttachment

from jupyter_ai_acp_client.default_acp_client import JaiAcpClient

SESSION_ID = "session-1"


def make_client():
    """Build a minimal JaiAcpClient with mocked internals, bypassing __init__."""
    mock_persona = MagicMock()
    mock_persona.ychat.get_message.return_value = None
    mock_persona.log = MagicMock()
    mock_persona.awareness = MagicMock()

    client = object.__new__(JaiAcpClient)
    client.event_loop = asyncio.get_running_loop()
    client._personas_by_session = {SESSION_ID: mock_persona}
    client._prompt_locks_by_session = {}
    client._terminal_manager = MagicMock()
    client._tool_call_manager = MagicMock()
    client._tool_call_manager.get_message_id.return_value = None
    return client


class TestPromptAndReplyContentBlocks:
    async def test_no_attachments_sends_only_text_block(self):
        mock_conn = AsyncMock()
        mock_conn.prompt = AsyncMock(return_value=MagicMock())
        client = make_client()

        with patch.object(client, "get_connection", AsyncMock(return_value=mock_conn)):
            await client.prompt_and_reply(SESSION_ID, "hello")

        prompt_blocks = mock_conn.prompt.call_args.kwargs["prompt"]
        assert len(prompt_blocks) == 1
        assert isinstance(prompt_blocks[0], TextContentBlock)
        assert prompt_blocks[0].text == "hello"

    async def test_file_attachment_appends_resource_block(self):
        mock_conn = AsyncMock()
        mock_conn.prompt = AsyncMock(return_value=MagicMock())
        client = make_client()
        attachment = FileAttachment(value="notebooks/analysis.py", type="file")

        with patch.object(client, "get_connection", AsyncMock(return_value=mock_conn)):
            await client.prompt_and_reply(
                SESSION_ID, "analyze this", attachments=[attachment]
            )

        prompt_blocks = mock_conn.prompt.call_args.kwargs["prompt"]
        assert len(prompt_blocks) == 2
        assert isinstance(prompt_blocks[0], TextContentBlock)
        assert isinstance(prompt_blocks[1], ResourceContentBlock)
        assert prompt_blocks[1].uri == "notebooks/analysis.py"
        assert prompt_blocks[1].name == "analysis.py"

    async def test_notebook_attachment_appends_resource_block(self):
        mock_conn = AsyncMock()
        mock_conn.prompt = AsyncMock(return_value=MagicMock())
        client = make_client()
        attachment = NotebookAttachment(value="work/analysis.ipynb", type="notebook")

        with patch.object(client, "get_connection", AsyncMock(return_value=mock_conn)):
            await client.prompt_and_reply(
                SESSION_ID, "review notebook", attachments=[attachment]
            )

        prompt_blocks = mock_conn.prompt.call_args.kwargs["prompt"]
        assert len(prompt_blocks) == 2
        assert isinstance(prompt_blocks[1], ResourceContentBlock)
        assert prompt_blocks[1].uri == "work/analysis.ipynb"
        assert prompt_blocks[1].name == "analysis.ipynb"

    async def test_multiple_attachments_appended_in_order(self):
        mock_conn = AsyncMock()
        mock_conn.prompt = AsyncMock(return_value=MagicMock())
        client = make_client()
        attachments = [
            FileAttachment(value="a.py", type="file"),
            NotebookAttachment(value="b.ipynb", type="notebook"),
        ]

        with patch.object(client, "get_connection", AsyncMock(return_value=mock_conn)):
            await client.prompt_and_reply(SESSION_ID, "review", attachments=attachments)

        prompt_blocks = mock_conn.prompt.call_args.kwargs["prompt"]
        assert len(prompt_blocks) == 3
        assert prompt_blocks[1].uri == "a.py"
        assert prompt_blocks[1].name == "a.py"
        assert prompt_blocks[2].uri == "b.ipynb"
        assert prompt_blocks[2].name == "b.ipynb"

    async def test_empty_attachments_list_sends_only_text(self):
        mock_conn = AsyncMock()
        mock_conn.prompt = AsyncMock(return_value=MagicMock())
        client = make_client()

        with patch.object(client, "get_connection", AsyncMock(return_value=mock_conn)):
            await client.prompt_and_reply(SESSION_ID, "hello", attachments=[])

        prompt_blocks = mock_conn.prompt.call_args.kwargs["prompt"]
        assert len(prompt_blocks) == 1
        assert isinstance(prompt_blocks[0], TextContentBlock)

    async def test_none_attachments_sends_only_text(self):
        mock_conn = AsyncMock()
        mock_conn.prompt = AsyncMock(return_value=MagicMock())
        client = make_client()

        with patch.object(client, "get_connection", AsyncMock(return_value=mock_conn)):
            await client.prompt_and_reply(SESSION_ID, "hello", attachments=None)

        prompt_blocks = mock_conn.prompt.call_args.kwargs["prompt"]
        assert len(prompt_blocks) == 1

    async def test_empty_attachment_value_uses_fallback_name(self):
        """M-1: When attachment.value is empty, name falls back to '<attachment>'."""
        mock_conn = AsyncMock()
        mock_conn.prompt = AsyncMock(return_value=MagicMock())
        client = make_client()
        attachment = FileAttachment(value="", type="file")

        with patch.object(client, "get_connection", AsyncMock(return_value=mock_conn)):
            await client.prompt_and_reply(
                SESSION_ID, "attach this", attachments=[attachment]
            )

        prompt_blocks = mock_conn.prompt.call_args.kwargs["prompt"]
        resource_block = prompt_blocks[1]
        assert isinstance(resource_block, ResourceContentBlock)
        assert resource_block.name == "<attachment>"

    async def test_file_attachment_mimetype_forwarded(self):
        """M-2: FileAttachment.mimetype is forwarded to ResourceContentBlock.mime_type."""
        mock_conn = AsyncMock()
        mock_conn.prompt = AsyncMock(return_value=MagicMock())
        client = make_client()
        attachment = FileAttachment(value="script.py", type="file", mimetype="text/x-python")

        with patch.object(client, "get_connection", AsyncMock(return_value=mock_conn)):
            await client.prompt_and_reply(
                SESSION_ID, "review", attachments=[attachment]
            )

        prompt_blocks = mock_conn.prompt.call_args.kwargs["prompt"]
        resource_block = prompt_blocks[1]
        assert isinstance(resource_block, ResourceContentBlock)
        assert resource_block.mime_type == "text/x-python"

    async def test_file_attachment_none_mimetype_forwarded(self):
        """M-2: When FileAttachment.mimetype is None, ResourceContentBlock.mime_type is also None."""
        mock_conn = AsyncMock()
        mock_conn.prompt = AsyncMock(return_value=MagicMock())
        client = make_client()
        attachment = FileAttachment(value="data.csv", type="file", mimetype=None)

        with patch.object(client, "get_connection", AsyncMock(return_value=mock_conn)):
            await client.prompt_and_reply(
                SESSION_ID, "analyze", attachments=[attachment]
            )

        prompt_blocks = mock_conn.prompt.call_args.kwargs["prompt"]
        resource_block = prompt_blocks[1]
        assert isinstance(resource_block, ResourceContentBlock)
        assert resource_block.mime_type is None
