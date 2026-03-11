"""Tests for content block building in JaiAcpClient.prompt_and_reply()."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from acp.schema import ResourceContentBlock, TextContentBlock
from jupyterlab_chat.models import (
    AttachmentSelection,
    FileAttachment,
    NotebookAttachment,
    NotebookAttachmentCell,
)

from jupyter_ai_acp_client.default_acp_client import (
    JaiAcpClient,
    _build_attachment_description,
)


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
        """A FileAttachment produces a ResourceContentBlock with correct fields."""
        client, conn, _ = _make_client_and_persona()

        att = FileAttachment(value="src/main.py", mimetype="text/x-python")
        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="review",
            attachments=[att],
            root_dir="/home/user/notebooks",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert len(blocks) == 2
        assert isinstance(blocks[1], ResourceContentBlock)
        assert blocks[1].uri == Path("/home/user/notebooks/src/main.py").resolve().as_uri()
        assert blocks[1].name == "main.py"
        assert blocks[1].mime_type == "text/x-python"

    async def test_notebook_attachment_default_mime_type(self):
        """NotebookAttachment gets default application/x-ipynb+json MIME type."""
        client, conn, _ = _make_client_and_persona()

        att = NotebookAttachment(value="analysis.ipynb")
        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="check",
            attachments=[att],
            root_dir="/tmp",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].mime_type == "application/x-ipynb+json"

    async def test_file_explicit_mimetype_preserved(self):
        """Explicit mimetype on FileAttachment is preserved."""
        client, conn, _ = _make_client_and_persona()

        att = FileAttachment(value="data.csv", mimetype="text/csv")
        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="parse",
            attachments=[att],
            root_dir="/tmp",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].mime_type == "text/csv"

    async def test_multiple_attachments_in_order(self):
        """Multiple attachments produce resource blocks in order after text."""
        client, conn, _ = _make_client_and_persona()

        atts = [
            FileAttachment(value="a.py"),
            NotebookAttachment(value="b.ipynb"),
        ]
        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="review",
            attachments=atts,
            root_dir="/tmp",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert len(blocks) == 3
        assert isinstance(blocks[0], TextContentBlock)
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

        att = FileAttachment(value="")
        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="check",
            attachments=[att],
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].name == "<attachment>"

    async def test_mimetype_none_for_file(self):
        """File without explicit mimetype gets None mime_type."""
        client, conn, _ = _make_client_and_persona()

        att = FileAttachment(value="unknown.xyz")
        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="check",
            attachments=[att],
            root_dir="/tmp",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].mime_type is None

    async def test_no_root_dir_uses_relative_path(self):
        """Without root_dir, attachment value is used as-is for URI."""
        client, conn, _ = _make_client_and_persona()

        att = FileAttachment(value="src/main.py")
        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="check",
            attachments=[att],
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].uri == "src/main.py"

    async def test_file_uri_format(self):
        """With root_dir, attachment gets a file:// URI."""
        client, conn, _ = _make_client_and_persona()

        att = FileAttachment(value="main.py")
        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="check",
            attachments=[att],
            root_dir="/tmp",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].uri == Path("/tmp/main.py").as_uri()

    async def test_path_traversal_blocked(self):
        """Path traversal attempt falls back to raw value."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="check",
            attachments=[FileAttachment(value="../../../etc/passwd")],
            root_dir="/home/user/notebooks",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].uri == "../../../etc/passwd"

    async def test_file_selection_produces_description(self):
        """File with selection gets human-readable line range in description."""
        client, conn, _ = _make_client_and_persona()

        att = FileAttachment(
            value="main.py",
            selection=AttachmentSelection(start=(5, 0), end=(10, 0), content="..."),
        )
        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="check",
            attachments=[att],
            root_dir="/tmp",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].description == "Lines 6-11"

    async def test_notebook_cells_produce_description(self):
        """Notebook with cells gets cell IDs in description."""
        client, conn, _ = _make_client_and_persona()

        att = NotebookAttachment(
            value="nb.ipynb",
            cells=[
                NotebookAttachmentCell(id="abc", input_type="code"),
                NotebookAttachmentCell(id="def", input_type="markdown"),
            ],
        )
        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="check",
            attachments=[att],
            root_dir="/tmp",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].description == 'Selected cells: Cell "abc", Cell "def"'

    async def test_no_selection_no_description(self):
        """File without selection gets None description."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="check",
            attachments=[FileAttachment(value="plain.py")],
            root_dir="/tmp",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].description is None


class TestAttachmentDescription:
    """Tests for _build_attachment_description helper."""

    def test_file_no_selection(self):
        att = FileAttachment(value="main.py")
        assert _build_attachment_description(att) is None

    def test_file_single_line_selection(self):
        att = FileAttachment(
            value="main.py",
            selection=AttachmentSelection(start=(6, 0), end=(6, 15), content="x = 1"),
        )
        assert _build_attachment_description(att) == "Line 7"

    def test_file_multi_line_selection(self):
        att = FileAttachment(
            value="main.py",
            selection=AttachmentSelection(start=(5, 0), end=(10, 0), content="..."),
        )
        assert _build_attachment_description(att) == "Lines 6-11"

    def test_notebook_no_cells(self):
        att = NotebookAttachment(value="nb.ipynb")
        assert _build_attachment_description(att) is None

    def test_notebook_cells_without_selection(self):
        att = NotebookAttachment(
            value="nb.ipynb",
            cells=[
                NotebookAttachmentCell(id="abc", input_type="code"),
                NotebookAttachmentCell(id="def", input_type="markdown"),
            ],
        )
        assert _build_attachment_description(att) == 'Selected cells: Cell "abc", Cell "def"'

    def test_notebook_cell_with_single_line_selection(self):
        att = NotebookAttachment(
            value="nb.ipynb",
            cells=[
                NotebookAttachmentCell(
                    id="abc",
                    input_type="code",
                    selection=AttachmentSelection(start=(4, 0), end=(4, 10), content="x"),
                ),
            ],
        )
        assert _build_attachment_description(att) == 'Selected cells: Cell "abc" (line 5)'

    def test_notebook_cell_with_multi_line_selection(self):
        att = NotebookAttachment(
            value="nb.ipynb",
            cells=[
                NotebookAttachmentCell(
                    id="abc",
                    input_type="code",
                    selection=AttachmentSelection(start=(0, 0), end=(2, 5), content="..."),
                ),
            ],
        )
        assert _build_attachment_description(att) == 'Selected cells: Cell "abc" (lines 1-3)'

    def test_notebook_mixed_cells(self):
        """Some cells with selection, some without."""
        att = NotebookAttachment(
            value="nb.ipynb",
            cells=[
                NotebookAttachmentCell(
                    id="abc",
                    input_type="code",
                    selection=AttachmentSelection(start=(0, 0), end=(2, 0), content="..."),
                ),
                NotebookAttachmentCell(id="def", input_type="markdown"),
                NotebookAttachmentCell(
                    id="ghi",
                    input_type="code",
                    selection=AttachmentSelection(start=(4, 0), end=(7, 0), content="..."),
                ),
            ],
        )
        assert _build_attachment_description(att) == 'Selected cells: Cell "abc" (lines 1-3), Cell "def", Cell "ghi" (lines 5-8)'

    def test_notebook_empty_cells_list(self):
        att = NotebookAttachment(value="nb.ipynb", cells=[])
        assert _build_attachment_description(att) is None

    def test_zero_indexed_to_one_indexed(self):
        """Verifies 0-indexed CodeMirror positions become 1-indexed."""
        att = FileAttachment(
            value="main.py",
            selection=AttachmentSelection(start=(0, 0), end=(0, 5), content="x"),
        )
        assert _build_attachment_description(att) == "Line 1"
