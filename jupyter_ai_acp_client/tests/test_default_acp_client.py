"""Tests for content block building and session management in JaiAcpClient."""

import asyncio
import json
import logging
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from acp.exceptions import RequestError
from acp.schema import (
    AvailableCommand,
    AvailableCommandsUpdate,
    ConfigOptionUpdate,
    CurrentModeUpdate,
    ResourceContentBlock,
    TextContentBlock,
    Usage,
    UsageUpdate,
)

from jupyterlab_chat.models import (
    FileAttachment,
    NotebookAttachment,
    NotebookAttachmentCell,
)

from jupyter_ai_persona_manager.persona_events import PersonaSessionState

from jupyter_ai_acp_client.base_acp_persona import BaseAcpPersona
from jupyter_ai_acp_client.default_acp_client import JaiAcpClient


SESSION_ID = "sess-1"


def _state() -> PersonaSessionState:
    """A real PersonaSessionState with no event logger, so its typed properties
    store values in memory without emitting. The report_*/get_* methods
    round-trip through the real state object."""
    return PersonaSessionState(
        event_logger=None,
        chat_id="test-room",
        persona_id="test-persona",
        log=logging.getLogger("test"),
    )


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
    persona.state = MagicMock()
    persona.chat = MagicMock()
    persona.chat.get_message.return_value = None

    # Mock tool call manager
    client._tool_call_manager = MagicMock()

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
            attachments=[FileAttachment(value="src/main.py", mimetype="text/x-python")],
            root_dir="/home/user/notebooks",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert len(blocks) == 2
        assert isinstance(blocks[1], ResourceContentBlock)
        assert blocks[1].uri == Path("/home/user/notebooks/src/main.py").resolve().as_uri()
        assert blocks[1].name == "main.py"
        assert blocks[1].mime_type == "text/x-python"

    async def test_notebook_attachment_without_mimetype_gets_none(self):
        """A notebook attachment with no mimetype is sent without a media type,
        like a file attachment: no application/x-ipynb+json default."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="review",
            attachments=[NotebookAttachment(value="analysis.ipynb")],
            root_dir="/home/user",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert isinstance(blocks[1], ResourceContentBlock)
        assert blocks[1].mime_type is None

    @pytest.mark.parametrize("attachment_class", [NotebookAttachment, FileAttachment])
    async def test_notebook_mimetype_not_forwarded(self, attachment_class):
        """application/x-ipynb+json, which the chat frontend sets on every
        notebook-cell attachment and which agents may refuse, is dropped from
        the link whichever attachment type carries it."""
        client, conn, _ = _make_client_and_persona()
        attachment = attachment_class(value="nb.ipynb", mimetype="application/x-ipynb+json")

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="review",
            attachments=[attachment],
            root_dir=None,
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert isinstance(blocks[1], ResourceContentBlock)
        assert blocks[1].name == "nb.ipynb"
        assert blocks[1].mime_type is None

    async def test_notebook_explicit_mimetype_preserved(self):
        """When notebook has explicit mimetype, it is preserved."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="review",
            attachments=[NotebookAttachment(value="nb.ipynb", mimetype="custom/type")],
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
                FileAttachment(value="a.py"),
                NotebookAttachment(value="b.ipynb"),
            ],
            root_dir="/tmp",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert len(blocks) == 3
        assert blocks[0].text == "review all"
        assert blocks[1].name == "a.py"
        assert blocks[2].name == "b.ipynb"
        assert blocks[2].mime_type is None

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
            attachments=[FileAttachment(value="")],
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].name == "<attachment>"

    async def test_mimetype_none_for_file(self):
        """File attachment with no mimetype gets None mime_type."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="check",
            attachments=[FileAttachment(value="data.csv")],
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
            attachments=[FileAttachment(value="subdir/file.py")],
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
            attachments=[FileAttachment(value="test.py")],
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
            attachments=[FileAttachment(value="../../../etc/passwd")],
            root_dir="/home/user/notebooks",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].uri == "../../../etc/passwd"


NOTEBOOK_MIME = "application/x-ipynb+json"
"""What the chat frontend sets on every notebook-cell attachment."""


def _write_notebook(tmp_path: Path, name: str, *cells: tuple[str, str]) -> Path:
    """An nbformat 4.5 notebook of code cells, given as (id, source) pairs."""
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "code",
                "id": cell_id,
                "source": source,
                "metadata": {},
                "outputs": [],
                "execution_count": None,
            }
            for cell_id, source in cells
        ],
    }
    path = tmp_path / name
    path.write_text(json.dumps(notebook), encoding="utf-8")
    return path


def _dragged(value: str, *ids: str) -> NotebookAttachment:
    """A cell attachment as the chat frontend builds it: the notebook's
    path, the notebook media type and the dragged cells' ids."""
    cells = [NotebookAttachmentCell(id=cell_id, input_type="code") for cell_id in ids]
    return NotebookAttachment(value=value, mimetype=NOTEBOOK_MIME, cells=cells)


class TestNotebookCellAttachments:
    """How prompt_and_reply sends a notebook attachment that names cells.
    The rendering itself is tested in test_notebook_cells.py."""

    @pytest.fixture(autouse=True)
    def _without_jupyter_ai_tools(self, monkeypatch):
        """Make the live-document lookup unavailable, so the file on disk is
        the source regardless of what the environment has installed."""
        monkeypatch.setitem(sys.modules, "jupyter_ai_tools", None)

    async def test_notebook_cells_are_sent_as_text(self, tmp_path):
        """The named cell is sent as a text block after the prompt, with the
        notebook's absolute path, instead of a resource link."""
        client, conn, _ = _make_client_and_persona()
        path = _write_notebook(
            tmp_path, "analysis.ipynb", ("c1", "secret = 1\n"), ("c2", "print(secret)\n")
        )

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="explain",
            attachments=[_dragged("analysis.ipynb", "c2")],
            root_dir=str(tmp_path),
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert len(blocks) == 2
        assert blocks[0].text == "explain"
        assert isinstance(blocks[1], TextContentBlock)
        assert "Cell 2 of 2 (code, never run, id=c2):" in blocks[1].text
        assert "print(secret)" in blocks[1].text
        assert "secret = 1" not in blocks[1].text
        assert str(path.resolve()) in blocks[1].text

    async def test_notebook_cells_renderer_failure_falls_back_to_resource_link(
        self, tmp_path, monkeypatch
    ):
        """A failure inside the renderer is logged and the attachment is sent
        as a resource link without the notebook media type, instead of
        failing the prompt."""
        client, conn, persona = _make_client_and_persona()
        _write_notebook(tmp_path, "nb.ipynb", ("c1", "x = 1\n"))

        async def boom(*args, **kwargs):
            raise RuntimeError("renderer bug")

        monkeypatch.setattr("jupyter_ai_acp_client.default_acp_client.render_cell_block", boom)

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="explain",
            attachments=[_dragged("nb.ipynb", "c1")],
            root_dir=str(tmp_path),
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert len(blocks) == 2
        assert isinstance(blocks[1], ResourceContentBlock)
        assert blocks[1].uri == (tmp_path / "nb.ipynb").resolve().as_uri()
        assert blocks[1].mime_type is None
        assert "Could not render the cells" in persona.log.warning.call_args.args[0]

    async def test_stop_while_rendering_sends_no_prompt(self, tmp_path, monkeypatch):
        """Rendering awaits a live lookup and a file read. If the user stops
        the turn meanwhile, the prompt is not sent afterwards: the turn that
        stop_streaming() finalized does not run unseen."""
        client, conn, _ = _make_client_and_persona()
        _write_notebook(tmp_path, "nb.ipynb", ("c1", "x = 1\n"))
        rendering = asyncio.Event()
        release = asyncio.Event()

        async def slow_render(*args, **kwargs):
            rendering.set()
            await release.wait()
            return TextContentBlock(type="text", text="cells")

        monkeypatch.setattr(
            "jupyter_ai_acp_client.default_acp_client.render_cell_block", slow_render
        )

        task = asyncio.create_task(
            client.prompt_and_reply(
                session_id=SESSION_ID,
                prompt="explain",
                attachments=[_dragged("nb.ipynb", "c1")],
                root_dir=str(tmp_path),
            )
        )
        await asyncio.wait_for(rendering.wait(), 5)
        await client.stop_streaming(SESSION_ID)
        release.set()
        response = await asyncio.wait_for(task, 5)

        conn.cancel.assert_awaited_once_with(SESSION_ID)
        conn.prompt.assert_not_awaited()
        assert response.stop_reason == "cancelled"

    async def test_notebook_attachment_without_cells_keeps_resource_link(self, tmp_path):
        """A notebook attachment naming no cell id is still a resource link,
        without the notebook media type."""
        client, conn, _ = _make_client_and_persona()
        _write_notebook(tmp_path, "nb.ipynb", ("c1", "x = 1\n"))
        raw = {
            "value": "nb.ipynb",
            "type": "notebook",
            "mimetype": NOTEBOOK_MIME,
            "cells": [{"input_type": "code"}],
        }

        for attachment in (NotebookAttachment(value="nb.ipynb"), NotebookAttachment(**raw)):
            await client.prompt_and_reply(
                session_id=SESSION_ID,
                prompt="review",
                attachments=[attachment],
                root_dir=str(tmp_path),
            )
            blocks = conn.prompt.call_args.kwargs["prompt"]
            assert isinstance(blocks[1], ResourceContentBlock)
            assert blocks[1].uri == (tmp_path / "nb.ipynb").resolve().as_uri()
            assert blocks[1].mime_type is None

    async def test_notebook_cells_without_root_dir_keep_resource_link(self):
        """Without a root_dir the notebook cannot be located, so the
        attachment stays a resource link to the relative path, without the
        notebook media type, and the reason is logged."""
        client, conn, persona = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="explain",
            attachments=[_dragged("nb.ipynb", "c1")],
            root_dir=None,
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert isinstance(blocks[1], ResourceContentBlock)
        assert blocks[1].uri == "nb.ipynb"
        assert blocks[1].mime_type is None
        messages = [call.args[0] for call in persona.log.debug.call_args_list]
        assert any("did not resolve under root_dir" in message for message in messages)

    async def test_notebook_cells_do_not_bypass_root_dir_guard(self, tmp_path):
        """A cell attachment whose path escapes root_dir is not read; it falls
        back to the raw path like any other attachment (a regression guard:
        this behaviour predates cell rendering)."""
        client, conn, persona = _make_client_and_persona()
        _write_notebook(tmp_path, "outside.ipynb", ("c1", "secret = 1\n"))
        root = tmp_path / "root"
        root.mkdir()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="explain",
            attachments=[_dragged("../outside.ipynb", "c1")],
            root_dir=str(root),
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert isinstance(blocks[1], ResourceContentBlock)
        assert blocks[1].uri == "../outside.ipynb"
        assert blocks[1].mime_type is None
        assert persona.log.warning.called
        assert not any(isinstance(b, TextContentBlock) for b in blocks[1:])


def _real_usage_persona():
    """
    A `BaseAcpPersona` created without `__init__` (no subprocess or session),
    carrying the real usage setters and properties so tests cover the actual
    store-then-read round trip. Collaborators the client touches are mocked.
    """

    class _ConcreteAcpPersona(BaseAcpPersona):
        @property
        def defaults(self):  # pragma: no cover - never called in these tests
            return None

    persona = _ConcreteAcpPersona.__new__(_ConcreteAcpPersona)
    persona._acp_context_usage = None
    persona._acp_session_usage = None
    persona.log = logging.getLogger("test")
    # A real state slot so `_sync_awareness_usage` -> `report_usage`
    # round-trips through the real typed properties.
    persona.state = _state()
    persona.chat = MagicMock()
    # `set_status()` (called incidentally by `prompt_and_reply`) builds
    # `as_user()`, which reads `self.defaults`; this persona has none, so mock it.
    persona.as_user = MagicMock()
    return persona


class TestUsageStorage:
    """A usage report received by the client ends up readable on the persona."""

    async def test_usage_update_is_stored_as_context_usage(self):
        client, _, _ = _make_client_and_persona()
        client._loading_sessions = {}
        persona = _real_usage_persona()
        client._personas_by_session[SESSION_ID] = persona
        update = UsageUpdate(sessionUpdate="usage_update", used=41_000, size=200_000)

        await client.session_update(SESSION_ID, update)

        assert persona.acp_context_usage is update

    async def test_prompt_response_usage_is_stored_as_session_usage(self):
        client, conn, _ = _make_client_and_persona()
        persona = _real_usage_persona()
        client._personas_by_session[SESSION_ID] = persona
        usage = Usage(inputTokens=900, outputTokens=340, totalTokens=1_240)
        conn.prompt = AsyncMock(return_value=MagicMock(usage=usage))

        await client.prompt_and_reply(session_id=SESSION_ID, prompt="hello")

        assert persona.acp_session_usage is usage

    async def test_prompt_response_without_usage_stores_nothing(self):
        client, conn, _ = _make_client_and_persona()
        persona = _real_usage_persona()
        client._personas_by_session[SESSION_ID] = persona
        conn.prompt = AsyncMock(return_value=MagicMock(usage=None))

        await client.prompt_and_reply(session_id=SESSION_ID, prompt="hello")

        assert persona.acp_session_usage is None


class TestExtNotification:
    """The generic client is agent-agnostic: every ext notification (including
    vendor `kiro.dev/*` methods, now handled only by KiroAcpClient) is unknown
    to it and rejected as JSON-RPC method-not-found."""

    async def test_ext_notification_raises_method_not_found(self):
        client, _, _ = _make_client_and_persona()

        for method in ("kiro.dev/metadata", "kiro.dev/commands/available", "other.vendor/thing"):
            with pytest.raises(RequestError) as exc_info:
                await client.ext_notification(method, {"sessionId": SESSION_ID})
            assert exc_info.value.code == -32601, method

    async def test_ext_method_raises_method_not_found(self):
        client, _, _ = _make_client_and_persona()

        with pytest.raises(RequestError) as exc_info:
            await client.ext_method("kiro.dev/metadata", {"sessionId": SESSION_ID})
        assert exc_info.value.code == -32601


class TestAwarenessPush:
    """The client pushes ACP updates onto the persona's awareness API too."""

    async def test_available_commands_update_advertises_over_awareness(self):
        client, _, persona = _make_client_and_persona()
        client._loading_sessions = {}
        persona.report_slash_commands = MagicMock()
        update = AvailableCommandsUpdate(
            sessionUpdate="available_commands_update",
            availableCommands=[
                AvailableCommand(name="compact", description="Compact context"),
                AvailableCommand(name="/clear", description="Clear"),
            ],
        )

        await client.session_update(SESSION_ID, update)

        commands = persona.report_slash_commands.call_args[0][0]
        # Names are leading-slash normalized.
        assert [(c.name, c.description) for c in commands] == [
            ("/compact", "Compact context"),
            ("/clear", "Clear"),
        ]

    async def test_current_mode_update_rebuilds_awareness_config(self):
        client, _, persona = _make_client_and_persona()
        client._loading_sessions = {}
        persona._sync_awareness_config = MagicMock()
        update = CurrentModeUpdate(sessionUpdate="current_mode_update", currentModeId="code")

        await client.session_update(SESSION_ID, update)

        persona.update_acp_current_mode.assert_called_once_with("code")
        persona._sync_awareness_config.assert_called_once()

    async def test_config_option_update_rebuilds_awareness_config(self):
        client, _, persona = _make_client_and_persona()
        client._loading_sessions = {}
        persona._sync_awareness_config = MagicMock()
        update = ConfigOptionUpdate(
            sessionUpdate="config_option_update", configOptions=[]
        )

        await client.session_update(SESSION_ID, update)

        persona.update_acp_config_options.assert_called_once()
        persona._sync_awareness_config.assert_called_once()

    async def test_usage_update_pushes_awareness_usage(self):
        client, _, persona = _make_client_and_persona()
        client._loading_sessions = {}
        persona._sync_awareness_usage = MagicMock()
        update = UsageUpdate(sessionUpdate="usage_update", used=1, size=2)

        await client.session_update(SESSION_ID, update)

        persona.update_acp_context_usage.assert_called_once_with(update)
        persona._sync_awareness_usage.assert_called_once()


class TestLoadSessionCleanup:
    """Tests for _loading_sessions cleanup on failure."""

    async def test_failed_load_session_removes_task_from_loading_sessions(self):
        """A failed load_session cleans up its task so retries can start fresh."""
        client = object.__new__(JaiAcpClient)
        client.event_loop = asyncio.get_running_loop()
        client._loading_sessions = {}

        persona = MagicMock()
        error = RequestError(-32002, "Resource not found")

        async def _failing_rpc(*args, **kwargs):
            raise error

        client._load_session_rpc = _failing_rpc

        with pytest.raises(RequestError):
            await client.load_session(persona, "stale-session-id")

        assert "stale-session-id" not in client._loading_sessions


class TestWriteTextFileNotebookGuard:
    """write_text_file refuses direct writes to .ipynb files (notebooks must be
    edited via the notebook MCP tools), while other files write normally."""

    async def test_write_to_notebook_is_rejected(self, tmp_path):
        client, _, _ = _make_client_and_persona()
        nb_path = str(tmp_path / "analysis.ipynb")

        with pytest.raises(RequestError) as exc_info:
            await client.write_text_file(
                content='{"cells": []}', path=nb_path, session_id=SESSION_ID
            )

        assert exc_info.value.code == -32602  # invalid_params
        assert "MCP tools" in str(exc_info.value.data)
        # The notebook file must not have been created.
        assert not (tmp_path / "analysis.ipynb").exists()

    async def test_write_to_non_notebook_still_succeeds(self, tmp_path):
        client, _, _ = _make_client_and_persona()
        txt_path = tmp_path / "notes.txt"

        result = await client.write_text_file(
            content="hello", path=str(txt_path), session_id=SESSION_ID
        )

        assert result is not None
        assert txt_path.read_text(encoding="utf-8") == "hello"
