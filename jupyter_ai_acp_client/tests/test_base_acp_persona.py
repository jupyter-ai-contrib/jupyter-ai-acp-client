"""Tests for attachment resolution in BaseAcpPersona.process_message()."""

from unittest.mock import AsyncMock, MagicMock

from jupyterlab_chat.models import (
    AttachmentSelection,
    FileAttachment,
    NotebookAttachment,
    NotebookAttachmentCell,
)

from jupyter_ai_acp_client.base_acp_persona import (
    BaseAcpPersona,
    _deserialize_attachment,
)


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
        """A single known attachment ID resolves to a FileAttachment."""
        client = _make_client()
        att_map = {
            "att-1": {"value": "file.py", "type": "file", "mimetype": "text/x-python"},
        }
        persona = _make_persona(att_map)
        persona.get_client.return_value = client
        msg = _make_message("@bot check this", ["att-1"])

        await BaseAcpPersona.process_message(persona, msg)

        call_kwargs = client.prompt_and_reply.call_args.kwargs
        atts = call_kwargs["attachments"]
        assert len(atts) == 1
        assert isinstance(atts[0], FileAttachment)
        assert atts[0].value == "file.py"
        assert atts[0].mimetype == "text/x-python"

    async def test_multiple_attachments(self):
        """Multiple attachment IDs all resolve in order as typed dataclasses."""
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
        atts = call_kwargs["attachments"]
        assert len(atts) == 2
        assert isinstance(atts[0], FileAttachment)
        assert isinstance(atts[1], NotebookAttachment)
        assert atts[0].value == "a.py"
        assert atts[1].value == "b.ipynb"

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
        atts = call_kwargs["attachments"]
        assert len(atts) == 1
        assert isinstance(atts[0], FileAttachment)
        assert atts[0].value == "good.py"

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

    async def test_unknown_type_skipped(self):
        """Attachments with unrecognized type are skipped with a warning."""
        client = _make_client()
        att_map = {"att-1": {"value": "x", "type": "unknown_type"}}
        persona = _make_persona(att_map)
        persona.get_client.return_value = client
        msg = _make_message("@bot check", ["att-1"])

        await BaseAcpPersona.process_message(persona, msg)

        call_kwargs = client.prompt_and_reply.call_args.kwargs
        assert call_kwargs["attachments"] is None
        persona.log.warning.assert_called()


class TestDeserializeAttachment:
    """Tests for _deserialize_attachment helper."""

    def test_file_attachment(self):
        raw = {"value": "main.py", "type": "file", "mimetype": "text/x-python"}
        result = _deserialize_attachment(raw)
        assert isinstance(result, FileAttachment)
        assert result.value == "main.py"
        assert result.mimetype == "text/x-python"

    def test_notebook_attachment(self):
        raw = {"value": "analysis.ipynb", "type": "notebook"}
        result = _deserialize_attachment(raw)
        assert isinstance(result, NotebookAttachment)
        assert result.value == "analysis.ipynb"

    def test_unknown_type_returns_none(self):
        raw = {"value": "x", "type": "unknown"}
        assert _deserialize_attachment(raw) is None

    def test_missing_type_returns_none(self):
        raw = {"value": "x"}
        assert _deserialize_attachment(raw) is None

    def test_unknown_keys_filtered(self):
        """Forward-compat: unknown keys from future schema versions are dropped."""
        raw = {"value": "x.py", "type": "file", "future_field": "ignored"}
        result = _deserialize_attachment(raw)
        assert isinstance(result, FileAttachment)
        assert not hasattr(result, "future_field")

    def test_file_with_selection(self):
        """AttachmentSelection is reconstructed with tuples from CRDT lists."""
        raw = {
            "value": "main.py",
            "type": "file",
            "selection": {"start": [5, 0], "end": [10, 0], "content": "hello"},
        }
        result = _deserialize_attachment(raw)
        assert isinstance(result, FileAttachment)
        assert isinstance(result.selection, AttachmentSelection)
        assert result.selection.start == (5, 0)
        assert result.selection.end == (10, 0)
        assert result.selection.content == "hello"

    def test_selection_with_native_tuples(self):
        """Tuples that survived without CRDT round-trip are preserved."""
        raw = {
            "value": "main.py",
            "type": "file",
            "selection": {"start": (2, 3), "end": (4, 5), "content": "x"},
        }
        result = _deserialize_attachment(raw)
        assert result.selection.start == (2, 3)

    def test_notebook_with_cells(self):
        raw = {
            "value": "nb.ipynb",
            "type": "notebook",
            "cells": [
                {"id": "cell-1", "input_type": "code"},
                {"id": "cell-2", "input_type": "markdown"},
            ],
        }
        result = _deserialize_attachment(raw)
        assert isinstance(result, NotebookAttachment)
        assert len(result.cells) == 2
        assert isinstance(result.cells[0], NotebookAttachmentCell)
        assert result.cells[0].id == "cell-1"
        assert result.cells[1].input_type == "markdown"

    def test_cell_with_selection(self):
        raw = {
            "value": "nb.ipynb",
            "type": "notebook",
            "cells": [
                {
                    "id": "c1",
                    "input_type": "code",
                    "selection": {"start": [1, 0], "end": [3, 5], "content": "x = 1"},
                },
            ],
        }
        result = _deserialize_attachment(raw)
        cell = result.cells[0]
        assert isinstance(cell.selection, AttachmentSelection)
        assert cell.selection.start == (1, 0)
        assert cell.selection.content == "x = 1"

    def test_malformed_selection_becomes_none(self):
        """Selection missing required fields is set to None."""
        raw = {
            "value": "x.py",
            "type": "file",
            "selection": {"start": [1, 0]},  # missing end and content
        }
        result = _deserialize_attachment(raw)
        assert isinstance(result, FileAttachment)
        assert result.selection is None

    def test_missing_required_field_returns_none(self):
        """Missing required 'value' field causes graceful None return."""
        raw = {"type": "file"}  # missing 'value'
        assert _deserialize_attachment(raw) is None
