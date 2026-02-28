from dataclasses import asdict as _asdict
from unittest.mock import AsyncMock, MagicMock

from jupyterlab_chat.models import (
    AttachmentSelection,
    FileAttachment,
    NotebookAttachment,
    NotebookAttachmentCell,
)

from jupyter_ai_acp_client.base_acp_persona import BaseAcpPersona


def make_persona(attachments_map: dict | None = None):
    """
    Build a mock object that satisfies BaseAcpPersona.process_message()'s
    attribute accesses. Uses unbound method call: BaseAcpPersona.process_message(persona, msg).

    get_attachments() is mocked to return plain dicts (via dataclasses.asdict),
    matching what ychat.get_attachments() actually returns at runtime: pycrdt's
    to_py() deserializes Yjs maps as plain Python dicts, not typed dataclasses.
    """
    mock_client = MagicMock()
    mock_client.prompt_and_reply = AsyncMock(return_value=MagicMock())

    persona = MagicMock()
    persona.as_user.return_value = MagicMock(mention_name="claude")
    persona.get_client = AsyncMock(return_value=mock_client)
    persona.get_session_id = AsyncMock(return_value="session-1")
    raw_map = {k: _asdict(v) for k, v in (attachments_map or {}).items()}
    persona.ychat.get_attachments.return_value = raw_map

    return persona, mock_client


def make_message(body: str, attachment_ids: list[str] | None = None):
    msg = MagicMock()
    msg.body = body
    msg.attachments = attachment_ids
    return msg


class TestProcessMessageAttachments:
    async def test_no_attachments_calls_prompt_with_none(self):
        persona, client = make_persona()
        msg = make_message("@claude hello")

        await BaseAcpPersona.process_message(persona, msg)

        client.prompt_and_reply.assert_called_once_with(
            session_id="session-1",
            prompt="hello",
            attachments=None,
        )

    async def test_attachment_ids_resolved_to_typed_objects(self):
        fa = FileAttachment(value="code.py", type="file")
        persona, client = make_persona(attachments_map={"att-1": fa})
        msg = make_message("@claude review this", attachment_ids=["att-1"])

        await BaseAcpPersona.process_message(persona, msg)

        client.prompt_and_reply.assert_called_once_with(
            session_id="session-1",
            prompt="review this",
            attachments=[fa],
        )

    async def test_multiple_ids_resolved_in_order(self):
        fa = FileAttachment(value="a.py", type="file")
        nb = NotebookAttachment(value="b.ipynb", type="notebook")
        persona, client = make_persona(attachments_map={"att-1": fa, "att-2": nb})
        msg = make_message("@claude check both", attachment_ids=["att-1", "att-2"])

        await BaseAcpPersona.process_message(persona, msg)

        args = client.prompt_and_reply.call_args
        assert args.kwargs["attachments"] == [fa, nb]

    async def test_unknown_attachment_id_filtered_out(self):
        persona, client = make_persona(attachments_map={})
        msg = make_message("@claude hi", attachment_ids=["att-missing"])

        await BaseAcpPersona.process_message(persona, msg)

        client.prompt_and_reply.assert_called_once_with(
            session_id="session-1",
            prompt="hi",
            attachments=None,
        )

    async def test_empty_attachment_list_passes_none(self):
        persona, client = make_persona()
        msg = make_message("@claude hi", attachment_ids=[])

        await BaseAcpPersona.process_message(persona, msg)

        client.prompt_and_reply.assert_called_once_with(
            session_id="session-1",
            prompt="hi",
            attachments=None,
        )

    async def test_partial_ids_only_resolved_ids_forwarded(self):
        fa = FileAttachment(value="found.py", type="file")
        persona, client = make_persona(attachments_map={"att-good": fa})
        msg = make_message("@claude check", attachment_ids=["att-good", "att-bad"])

        await BaseAcpPersona.process_message(persona, msg)

        args = client.prompt_and_reply.call_args
        assert args.kwargs["attachments"] == [fa]

    async def test_schema_drift_in_nested_field_skips_attachment(self):
        """H-1: Schema drift in a nested field should skip the attachment, not crash.

        When a nested selection dict contains an unexpected key, AttachmentSelection(**raw)
        raises TypeError. _reconstruct_attachment catches this and returns None, so the
        attachment is skipped rather than crashing process_message().
        """
        persona, client = make_persona()
        # Simulate a future schema where AttachmentSelection gained a new required field
        persona.ychat.get_attachments.return_value = {
            "att-bad": {
                "value": "x.py",
                "type": "file",
                "selection": {
                    "start": [0, 0],
                    "end": [1, 10],
                    "content": "x = 1",
                    "unknown_future_key": True,  # schema drift in nested type
                },
            }
        }
        msg = make_message("@claude hi", attachment_ids=["att-bad"])

        await BaseAcpPersona.process_message(persona, msg)

        # Attachment skipped due to TypeError in nested reconstruction
        client.prompt_and_reply.assert_called_once_with(
            session_id="session-1",
            prompt="hi",
            attachments=None,
        )

    async def test_file_attachment_selection_reconstructed_as_typed_object(self):
        """M-1: Nested selection field should be AttachmentSelection, not a plain dict."""
        selection = AttachmentSelection(start=(0, 0), end=(2, 10), content="def foo():\n    pass")
        fa = FileAttachment(value="code.py", type="file", selection=selection)
        persona, client = make_persona(attachments_map={"att-1": fa})
        msg = make_message("@claude review this", attachment_ids=["att-1"])

        await BaseAcpPersona.process_message(persona, msg)

        args = client.prompt_and_reply.call_args
        resolved = args.kwargs["attachments"][0]
        assert isinstance(resolved, FileAttachment)
        assert isinstance(resolved.selection, AttachmentSelection), (
            f"Expected AttachmentSelection, got {type(resolved.selection)}"
        )
        assert resolved.selection == selection

    async def test_notebook_attachment_cells_reconstructed_as_typed_objects(self):
        """M-1: Nested cells list should contain NotebookAttachmentCell instances, not dicts."""
        cell_selection = AttachmentSelection(start=(1, 0), end=(3, 5), content="x = 1")
        cell = NotebookAttachmentCell(
            id="cell-abc",
            input_type="code",
            selection=cell_selection,
        )
        nb = NotebookAttachment(value="analysis.ipynb", type="notebook", cells=[cell])
        persona, client = make_persona(attachments_map={"att-nb": nb})
        msg = make_message("@claude check notebook", attachment_ids=["att-nb"])

        await BaseAcpPersona.process_message(persona, msg)

        args = client.prompt_and_reply.call_args
        resolved = args.kwargs["attachments"][0]
        assert isinstance(resolved, NotebookAttachment)
        assert resolved.cells is not None
        assert len(resolved.cells) == 1
        reconstructed_cell = resolved.cells[0]
        assert isinstance(reconstructed_cell, NotebookAttachmentCell), (
            f"Expected NotebookAttachmentCell, got {type(reconstructed_cell)}"
        )
        assert reconstructed_cell.id == "cell-abc"
        assert reconstructed_cell.input_type == "code"
        assert isinstance(reconstructed_cell.selection, AttachmentSelection), (
            f"Expected AttachmentSelection in cell.selection, got {type(reconstructed_cell.selection)}"
        )
        assert reconstructed_cell.selection == cell_selection
