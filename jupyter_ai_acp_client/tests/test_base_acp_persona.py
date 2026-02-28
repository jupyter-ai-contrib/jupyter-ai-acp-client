from dataclasses import asdict as _asdict
from unittest.mock import AsyncMock, MagicMock

from jupyterlab_chat.models import FileAttachment, NotebookAttachment

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
