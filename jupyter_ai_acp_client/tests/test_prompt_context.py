"""Prompt-only extension context, including isolation and failure behavior."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from jupyterlab_chat.models import Message

from jupyter_ai_acp_client import prompt_context
from jupyter_ai_acp_client.base_acp_persona import BaseAcpPersona
from .test_base_acp_persona import _make_persona


def message(body="Question", metadata=None, **kwargs):
    return Message(
        id="request", body=body, sender="user", time=0, metadata=metadata, **kwargs
    )


async def test_sync_async_providers_receive_independent_snapshots(monkeypatch):
    original = message(metadata={"project": "example"})
    history = [original, message(deleted=True)]

    def first(current, previous):
        assert len(previous) == 1
        current.body = "mutated"
        previous[0].metadata["project"] = "mutated"
        return "First context"

    async def second(current, previous):
        assert current.body == "Question"
        assert previous[0].metadata == {"project": "example"}
        return "Second context"

    monkeypatch.setattr(
        prompt_context,
        "get_prompt_context_providers",
        lambda: {
            "first": first,
            "second": second,
            "unrelated": lambda m, h: None,
        },
    )
    assert (
        await prompt_context.get_prompt_context(original, history)
        == "First context\n\nSecond context"
    )
    assert original.body == "Question"
    assert original.metadata == {"project": "example"}


@pytest.mark.parametrize("result", [None, "", "   "])
async def test_unrelated_context_is_omitted(monkeypatch, result):
    monkeypatch.setattr(
        prompt_context,
        "get_prompt_context_providers",
        lambda: {
            "unrelated": lambda m, h: result,
        },
    )
    assert await prompt_context.get_prompt_context(message(), []) == ""


@pytest.mark.parametrize("broken", [lambda m, h: 42, lambda m, h: 1 / 0])
async def test_provider_errors_abort_prompt(monkeypatch, broken):
    monkeypatch.setattr(
        prompt_context, "get_prompt_context_providers", lambda: {"broken": broken}
    )
    persona = _make_persona()
    with pytest.raises(ValueError, match="provider 'broken' failed"):
        await BaseAcpPersona.process_message(persona, message())
    persona.get_client.assert_not_awaited()


@pytest.mark.parametrize("recover", [False, True])
async def test_context_reaches_agent_on_each_turn_without_changing_chat(
    monkeypatch, recover
):
    provider = MagicMock(return_value="Project context")
    monkeypatch.setattr(
        prompt_context, "get_prompt_context_providers", lambda: {"example": provider}
    )
    persona = _make_persona()
    persona._pending_session_recovery_context = recover
    persona._build_history_context.return_value = "Earlier conversation"
    client = persona.get_client.return_value
    client.prompt_and_reply = AsyncMock()
    current = message(metadata={"example": {"project": "one"}})
    for _ in range(2):
        await BaseAcpPersona.process_message(persona, current)
        sent = client.prompt_and_reply.call_args.kwargs["prompt"]
        assert sent.count("Project context") == 1
        assert sent.endswith("Question\n\nProject context")
    assert current.body == "Question"
    assert current.metadata == {"example": {"project": "one"}}
    assert provider.call_count == 2
    persona.chat.add_message.assert_not_called()


async def test_authentication_resume_also_receives_context(monkeypatch):
    provider = MagicMock(return_value="Restored project")
    monkeypatch.setattr(
        prompt_context, "get_prompt_context_providers", lambda: {"example": provider}
    )
    persona = _make_persona()
    persona._build_history_context.return_value = "Original question"
    client = AsyncMock()
    await BaseAcpPersona._resume_after_auth(persona, client, "session")
    assert client.prompt_and_reply.call_args.kwargs["prompt"].endswith(
        "Restored project"
    )
    assert provider.call_args.args[0] is None


def test_entry_point_order_and_validation(monkeypatch):
    a, b = MagicMock(), MagicMock()
    a.name, b.name = "a", "b"
    a.load.return_value, b.load.return_value = lambda m, h: "A", lambda m, h: "B"
    monkeypatch.setattr(prompt_context, "entry_points", lambda **kw: [b, a])
    prompt_context.get_prompt_context_providers.cache_clear()
    try:
        assert list(prompt_context.get_prompt_context_providers()) == ["a", "b"]
        a.load.assert_called_once()
        prompt_context.get_prompt_context_providers()
        a.load.assert_called_once()
        prompt_context.get_prompt_context_providers.cache_clear()
        b.name = "a"
        with pytest.raises(ValueError, match="Duplicate"):
            prompt_context.get_prompt_context_providers()
    finally:
        prompt_context.get_prompt_context_providers.cache_clear()
