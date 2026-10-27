"""Tests for the auto_emit_event decorator."""

import pytest

from jupyter_ai_acp_client.telemetry import auto_emit_event


class _RecordingLogger:
    """Minimal stand-in for an EventLogger that records emitted events."""

    def __init__(self):
        self.events: list[dict] = []

    def emit(self, schema_id, data):  # noqa: ARG002 - signature parity
        self.events.append(data)


class _Persona:
    """Minimal object exposing the attributes auto_emit_event expects."""

    def __init__(self):
        self.event_logger = _RecordingLogger()


class _BenignError(Exception):
    """Expected, non-error control-flow signal (e.g. auth pending)."""


class _RealError(Exception):
    """A genuine operational failure."""


@pytest.mark.asyncio
async def test_success_emits_success_event():
    persona = _Persona()

    @auto_emit_event("acp_server_init")
    async def _op(self):
        return "ok"

    assert await _op(persona) == "ok"
    assert persona.event_logger.events == [
        {"operation": "acp_server_init", "outcome": "success",
         "details": {"persona_class": "_Persona"}}
    ]


@pytest.mark.asyncio
async def test_benign_exception_reraises_without_failure_event():
    """A benign exception must propagate but NOT be recorded as a failure.

    Regression test: PersonaNotAuthenticated (auth pending) was being
    mislabeled as an acp_server_init failure, producing false-positive
    ServerInit alarms.
    """
    persona = _Persona()

    @auto_emit_event("acp_server_init", benign=(_BenignError,))
    async def _op(self):
        raise _BenignError("auth pending")

    with pytest.raises(_BenignError):
        await _op(persona)

    assert persona.event_logger.events == []


@pytest.mark.asyncio
async def test_non_benign_exception_emits_failure_event():
    persona = _Persona()

    @auto_emit_event("acp_server_init", benign=(_BenignError,))
    async def _op(self):
        raise _RealError("boom")

    with pytest.raises(_RealError):
        await _op(persona)

    assert len(persona.event_logger.events) == 1
    event = persona.event_logger.events[0]
    assert event["operation"] == "acp_server_init"
    assert event["outcome"] == "failure"
    assert event["details"]["error_message"] == "_RealError: boom"


@pytest.mark.asyncio
async def test_default_benign_is_empty_all_exceptions_fail():
    """Without an explicit benign tuple, every exception is a failure."""
    persona = _Persona()

    @auto_emit_event("acp_server_init")
    async def _op(self):
        raise _BenignError("still a failure here")

    with pytest.raises(_BenignError):
        await _op(persona)

    assert len(persona.event_logger.events) == 1
    assert persona.event_logger.events[0]["outcome"] == "failure"
