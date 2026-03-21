"""
Telemetry module for the Jupyter AI ACP Client extension.

Provides a single unified event schema and a generic emit helper for
observability via Jupyter Server's built-in EventLogger system. All
telemetry is purely operational metadata — no customer content or PII
is emitted.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jupyter_events import EventLogger

logger = logging.getLogger(__name__)

# Single schema ID for all ACP telemetry events
SCHEMA_ID = (
    "https://jupyter-ai/jupyter_ai_acp_client/events/acp_telemetry"
)

ACP_TELEMETRY_SCHEMA: dict = {
    "$id": SCHEMA_ID,
    "version": "1",
    "title": "ACP Telemetry Event",
    "description": (
        "Unified telemetry event for the ACP client extension. "
        "No customer content or PII is recorded."
    ),
    "type": "object",
    "properties": {
        "operation": {
            "type": "string",
        },
        "outcome": {
            "type": "string",
            "enum": ["success", "failure"],
        },
        "details": {
            "type": "object",
            "additionalProperties": {
                "type": "string",
            },
        },
    },
    "required": ["operation", "outcome"],
    "additionalProperties": False,
}


def register_telemetry_schemas(event_logger: EventLogger) -> None:
    """Register the telemetry event schema with the EventLogger.

    Safe to call — logs errors but does not raise.
    """
    try:
        event_logger.register_event_schema(ACP_TELEMETRY_SCHEMA)
    except Exception:
        logger.error("Failed to register telemetry event schema.", exc_info=True)


def emit_event(
    event_logger: EventLogger | None,
    operation: str,
    outcome: str,
    details: dict[str, str] | None = None,
) -> None:
    """Emit a telemetry event.

    No-op if event_logger is None. Safe to call — logs errors but
    does not raise.
    """
    if event_logger is None:
        return
    try:
        data: dict = {
            "operation": operation,
            "outcome": outcome,
        }
        if details:
            data["details"] = details
        event_logger.emit(schema_id=SCHEMA_ID, data=data)
    except Exception:
        logger.warning("Failed to emit telemetry event %s.", operation, exc_info=True)

from contextlib import asynccontextmanager


@asynccontextmanager
async def track(event_logger, operation, details=None):
    """Emit success/failure telemetry around a block of code.

    On normal completion emits outcome="success". On exception emits
    outcome="failure" with the error message appended to details, then
    re-raises.
    """
    try:
        yield
        emit_event(event_logger, operation, "success", details)
    except Exception as e:
        fail_details = {**(details or {}), "error_message": f"{type(e).__name__}: {e}"}
        emit_event(event_logger, operation, "failure", fail_details)
        raise
