"""
Telemetry module for the Jupyter AI ACP Client extension.

Provides a single unified event schema and a generic emit helper for
observability via Jupyter Server's built-in EventLogger system. All
telemetry is purely operational metadata — no customer content or PII
is emitted.
"""

from __future__ import annotations

import logging
import functools
from contextlib import asynccontextmanager
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
        logger.error("Failed to register ACP event schema with EventLogger.", exc_info=True)


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
        logger.warning("Failed to emit ACP event to EventLogger: %s", operation, exc_info=True)

def auto_emit_event(operation: str, details_fn=None):
    def decorator(method):
        @functools.wraps(method)
        async def wrapper(self, *args, **kwargs):
            details = {"persona_class": self.__class__.__name__}
            if details_fn:
                details.update(details_fn(self))
            try:
                result = await method(self, *args, **kwargs)
                emit_event(self.event_logger, operation, "success", details)
                return result
            except Exception as e:
                fail_details = {**details, "error_message": f"{type(e).__name__}: {e}"}
                emit_event(self.event_logger, operation, "failure", fail_details)
                raise
        return wrapper
    return decorator
