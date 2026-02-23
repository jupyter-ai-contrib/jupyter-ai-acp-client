from dataclasses import dataclass, field
from typing import Optional

from acp.schema import ToolCallProgress, ToolCallStart
from jupyter_ai_persona_manager import BasePersona
from jupyterlab_chat.models import NewMessage

from .tool_call_renderer import (
    ToolCallState,
    serialize_tool_calls,
    update_tool_call_from_progress,
    update_tool_call_from_start,
)


@dataclass
class SessionState:
    """Bundles per-session tool call state. Mirrors TerminalInfo."""

    tool_calls: dict[str, ToolCallState] = field(default_factory=dict)
    message_id: Optional[str] = None


class ToolCallManager:
    """Manages per-session tool call state and Yjs message rendering."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def _ensure_session(self, session_id: str) -> SessionState:
        """Return the SessionState for a session, creating one if absent."""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState()
        return self._sessions[session_id]

    def reset(self, session_id: str) -> None:
        """
        Reset tool call state for a session.

        Should be called at the start of each prompt_and_reply to clear
        state from a previous turn.
        """
        self._sessions[session_id] = SessionState()

    def get_message_id(self, session_id: str) -> Optional[str]:
        """
        Return the current message ID for a session, or None if no message
        has been created yet.
        """
        session = self._sessions.get(session_id)
        return session.message_id if session else None

    def get_or_create_message(self, session_id: str, persona: BasePersona) -> str:
        """
        Get the existing message ID for a session, or create a new Yjs message.

        Returns the message ID.
        """
        session = self._ensure_session(session_id)
        if session.message_id:
            return session.message_id

        message_id = persona.ychat.add_message(
            NewMessage(body="", sender=persona.id),
            trigger_actions=[],
        )
        session.message_id = message_id
        persona.log.info(f"Created message {message_id} for session {session_id}")

        # Update awareness to point to the message
        persona.awareness.set_local_state_field("isWriting", message_id)

        return message_id

    def serialize(self, session_id: str) -> list[dict]:
        """
        Return the serialized tool calls list for the session.

        Used by callers that construct Yjs Message objects directly (e.g.
        JaiAcpClient._handle_agent_message_chunk).
        """
        session = self._sessions.get(session_id)
        if session is None:
            return []
        return serialize_tool_calls(session.tool_calls)

    def handle_start(
        self, session_id: str, update: ToolCallStart, persona: BasePersona
    ) -> None:
        """Handle a ToolCallStart event."""
        session = self._ensure_session(session_id)
        kind_str = update.kind if update.kind else None
        locations_paths = (
            [loc.path for loc in update.locations] if update.locations else None
        )
        persona.log.info(
            f"tool_call_start: id={update.tool_call_id} title={update.title!r}"
            f" kind={kind_str} locations={locations_paths}"
        )
        update_tool_call_from_start(
            session.tool_calls,
            tool_call_id=update.tool_call_id,
            title=update.title,
            kind=kind_str,
            locations=locations_paths,
        )

        self.get_or_create_message(session_id, persona)
        self._flush_to_message(session_id, persona)

    def handle_progress(
        self, session_id: str, update: ToolCallProgress, persona: BasePersona
    ) -> None:
        """Handle a ToolCallProgress event."""
        session = self._ensure_session(session_id)

        # Convert raw_output to a serializable value
        raw_output = update.raw_output
        if raw_output is not None and not isinstance(
            raw_output, (str, int, float, bool, list, dict)
        ):
            raw_output = str(raw_output)

        kind_str = update.kind if update.kind else None
        status_str = update.status if update.status else None
        locations_paths = (
            [loc.path for loc in update.locations] if update.locations else None
        )
        persona.log.info(
            f"tool_call_progress: id={update.tool_call_id} title={update.title!r}"
            f" status={status_str} locations={locations_paths}"
        )
        update_tool_call_from_progress(
            session.tool_calls,
            tool_call_id=update.tool_call_id,
            title=update.title,
            kind=kind_str,
            status=status_str,
            raw_output=raw_output,
            locations=locations_paths,
        )

        # Message should exist from the preceding ToolCallStart, but create if missing
        self.get_or_create_message(session_id, persona)
        self._flush_to_message(session_id, persona)

    def _flush_to_message(self, session_id: str, persona: BasePersona) -> None:
        """Update the Yjs message metadata with the current tool call state."""
        session = self._sessions.get(session_id)
        if session is None or not session.message_id:
            return

        msg = persona.ychat.get_message(session.message_id)
        if msg:
            serialized = serialize_tool_calls(session.tool_calls)
            persona.log.info(
                f"flush_to_message: message={session.message_id}"
                f" count={len(session.tool_calls)} payload={serialized}"
            )
            msg.metadata = {"tool_calls": serialized}
            persona.ychat.update_message(msg, trigger_actions=[])
