from dataclasses import dataclass, field
from typing import Literal, Optional

from acp.schema import ToolCallProgress, ToolCallStart
from jupyter_ai_persona_manager import BasePersona
from jupyterlab_chat.models import NewMessage

from .tool_call_renderer import (
    ToolCallState,
    ensure_serializable,
    extract_diffs,
    update_tool_call_from_progress,
    update_tool_call_from_start,
)


@dataclass
class SessionState:
    """Bundles per-session tool call state.

    Tracks a state machine with three states determined by
    ``current_message_type``:

    - ``None`` (IDLE): no message created yet this turn
    - ``"text"``: currently appending to a text message
    - ``"tool_call"``: last message created was for a tool call
    """

    tool_calls: dict[str, ToolCallState] = field(default_factory=dict)
    current_message_id: Optional[str] = None
    current_message_type: Optional[Literal["text", "tool_call"]] = None
    # Maps tool_call_id → message_id for targeted progress updates
    tool_call_message_ids: dict[str, str] = field(default_factory=dict)
    # All message IDs created this turn (for find_mentions at end)
    all_message_ids: list[str] = field(default_factory=list)


class ToolCallManager:
    """Manages per-session tool call state and Yjs message rendering.

    Implements a message-per-tool-call architecture: each ToolCallStart
    creates a new Yjs message with ``body=""`` and the tool call in metadata.
    Text chunks create or append to separate text messages. Stream order is
    preserved by Y.Array insertion order.
    """

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

    def cleanup(self, session_id: str) -> None:
        """
        Remove all state for a completed session.

        Should be called when a session ends to prevent unbounded memory growth.
        """
        self._sessions.pop(session_id, None)

    def get_all_message_ids(self, session_id: str) -> list[str]:
        """Return all message IDs created this turn, in creation order."""
        session = self._sessions.get(session_id)
        return list(session.all_message_ids) if session else []

    def get_tool_call(
        self, session_id: str, tool_call_id: str
    ) -> Optional[ToolCallState]:
        """Return the ToolCallState for a tool call, or None if not found."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        return session.tool_calls.get(tool_call_id)

    def _create_message(self, session_id: str, persona: BasePersona) -> str:
        """Create a new Yjs message and return its ID.

        Updates awareness and records the message ID in ``all_message_ids``.
        Does NOT set ``current_message_type`` — callers own that.
        """
        session = self._ensure_session(session_id)

        message_id = persona.ychat.add_message(
            NewMessage(body="", sender=persona.id),
            trigger_actions=[],
        )
        session.current_message_id = message_id
        session.all_message_ids.append(message_id)
        persona.log.info(f"Created message {message_id} for session {session_id}")
        persona.awareness.set_local_state_field("isWriting", message_id)

        return message_id

    def get_or_create_text_message(
        self, session_id: str, persona: BasePersona
    ) -> str:
        """Get the current text message, or create a new one.

        If currently appending to a text message, returns the existing one.
        Otherwise, creates a new text message.
        """
        session = self._ensure_session(session_id)
        if session.current_message_type == "text" and session.current_message_id:
            return session.current_message_id

        message_id = self._create_message(session_id, persona)
        session.current_message_type = "text"
        return message_id

    def flush_tool_call(
        self, session_id: str, tool_call_id: str, persona: BasePersona
    ) -> None:
        """Update a specific tool call's Yjs message with its current state."""
        session = self._sessions.get(session_id)
        if session is None:
            return

        message_id = session.tool_call_message_ids.get(tool_call_id)
        if not message_id:
            return

        tc = session.tool_calls.get(tool_call_id)
        if not tc:
            return

        msg = persona.ychat.get_message(message_id)
        if not msg:
            return
        msg.metadata = {"tool_calls": [tc.model_dump(exclude_none=True)]}
        persona.ychat.update_message(msg, trigger_actions=[])

    def handle_start(
        self, session_id: str, update: ToolCallStart, persona: BasePersona
    ) -> None:
        """Handle a ToolCallStart event.

        Creates a new Yjs message for the tool call. If this tool_call_id
        already has a message, updates the existing one instead.
        """
        session = self._ensure_session(session_id)
        kind_str = update.kind or None
        locations_paths = (
            [loc.path for loc in update.locations] if update.locations else None
        )
        diffs = extract_diffs(update.content)

        raw_input = ensure_serializable(update.raw_input)

        persona.log.info(
            f"tool_call_start: id={update.tool_call_id} title={update.title!r}"
            f" kind={kind_str} locations={locations_paths}"
            f" diffs={len(diffs) if diffs else 0}"
        )
        update_tool_call_from_start(
            session.tool_calls,
            tool_call_id=update.tool_call_id,
            title=update.title,
            kind=kind_str,
            locations=locations_paths,
            diffs=diffs,
            raw_input=raw_input,
        )

        if update.tool_call_id not in session.tool_call_message_ids:
            message_id = self._create_message(session_id, persona)
            session.tool_call_message_ids[update.tool_call_id] = message_id

        session.current_message_type = "tool_call"
        self.flush_tool_call(session_id, update.tool_call_id, persona)

    def handle_progress(
        self, session_id: str, update: ToolCallProgress, persona: BasePersona
    ) -> None:
        """Handle a ToolCallProgress event.

        Updates the tool call state and flushes to its dedicated message.
        If the tool_call_id has no prior message, creates one.
        """
        session = self._ensure_session(session_id)

        raw_input = ensure_serializable(update.raw_input)
        raw_output = ensure_serializable(update.raw_output)

        kind_str = update.kind or None
        status_str = update.status or None
        locations_paths = (
            [loc.path for loc in update.locations] if update.locations else None
        )
        diffs = extract_diffs(update.content)
        persona.log.info(
            f"tool_call_progress: id={update.tool_call_id} title={update.title!r}"
            f" status={status_str} locations={locations_paths}"
            f" diffs={len(diffs) if diffs else 0}"
        )
        update_tool_call_from_progress(
            session.tool_calls,
            tool_call_id=update.tool_call_id,
            title=update.title,
            kind=kind_str,
            status=status_str,
            raw_input=raw_input,
            raw_output=raw_output,
            locations=locations_paths,
            diffs=diffs,
        )

        # Create a message if this tool_call_id has no prior one
        if update.tool_call_id not in session.tool_call_message_ids:
            message_id = self._create_message(session_id, persona)
            session.current_message_type = "tool_call"
            session.tool_call_message_ids[update.tool_call_id] = message_id

        self.flush_tool_call(session_id, update.tool_call_id, persona)
