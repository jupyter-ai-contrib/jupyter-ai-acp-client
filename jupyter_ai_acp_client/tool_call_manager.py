from dataclasses import dataclass, field
from typing import Optional

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
    """Bundles per-session tool call state."""

    tool_calls: dict[str, ToolCallState] = field(default_factory=dict)
    current_message_id: Optional[str] = None
    # Maps tool_call_id → message_id for targeted progress updates
    tool_call_message_ids: dict[str, str] = field(default_factory=dict)
    # Maps message_id → [tool_call_id, ...] (reverse of tool_call_message_ids)
    message_tool_call_ids: dict[str, list[str]] = field(default_factory=dict)
    # All message IDs created this turn (for find_mentions at end)
    all_message_ids: list[str] = field(default_factory=list)


class ToolCallManager:
    """Manages per-session tool call state and Yjs message rendering.

    Consecutive tool calls are grouped into a single Yjs message;
    text chunks get separate messages. A text message between tool calls
    starts a new group.
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
        if (session.current_message_id is not None
                and session.current_message_id not in session.message_tool_call_ids):
            return session.current_message_id

        return self._create_message(session_id, persona)

    def _assign_tool_call(
        self, session_id: str, tool_call_id: str, persona: BasePersona
    ) -> bool:
        """Assign a tool call to a Yjs message, grouping consecutive tool calls.

        Returns True if assigned to a new or existing message, False if already
        assigned to a message.

        If the current message contains no tool calls (i.e. is a text message),
        creates a new message and assigns the tool call there. Otherwise,
        appends the tool call to the current message.
        """
        session = self._ensure_session(session_id)
        if tool_call_id in session.tool_call_message_ids:
            return False

        if (session.current_message_id is not None
                and session.current_message_id in session.message_tool_call_ids):
            message_id = session.current_message_id
        else:
            message_id = self._create_message(session_id, persona)

        session.tool_call_message_ids[tool_call_id] = message_id
        session.message_tool_call_ids.setdefault(message_id, []).append(
            tool_call_id
        )
        return True

    def flush_tool_call(
        self, session_id: str, tool_call_id: str, persona: BasePersona
    ) -> None:
        """Update a specific tool call's Yjs message with its current state."""
        session = self._sessions.get(session_id)
        if session is None:
            return

        message_id = session.tool_call_message_ids.get(tool_call_id)
        if not message_id:
            persona.log.warning(
                f"flush_tool_call: no message_id for tool_call {tool_call_id}"
                f" in session {session_id}"
            )
            return

        tc = session.tool_calls.get(tool_call_id)
        if not tc:
            persona.log.warning(
                f"flush_tool_call: no ToolCallState for {tool_call_id}"
                f" in session {session_id}"
            )
            return

        msg = persona.ychat.get_message(message_id)
        if not msg:
            persona.log.warning(
                f"flush_tool_call: Yjs message {message_id} not found"
                f" for tool_call {tool_call_id}"
            )
            return
        # Build tool_calls array from reverse index
        tc_ids = session.message_tool_call_ids.get(message_id, [])
        all_tcs = [
            session.tool_calls[tc_id].model_dump(exclude_none=True)
            for tc_id in tc_ids
            if tc_id in session.tool_calls
        ]
        metadata = dict(msg.metadata or {})
        metadata["tool_calls"] = all_tcs
        msg.metadata = metadata
        # `mime_model` cannot be updated in place by the current jupyter-chat
        # shared model, so keep tool call state in message metadata and render
        # it through the chat preamble registry on the frontend.
        persona.ychat.update_message(msg, trigger_actions=[])

    def cancel_pending_tool_calls(
        self, session_id: str, persona: BasePersona
    ) -> None:
        """Mark non-finished tool calls as failed and flush each to its message."""
        session = self._sessions.get(session_id)
        if session is None:
            return
        for tc_id, tc in session.tool_calls.items():
            if tc.status not in ("completed", "failed"):
                tc.status = "failed"
                self.flush_tool_call(session_id, tc_id, persona)

    def handle_start(
        self, session_id: str, update: ToolCallStart, persona: BasePersona
    ) -> None:
        """Handle a ToolCallStart event.

        Groups with the current message if consecutive tool calls, otherwise
        creates a new one. If this tool_call_id already has a message, updates
        the existing one instead.
        """
        session = self._ensure_session(session_id)
        kind_str = update.kind or None
        locations_paths = (
            [loc.path for loc in update.locations] if update.locations else None
        )
        diffs = extract_diffs(update.content, root_dir=persona.parent.root_dir)

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

        self._assign_tool_call(session_id, update.tool_call_id, persona)
        self.flush_tool_call(session_id, update.tool_call_id, persona)

    def handle_progress(
        self, session_id: str, update: ToolCallProgress, persona: BasePersona
    ) -> None:
        """Handle a ToolCallProgress event.

        Updates the tool call state and flushes to its assigned message.
        If the tool_call_id has no prior message, groups with the current
        message or creates a new one.
        """
        session = self._ensure_session(session_id)

        raw_input = ensure_serializable(update.raw_input)
        raw_output = ensure_serializable(update.raw_output)

        kind_str = update.kind or None
        status_str = update.status or None
        locations_paths = (
            [loc.path for loc in update.locations] if update.locations else None
        )
        diffs = extract_diffs(update.content, root_dir=persona.parent.root_dir)
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

        self._assign_tool_call(session_id, update.tool_call_id, persona)
        self.flush_tool_call(session_id, update.tool_call_id, persona)
