"""
Tool call state tracking and serialization for the ACP tool call UI.

This module provides pure functions and Pydantic models for managing tool call
state from ACP ToolCallStart/ToolCallProgress events, and serializing them
for Yjs transport as part of chat messages.
"""

from typing import Optional, Any, Literal

from pydantic import BaseModel
from acp.schema import PermissionOption


class ToolCallState(BaseModel):
    """Tracks the state of a single tool call."""
    tool_call_id: str
    title: str
    kind: Optional[str] = None
    status: Optional[str] = None
    raw_output: Optional[Any] = None
    locations: Optional[list[str]] = None
    permission_options: Optional[list[PermissionOption]] = None
    permission_status: Optional[Literal['pending', 'resolved']] = None
    selected_option_id: Optional[str] = None
    session_id: Optional[str] = None


def _generate_title(kind: Optional[str], locations: Optional[list[str]] = None) -> str:
    """Generate a human-readable title from tool call metadata."""
    kind_verbs = {
        "read": "Reading",
        "edit": "Editing",
        "delete": "Deleting",
        "move": "Moving",
        "search": "Searching",
        "execute": "Running command",
        "think": "Thinking",
        "fetch": "Fetching",
        "switch_mode": "Switching mode",
    }
    verb = kind_verbs.get(kind or "", "Working")

    if locations:
        path = locations[0]
        filename = path.rsplit("/", 1)[-1] if "/" in path else path
        return f"{verb} {filename}"

    return f"{verb}..."


def _shorten_title(title: str) -> str:
    """Replace absolute paths in a title with just the filename."""
    words = title.split()
    return " ".join(
        word.rsplit("/", 1)[-1] if word.startswith("/") and "/" in word[1:] else word
        for word in words
    )


def update_tool_call_from_start(
    tool_calls: dict[str, ToolCallState],
    tool_call_id: str,
    title: str,
    kind: Optional[str] = None,
    locations: Optional[list[str]] = None,
) -> None:
    """
    Apply a ToolCallStart event to the tool calls dict.

    Creates a new ToolCallState with status 'in_progress'.
    Generates a title from kind/locations if the agent sends an empty title.
    """
    if not title and (kind or locations):
        title = _generate_title(kind, locations)
    elif not title:
        title = "Working..."
    else:
        title = _shorten_title(title)

    tool_calls[tool_call_id] = ToolCallState(
        tool_call_id=tool_call_id,
        title=title,
        kind=kind,
        status="in_progress",
        locations=locations,
    )


def update_tool_call_from_progress(
    tool_calls: dict[str, ToolCallState],
    tool_call_id: str,
    title: Optional[str] = None,
    kind: Optional[str] = None,
    status: Optional[str] = None,
    raw_output: Optional[Any] = None,
    locations: Optional[list[str]] = None,
) -> None:
    """
    Apply a ToolCallProgress event to the tool calls dict.

    Updates an existing ToolCallState with new title, status, and/or raw_output.
    If the tool_call_id doesn't exist, creates one.
    Generates a title from kind/locations if the title is empty.
    """
    if tool_call_id not in tool_calls:
        resolved_title = _shorten_title(title) if title else ""
        if not resolved_title and (kind or locations):
            resolved_title = _generate_title(kind, locations)
        elif not resolved_title:
            resolved_title = "Working..."
        tool_calls[tool_call_id] = ToolCallState(
            tool_call_id=tool_call_id,
            title=resolved_title,
            kind=kind,
            status=status or "in_progress",
            raw_output=raw_output,
            locations=locations,
        )
        return

    tc = tool_calls[tool_call_id]
    if title is not None:
        tc.title = _shorten_title(title)
    if kind is not None:
        tc.kind = kind
    if status is not None:
        tc.status = status
    if raw_output is not None:
        tc.raw_output = raw_output
    if locations is not None:
        tc.locations = locations
