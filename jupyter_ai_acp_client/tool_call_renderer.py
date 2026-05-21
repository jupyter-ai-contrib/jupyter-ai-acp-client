"""
Tool call state tracking and serialization for the ACP tool call UI.

This module provides pure functions and Pydantic models for managing tool call
state from ACP ToolCallStart/ToolCallProgress events, and serializing them
for Yjs transport as part of chat messages.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from acp.schema import (
    ContentToolCallContent,
    FileEditToolCallContent,
    PermissionOption,
    TerminalToolCallContent,
)
from pydantic import BaseModel


def ensure_serializable(value: Optional[Any]) -> Optional[Any]:
    """Convert non-JSON-serializable values to strings for Yjs transport."""
    if value is not None and not isinstance(
        value, (str, int, float, bool, list, dict)
    ):
        return str(value)
    return value


@dataclass
class ToolCallDiff:
    """A single file diff from an ACP tool call."""
    path: str
    new_text: str
    old_text: Optional[str] = None


class ToolCallState(BaseModel):
    """Tracks the state of a single tool call."""
    tool_call_id: str
    title: str
    kind: Optional[str] = None
    status: Optional[str] = None
    raw_input: Optional[Any] = None
    raw_output: Optional[Any] = None
    locations: Optional[list[str]] = None
    permission_options: Optional[list[PermissionOption]] = None
    permission_status: Optional[Literal['pending', 'resolved']] = None
    selected_option_id: Optional[str] = None
    session_id: Optional[str] = None
    diffs: Optional[list[ToolCallDiff]] = None

def _resolve_path(path: str, root_dir: Optional[str]) -> str:
    """Normalize a file path to an absolute path using root_dir if needed."""
    if root_dir:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = (Path(root_dir) / p).resolve()
        return str(p)
    return path


def _parse_unified_diff(diff_str: str) -> tuple[str, str] | None:
    """Parse a unified diff string into (old_text, new_text).

    Returns None if the string doesn't contain valid unified diff hunks.
    """
    if "@@" not in diff_str:
        return None

    old_lines: list[str] = []
    new_lines: list[str] = []
    in_hunk = False

    for line in diff_str.split("\n"):
        if line.startswith("@@"):
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("+"):
            new_lines.append(line[1:])
        elif line.startswith("-"):
            old_lines.append(line[1:])
        elif line.startswith(" "):
            old_lines.append(line[1:])
            new_lines.append(line[1:])

    if not old_lines and not new_lines:
        return None

    return "\n".join(old_lines), "\n".join(new_lines)


def extract_diffs_from_raw_input(
    raw_input: Optional[Any],
    root_dir: Optional[str] = None,
) -> Optional[list[ToolCallDiff]]:
    """Extract diffs from raw_input when tool_call.content has no FileEditToolCallContent.

    Handles agents that send a unified diff string in raw_input.diff
    instead of using FileEditToolCallContent in tool_call.content.
    """
    if not isinstance(raw_input, dict):
        return None

    filepath = raw_input.get("filepath") or raw_input.get("filePath")
    diff_str = raw_input.get("diff")

    if not filepath or not isinstance(diff_str, str):
        return None

    parsed = _parse_unified_diff(diff_str)
    if parsed is None:
        return None

    old_text, new_text = parsed
    path = _resolve_path(filepath, root_dir)

    return [ToolCallDiff(path=path, new_text=new_text, old_text=old_text or None)]


def extract_diffs(
    content: Optional[
        list[ContentToolCallContent | FileEditToolCallContent | TerminalToolCallContent]
    ],
    root_dir: Optional[str] = None,
) -> Optional[list[ToolCallDiff]]:
    """Extract FileEditToolCallContent items from an ACP content list.

    When root_dir is provided, normalizes relative and tilde paths to absolute
    so that ToolCallDiff.path is always a resolved filesystem path.
    """
    if not content:
        return None
    diffs = []
    for item in content:
        if isinstance(item, FileEditToolCallContent):
            path = _resolve_path(item.path, root_dir)
            diffs.append(
                ToolCallDiff(path=path, new_text=item.new_text, old_text=item.old_text)
            )
    return diffs or None


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
    diffs: Optional[list[ToolCallDiff]] = None,
    raw_input: Optional[Any] = None,
) -> None:
    """
    Apply a ToolCallStart event to the tool calls dict.

    Creates a new ToolCallState with status 'in_progress'.
    Merges into an existing ToolCallState if one already exists (e.g. from a
    prior start or request_permission), preserving permission-related fields.
    Generates a title from kind/locations if the agent sends an empty title.
    """
    if not title and (kind or locations):
        title = _generate_title(kind, locations)
    elif not title:
        title = "Working..."
    else:
        title = _shorten_title(title)

    if tool_call_id in tool_calls:
        tc = tool_calls[tool_call_id]
        tc.title = title
        if kind is not None:
            tc.kind = kind
        if locations is not None:
            tc.locations = locations
        if diffs is not None:
            tc.diffs = diffs
        if raw_input is not None:
            tc.raw_input = raw_input
        return

    tool_calls[tool_call_id] = ToolCallState(
        tool_call_id=tool_call_id,
        title=title,
        kind=kind,
        status="in_progress",
        raw_input=raw_input,
        locations=locations,
        diffs=diffs,
    )


def update_tool_call_from_progress(
    tool_calls: dict[str, ToolCallState],
    tool_call_id: str,
    title: Optional[str] = None,
    kind: Optional[str] = None,
    status: Optional[str] = None,
    raw_input: Optional[Any] = None,
    raw_output: Optional[Any] = None,
    locations: Optional[list[str]] = None,
    diffs: Optional[list[ToolCallDiff]] = None,
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
            raw_input=raw_input,
            raw_output=raw_output,
            locations=locations,
            diffs=diffs,
        )
        return

    tc = tool_calls[tool_call_id]
    if title is not None:
        tc.title = _shorten_title(title)
    if kind is not None:
        tc.kind = kind
    # "failed" is terminal: don't let late-arriving updates overwrite it
    if status is not None and tc.status != "failed":
        tc.status = status
    if raw_input is not None:
        tc.raw_input = raw_input
    if raw_output is not None:
        tc.raw_output = raw_output
    if locations is not None:
        tc.locations = locations
    if diffs is not None:
        tc.diffs = diffs
