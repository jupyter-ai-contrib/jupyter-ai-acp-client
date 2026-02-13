"""Unit tests for the tool_call_renderer module."""

import pytest
from jupyter_ai_acp_client.tool_call_renderer import (
    ToolCallState,
    update_tool_call_from_start,
    update_tool_call_from_progress,
    serialize_tool_calls,
)


class TestUpdateToolCallFromStart:
    def test_creates_new_tool_call(self):
        tool_calls = {}
        update_tool_call_from_start(
            tool_calls,
            tool_call_id="tc-1",
            title="Reading file.py...",
            kind="read",
        )
        assert "tc-1" in tool_calls
        tc = tool_calls["tc-1"]
        assert tc.tool_call_id == "tc-1"
        assert tc.title == "Reading file.py..."
        assert tc.kind == "read"
        assert tc.status == "in_progress"
        assert tc.raw_output is None

    def test_overwrites_existing_tool_call(self):
        tool_calls = {
            "tc-1": ToolCallState(
                tool_call_id="tc-1",
                title="Old title",
                kind="read",
                status="completed",
            )
        }
        update_tool_call_from_start(
            tool_calls,
            tool_call_id="tc-1",
            title="New title",
            kind="edit",
        )
        tc = tool_calls["tc-1"]
        assert tc.title == "New title"
        assert tc.kind == "edit"
        assert tc.status == "in_progress"

    def test_without_kind(self):
        tool_calls = {}
        update_tool_call_from_start(
            tool_calls,
            tool_call_id="tc-1",
            title="Doing something",
        )
        assert tool_calls["tc-1"].kind is None


class TestUpdateToolCallFromProgress:
    def test_updates_existing_tool_call(self):
        tool_calls = {
            "tc-1": ToolCallState(
                tool_call_id="tc-1",
                title="Reading file.py...",
                kind="read",
                status="in_progress",
            )
        }
        update_tool_call_from_progress(
            tool_calls,
            tool_call_id="tc-1",
            title="Read file.py (42 lines)",
            status="completed",
        )
        tc = tool_calls["tc-1"]
        assert tc.title == "Read file.py (42 lines)"
        assert tc.status == "completed"

    def test_creates_if_not_exists(self):
        tool_calls = {}
        update_tool_call_from_progress(
            tool_calls,
            tool_call_id="tc-1",
            title="Some progress",
            status="in_progress",
        )
        assert "tc-1" in tool_calls
        tc = tool_calls["tc-1"]
        assert tc.title == "Some progress"
        assert tc.status == "in_progress"

    def test_updates_raw_output(self):
        tool_calls = {
            "tc-1": ToolCallState(
                tool_call_id="tc-1",
                title="Running ls",
                kind="execute",
                status="in_progress",
            )
        }
        update_tool_call_from_progress(
            tool_calls,
            tool_call_id="tc-1",
            status="completed",
            raw_output="file1.py\nfile2.py\n",
        )
        tc = tool_calls["tc-1"]
        assert tc.status == "completed"
        assert tc.raw_output == "file1.py\nfile2.py\n"

    def test_partial_update_preserves_existing(self):
        tool_calls = {
            "tc-1": ToolCallState(
                tool_call_id="tc-1",
                title="Reading file.py...",
                kind="read",
                status="in_progress",
            )
        }
        # Only update status, not title
        update_tool_call_from_progress(
            tool_calls,
            tool_call_id="tc-1",
            status="completed",
        )
        tc = tool_calls["tc-1"]
        assert tc.title == "Reading file.py..."  # preserved
        assert tc.status == "completed"  # updated

    def test_none_values_dont_overwrite(self):
        tool_calls = {
            "tc-1": ToolCallState(
                tool_call_id="tc-1",
                title="Original title",
                kind="read",
                status="in_progress",
            )
        }
        update_tool_call_from_progress(
            tool_calls,
            tool_call_id="tc-1",
            title=None,
            status=None,
        )
        tc = tool_calls["tc-1"]
        assert tc.title == "Original title"
        assert tc.status == "in_progress"


class TestSerializeToolCalls:
    def test_empty_dict(self):
        result = serialize_tool_calls({})
        assert result == []

    def test_single_tool_call(self):
        tool_calls = {
            "tc-1": ToolCallState(
                tool_call_id="tc-1",
                title="Reading file.py...",
                kind="read",
                status="in_progress",
            )
        }
        result = serialize_tool_calls(tool_calls)
        assert len(result) == 1
        assert result[0] == {
            "tool_call_id": "tc-1",
            "title": "Reading file.py...",
            "kind": "read",
            "status": "in_progress",
        }

    def test_strips_none_values(self):
        tool_calls = {
            "tc-1": ToolCallState(
                tool_call_id="tc-1",
                title="Something",
                kind=None,
                status=None,
                raw_output=None,
            )
        }
        result = serialize_tool_calls(tool_calls)
        assert len(result) == 1
        assert result[0] == {
            "tool_call_id": "tc-1",
            "title": "Something",
        }

    def test_preserves_raw_output(self):
        tool_calls = {
            "tc-1": ToolCallState(
                tool_call_id="tc-1",
                title="Ran ls",
                kind="execute",
                status="completed",
                raw_output="file1\nfile2\n",
            )
        }
        result = serialize_tool_calls(tool_calls)
        assert result[0]["raw_output"] == "file1\nfile2\n"

    def test_multiple_tool_calls(self):
        tool_calls = {
            "tc-1": ToolCallState(
                tool_call_id="tc-1",
                title="Read file.py (42 lines)",
                kind="read",
                status="completed",
            ),
            "tc-2": ToolCallState(
                tool_call_id="tc-2",
                title="Writing output.py...",
                kind="edit",
                status="in_progress",
            ),
        }
        result = serialize_tool_calls(tool_calls)
        assert len(result) == 2
        ids = [r["tool_call_id"] for r in result]
        assert "tc-1" in ids
        assert "tc-2" in ids

    def test_dict_raw_output(self):
        tool_calls = {
            "tc-1": ToolCallState(
                tool_call_id="tc-1",
                title="API call",
                status="completed",
                raw_output={"key": "value"},
            )
        }
        result = serialize_tool_calls(tool_calls)
        assert result[0]["raw_output"] == {"key": "value"}

    def test_list_raw_output(self):
        tool_calls = {
            "tc-1": ToolCallState(
                tool_call_id="tc-1",
                title="Search",
                status="completed",
                raw_output=["item1", "item2"],
            )
        }
        result = serialize_tool_calls(tool_calls)
        assert result[0]["raw_output"] == ["item1", "item2"]
