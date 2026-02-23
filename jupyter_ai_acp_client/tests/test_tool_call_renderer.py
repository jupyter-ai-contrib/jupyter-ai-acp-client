from jupyter_ai_acp_client.tool_call_renderer import (
    ToolCallState,
    _shorten_title,
    _generate_title,
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

    def test_empty_title_generates_from_kind_and_locations(self):
        tool_calls = {}
        update_tool_call_from_start(
            tool_calls,
            tool_call_id="tc-1",
            title="",
            kind="read",
            locations=["/Users/foo/project/justfile"],
        )
        assert tool_calls["tc-1"].title == "Reading justfile"


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

    def test_serializes_locations(self):
        tool_calls = {
            "tc-1": ToolCallState(
                tool_call_id="tc-1",
                title="Read justfile",
                kind="read",
                status="completed",
                locations=["/Users/foo/project/justfile"],
            )
        }
        result = serialize_tool_calls(tool_calls)
        assert result[0]["locations"] == ["/Users/foo/project/justfile"]

    def test_strips_none_locations(self):
        tool_calls = {
            "tc-1": ToolCallState(
                tool_call_id="tc-1",
                title="Working...",
                status="in_progress",
                locations=None,
            )
        }
        result = serialize_tool_calls(tool_calls)
        assert "locations" not in result[0]


class TestShortenTitle:
    def test_absolute_path(self):
        assert _shorten_title("Read /Users/foo/bar/justfile") == "Read justfile"

    def test_multiple_absolute_paths(self):
        assert _shorten_title("Moved /a/b/old.py /a/b/new.py") == "Moved old.py new.py"

    def test_no_paths(self):
        assert _shorten_title("Read File") == "Read File"

    def test_relative_path(self):
        # Doesn't start with / — not an absolute path, leave as-is
        assert _shorten_title("Read src/foo.py") == "Read src/foo.py"

    def test_single_component_path(self):
        # Starts with / but only one component — not shortened
        assert _shorten_title("Read /justfile") == "Read /justfile"

    def test_empty_string(self):
        assert _shorten_title("") == ""

    def test_only_path(self):
        assert _shorten_title("/Users/foo/bar/baz.py") == "baz.py"

    def test_mixed_words_and_paths(self):
        assert _shorten_title("Editing /a/b/c.py with changes") == "Editing c.py with changes"


class TestShortenTitleIntegration:
    def test_start_shortens_agent_title(self):
        tool_calls = {}
        update_tool_call_from_start(
            tool_calls,
            tool_call_id="tc-1",
            title="Read /Users/aieroshe/Documents/project/justfile",
            kind="read",
        )
        assert tool_calls["tc-1"].title == "Read justfile"

    def test_start_preserves_short_title(self):
        tool_calls = {}
        update_tool_call_from_start(
            tool_calls,
            tool_call_id="tc-1",
            title="Read File",
            kind="read",
        )
        assert tool_calls["tc-1"].title == "Read File"

    def test_start_stores_locations(self):
        tool_calls = {}
        update_tool_call_from_start(
            tool_calls,
            tool_call_id="tc-1",
            title="Read File",
            kind="read",
            locations=["/Users/foo/project/justfile"],
        )
        assert tool_calls["tc-1"].locations == ["/Users/foo/project/justfile"]

    def test_progress_shortens_title_update(self):
        tool_calls = {
            "tc-1": ToolCallState(
                tool_call_id="tc-1",
                title="Read File",
                kind="read",
                status="in_progress",
            )
        }
        update_tool_call_from_progress(
            tool_calls,
            tool_call_id="tc-1",
            title="Read /Users/aieroshe/Documents/project/justfile",
            status="completed",
        )
        assert tool_calls["tc-1"].title == "Read justfile"

    def test_progress_creates_with_shortened_title(self):
        tool_calls = {}
        update_tool_call_from_progress(
            tool_calls,
            tool_call_id="tc-1",
            title="Read /Users/aieroshe/Documents/project/justfile",
            status="completed",
        )
        assert tool_calls["tc-1"].title == "Read justfile"

    def test_progress_stores_locations(self):
        tool_calls = {}
        update_tool_call_from_progress(
            tool_calls,
            tool_call_id="tc-1",
            title="Read File",
            status="in_progress",
            locations=["/Users/foo/project/justfile"],
        )
        assert tool_calls["tc-1"].locations == ["/Users/foo/project/justfile"]

    def test_progress_updates_locations(self):
        tool_calls = {
            "tc-1": ToolCallState(
                tool_call_id="tc-1",
                title="Read File",
                kind="read",
                status="in_progress",
            )
        }
        update_tool_call_from_progress(
            tool_calls,
            tool_call_id="tc-1",
            locations=["/Users/foo/project/justfile"],
        )
        assert tool_calls["tc-1"].locations == ["/Users/foo/project/justfile"]

    def test_full_flow_start_start_progress(self):
        """Simulate actual agent flow: two starts + completed progress."""
        tool_calls = {}

        # ToolCallStart #1: generic title, no locations
        update_tool_call_from_start(
            tool_calls, tool_call_id="tc-1",
            title="Read File", kind="read",
        )
        assert tool_calls["tc-1"].title == "Read File"
        assert tool_calls["tc-1"].status == "in_progress"

        # ToolCallStart #2: full path, with locations
        update_tool_call_from_start(
            tool_calls, tool_call_id="tc-1",
            title="Read /Users/aieroshe/Documents/project/justfile",
            kind="read",
            locations=["/Users/aieroshe/Documents/project/justfile"],
        )
        assert tool_calls["tc-1"].title == "Read justfile"
        assert tool_calls["tc-1"].locations == ["/Users/aieroshe/Documents/project/justfile"]

        # ToolCallProgress: completed, no title change
        update_tool_call_from_progress(
            tool_calls, tool_call_id="tc-1",
            status="completed",
        )
        assert tool_calls["tc-1"].title == "Read justfile"
        assert tool_calls["tc-1"].status == "completed"

        # Serialize — locations should be included for frontend
        result = serialize_tool_calls(tool_calls)
        assert result[0]["title"] == "Read justfile"
        assert result[0]["locations"] == ["/Users/aieroshe/Documents/project/justfile"]
        assert result[0]["status"] == "completed"


class TestGenerateTitle:
    def test_kind_with_locations(self):
        assert _generate_title("read", ["/Users/foo/project/justfile"]) == "Reading justfile"

    def test_kind_without_locations(self):
        assert _generate_title("read") == "Reading..."

    def test_edit_kind(self):
        assert _generate_title("edit", ["/a/b/c.py"]) == "Editing c.py"

    def test_execute_kind(self):
        assert _generate_title("execute") == "Running command..."

    def test_no_kind_no_locations(self):
        assert _generate_title(None) == "Working..."

    def test_unknown_kind(self):
        assert _generate_title("unknown_kind") == "Working..."

    def test_location_without_slash(self):
        assert _generate_title("read", ["justfile"]) == "Reading justfile"

    def test_multiple_locations_uses_first(self):
        assert _generate_title("read", ["/a/b/first.py", "/a/b/second.py"]) == "Reading first.py"
