from unittest.mock import MagicMock

from jupyter_ai_acp_client.tool_call_manager import ToolCallManager


def make_persona(message_id_seq: list[str] | str | None = None) -> MagicMock:
    """Return a mock persona whose ychat.add_message returns message IDs.

    If ``message_id_seq`` is a list, successive calls return successive IDs.
    If it is a single string, every call returns that string.
    """
    persona = MagicMock()
    persona.id = "persona-id"
    if message_id_seq is None:
        message_id_seq = ["msg-1"]
    if isinstance(message_id_seq, str):
        persona.ychat.add_message.return_value = message_id_seq
    else:
        persona.ychat.add_message.side_effect = list(message_id_seq)
    persona.ychat.get_message.return_value = MagicMock()
    return persona


def make_tool_call_start(
    tool_call_id: str = "tc-1",
    title: str = "Reading file.py",
    kind: str = "read",
    locations: list[MagicMock] | None = None,
    raw_input: object = None,
) -> MagicMock:
    update = MagicMock()
    update.tool_call_id = tool_call_id
    update.title = title
    update.kind = kind
    update.locations = locations
    update.raw_input = raw_input
    update.content = None
    return update


def make_tool_call_progress(
    tool_call_id: str = "tc-1",
    title: str | None = None,
    kind: str | None = None,
    status: str | None = "completed",
    raw_input: object = None,
    raw_output: object = None,
    locations: list[MagicMock] | None = None,
) -> MagicMock:
    update = MagicMock()
    update.tool_call_id = tool_call_id
    update.title = title
    update.kind = kind
    update.status = status
    update.raw_input = raw_input
    update.raw_output = raw_output
    update.locations = locations
    update.content = None
    return update


SESSION_ID = "session-abc"


class TestReset:
    def test_creates_fresh_session(self):
        mgr = ToolCallManager()
        mgr.reset(SESSION_ID)
        session = mgr._sessions[SESSION_ID]
        assert session.tool_calls == {}
        assert session.current_message_id is None
        assert session.current_message_type is None
        assert session.tool_call_message_ids == {}
        assert session.all_message_ids == []

    def test_clears_existing_state(self):
        mgr = ToolCallManager()
        persona = make_persona(["msg-1"])
        mgr.handle_start(SESSION_ID, make_tool_call_start(), persona)
        assert mgr._sessions[SESSION_ID].tool_calls  # non-empty

        mgr.reset(SESSION_ID)
        session = mgr._sessions[SESSION_ID]
        assert session.tool_calls == {}
        assert session.current_message_id is None
        assert session.current_message_type is None
        assert session.tool_call_message_ids == {}
        assert session.all_message_ids == []


class TestCleanup:
    def test_removes_session(self):
        mgr = ToolCallManager()
        persona = make_persona(["msg-1"])
        mgr.handle_start(SESSION_ID, make_tool_call_start(), persona)
        mgr.cleanup(SESSION_ID)
        assert SESSION_ID not in mgr._sessions
        assert mgr.get_all_message_ids(SESSION_ID) == []

    def test_noop_for_unknown_session(self):
        mgr = ToolCallManager()
        mgr.cleanup("nonexistent")  # should not raise


class TestGetToolCall:
    def test_returns_tool_call_state(self):
        mgr = ToolCallManager()
        persona = make_persona(["msg-1"])
        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1", "Reading"), persona)

        tc = mgr.get_tool_call(SESSION_ID, "tc-1")
        assert tc is not None
        assert tc.tool_call_id == "tc-1"
        assert tc.title == "Reading"

    def test_returns_none_for_unknown_session(self):
        mgr = ToolCallManager()
        assert mgr.get_tool_call("nonexistent", "tc-1") is None

    def test_returns_none_for_unknown_tool_call(self):
        mgr = ToolCallManager()
        mgr.reset(SESSION_ID)
        assert mgr.get_tool_call(SESSION_ID, "nonexistent") is None


class TestGetAllMessageIds:
    def test_returns_empty_for_unknown_session(self):
        mgr = ToolCallManager()
        assert mgr.get_all_message_ids(SESSION_ID) == []

    def test_returns_empty_before_any_messages(self):
        mgr = ToolCallManager()
        mgr.reset(SESSION_ID)
        assert mgr.get_all_message_ids(SESSION_ID) == []

    def test_returns_ids_in_creation_order(self):
        mgr = ToolCallManager()
        persona = make_persona(["msg-1", "msg-2", "msg-3"])
        mgr.reset(SESSION_ID)

        mgr.get_or_create_text_message(SESSION_ID, persona)
        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1"), persona)
        mgr.get_or_create_text_message(SESSION_ID, persona)

        assert mgr.get_all_message_ids(SESSION_ID) == ["msg-1", "msg-2", "msg-3"]


class TestGetOrCreateTextMessage:
    def test_creates_text_message_on_first_call(self):
        mgr = ToolCallManager()
        persona = make_persona("msg-1")
        mgr.reset(SESSION_ID)

        msg_id = mgr.get_or_create_text_message(SESSION_ID, persona)

        assert msg_id == "msg-1"
        persona.ychat.add_message.assert_called_once()
        assert mgr._sessions[SESSION_ID].current_message_type == "text"

    def test_returns_existing_on_consecutive_calls(self):
        mgr = ToolCallManager()
        persona = make_persona("msg-1")
        mgr.reset(SESSION_ID)

        first = mgr.get_or_create_text_message(SESSION_ID, persona)
        second = mgr.get_or_create_text_message(SESSION_ID, persona)

        assert first == second == "msg-1"
        persona.ychat.add_message.assert_called_once()

    def test_creates_new_after_tool_call(self):
        """After a tool call, the next text chunk creates a new message."""
        mgr = ToolCallManager()
        persona = make_persona(["msg-tc", "msg-text"])
        mgr.reset(SESSION_ID)

        mgr.handle_start(SESSION_ID, make_tool_call_start(), persona)
        assert mgr._sessions[SESSION_ID].current_message_type == "tool_call"

        msg_id = mgr.get_or_create_text_message(SESSION_ID, persona)
        assert msg_id == "msg-text"
        assert mgr._sessions[SESSION_ID].current_message_type == "text"
        assert persona.ychat.add_message.call_count == 2

    def test_sets_awareness_on_creation(self):
        mgr = ToolCallManager()
        persona = make_persona("msg-1")
        mgr.reset(SESSION_ID)

        mgr.get_or_create_text_message(SESSION_ID, persona)

        persona.awareness.set_local_state_field.assert_called_with("isWriting", "msg-1")

    def test_does_not_flush_tool_calls_on_creation(self):
        """get_or_create_text_message must not write tool call metadata."""
        mgr = ToolCallManager()
        persona = make_persona("msg-1")
        mgr.reset(SESSION_ID)

        mgr.get_or_create_text_message(SESSION_ID, persona)

        persona.ychat.update_message.assert_not_called()


class TestHandleStart:
    def test_adds_tool_call_to_session(self):
        mgr = ToolCallManager()
        persona = make_persona(["msg-1"])
        mgr.reset(SESSION_ID)

        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1", "Reading", "read"), persona)

        assert "tc-1" in mgr._sessions[SESSION_ID].tool_calls
        assert mgr._sessions[SESSION_ID].tool_calls["tc-1"].status == "in_progress"

    def test_creates_new_message_for_each_tool_call(self):
        """Each ToolCallStart gets its own Yjs message."""
        mgr = ToolCallManager()
        persona = make_persona(["msg-1", "msg-2"])
        mgr.reset(SESSION_ID)

        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1"), persona)
        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-2"), persona)

        assert persona.ychat.add_message.call_count == 2
        session = mgr._sessions[SESSION_ID]
        assert session.tool_call_message_ids["tc-1"] == "msg-1"
        assert session.tool_call_message_ids["tc-2"] == "msg-2"

    def test_flushes_to_message_after_state_update(self):
        mgr = ToolCallManager()
        persona = make_persona("msg-1")
        mgr.reset(SESSION_ID)

        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1"), persona)

        persona.ychat.update_message.assert_called_once()
        # Verify the metadata contains only this tool call
        flushed_msg = persona.ychat.update_message.call_args[0][0]
        assert len(flushed_msg.metadata["tool_calls"]) == 1
        assert flushed_msg.metadata["tool_calls"][0]["tool_call_id"] == "tc-1"

    def test_locations_extracted_from_update(self):
        mgr = ToolCallManager()
        persona = make_persona(["msg-1"])
        loc = MagicMock()
        loc.path = "/some/file.py"
        update = make_tool_call_start("tc-1", "Reading", "read", locations=[loc])

        mgr.handle_start(SESSION_ID, update, persona)

        tc = mgr._sessions[SESSION_ID].tool_calls["tc-1"]
        assert tc.locations == ["/some/file.py"]

    def test_empty_kind_treated_as_none(self):
        mgr = ToolCallManager()
        persona = make_persona(["msg-1"])
        update = make_tool_call_start("tc-1", "Working", kind="")

        mgr.handle_start(SESSION_ID, update, persona)

        tc = mgr._sessions[SESSION_ID].tool_calls["tc-1"]
        assert tc.kind is None

    def test_raw_input_passed_through(self):
        mgr = ToolCallManager()
        persona = make_persona()
        mgr.reset(SESSION_ID)

        update = make_tool_call_start("tc-1", "Running command", "execute", raw_input={"command": "ls -la"})
        mgr.handle_start(SESSION_ID, update, persona)

        tc = mgr._sessions[SESSION_ID].tool_calls["tc-1"]
        assert tc.raw_input == {"command": "ls -la"}

    def test_non_serializable_raw_input_is_stringified(self):
        mgr = ToolCallManager()
        persona = make_persona()
        mgr.reset(SESSION_ID)

        class CustomObj:
            def __str__(self):
                return "custom-input"

        update = make_tool_call_start("tc-1", "Running command", "execute", raw_input=CustomObj())
        mgr.handle_start(SESSION_ID, update, persona)

        tc = mgr._sessions[SESSION_ID].tool_calls["tc-1"]
        assert tc.raw_input == "custom-input"

    def test_does_not_bleed_into_another_session(self):
        """Two concurrent sessions must be fully isolated."""
        mgr = ToolCallManager()
        persona_a = make_persona(["msg-a"])
        persona_b = make_persona(["msg-b"])
        session_a = "session-A"
        session_b = "session-B"

        mgr.handle_start(session_a, make_tool_call_start("tc-a", "Task A"), persona_a)
        mgr.handle_start(session_b, make_tool_call_start("tc-b", "Task B"), persona_b)

        assert "tc-a" in mgr._sessions[session_a].tool_calls
        assert "tc-b" not in mgr._sessions[session_a].tool_calls
        assert "tc-b" in mgr._sessions[session_b].tool_calls
        assert "tc-a" not in mgr._sessions[session_b].tool_calls
        assert mgr.get_all_message_ids(session_a) == ["msg-a"]
        assert mgr.get_all_message_ids(session_b) == ["msg-b"]


    def test_repeated_start_same_id_reuses_message(self):
        """ACP sends two ToolCallStart for the same ID; must reuse the message."""
        mgr = ToolCallManager()
        persona = make_persona(["msg-1"])
        mgr.reset(SESSION_ID)

        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1", "read", "read"), persona)
        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1", "Reading file.py", "read"), persona)

        persona.ychat.add_message.assert_called_once()  # only 1 message created
        assert mgr._sessions[SESSION_ID].tool_call_message_ids["tc-1"] == "msg-1"
        assert mgr._sessions[SESSION_ID].tool_calls["tc-1"].title == "Reading file.py"
        assert mgr._sessions[SESSION_ID].current_message_type == "tool_call"
        # Flush called twice (once per start), both targeting the same message
        assert persona.ychat.update_message.call_count == 2

    def test_repeated_start_always_sets_message_type(self):
        """current_message_type must be 'tool_call' even for a repeated start."""
        mgr = ToolCallManager()
        persona = make_persona(["msg-tc", "msg-text"])
        mgr.reset(SESSION_ID)

        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1", "read"), persona)
        mgr.get_or_create_text_message(SESSION_ID, persona)  # switches to "text"
        assert mgr._sessions[SESSION_ID].current_message_type == "text"

        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1", "Reading file.py"), persona)
        assert mgr._sessions[SESSION_ID].current_message_type == "tool_call"


class TestHandleProgress:
    def test_updates_existing_tool_call_status(self):
        mgr = ToolCallManager()
        persona = make_persona(["msg-1"])
        mgr.reset(SESSION_ID)
        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1"), persona)

        mgr.handle_progress(
            SESSION_ID, make_tool_call_progress("tc-1", status="completed"), persona
        )

        tc = mgr._sessions[SESSION_ID].tool_calls["tc-1"]
        assert tc.status == "completed"

    def test_creates_tool_call_and_message_if_orphaned(self):
        """handle_progress for an unseen tool_call_id creates state + message."""
        mgr = ToolCallManager()
        persona = make_persona(["msg-orphan"])
        mgr.reset(SESSION_ID)

        mgr.handle_progress(
            SESSION_ID, make_tool_call_progress("tc-orphan", status="completed"), persona
        )

        assert "tc-orphan" in mgr._sessions[SESSION_ID].tool_calls
        assert mgr._sessions[SESSION_ID].tool_call_message_ids["tc-orphan"] == "msg-orphan"
        persona.ychat.add_message.assert_called_once()

    def test_flushes_to_tool_calls_own_message(self):
        """Progress must update the tool call's specific message, not the current one."""
        mgr = ToolCallManager()
        persona = make_persona(["msg-tc1", "msg-tc2"])
        mgr.reset(SESSION_ID)

        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1"), persona)
        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-2"), persona)
        persona.ychat.update_message.reset_mock()

        # Progress for tc-1 should update msg-tc1, not msg-tc2
        mgr.handle_progress(
            SESSION_ID, make_tool_call_progress("tc-1", status="completed"), persona
        )

        flushed_msg = persona.ychat.update_message.call_args[0][0]
        # get_message was called with msg-tc1 (tc-1's message)
        persona.ychat.get_message.assert_called_with("msg-tc1")

    def test_non_serializable_raw_output_is_stringified(self):
        mgr = ToolCallManager()
        persona = make_persona(["msg-1"])
        mgr.reset(SESSION_ID)
        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1"), persona)

        class CustomObj:
            def __str__(self):
                return "custom-string"

        update = make_tool_call_progress("tc-1", raw_output=CustomObj())
        mgr.handle_progress(SESSION_ID, update, persona)

        tc = mgr._sessions[SESSION_ID].tool_calls["tc-1"]
        assert tc.raw_output == "custom-string"

    def test_serializable_raw_output_is_preserved(self):
        mgr = ToolCallManager()
        persona = make_persona(["msg-1"])
        mgr.reset(SESSION_ID)
        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1"), persona)

        update = make_tool_call_progress("tc-1", raw_output={"key": "value"})
        mgr.handle_progress(SESSION_ID, update, persona)

        tc = mgr._sessions[SESSION_ID].tool_calls["tc-1"]
        assert tc.raw_output == {"key": "value"}

    def test_orphaned_progress_sets_message_type(self):
        """Orphaned progress must set current_message_type so next text creates a new message."""
        mgr = ToolCallManager()
        persona = make_persona(["msg-orphan", "msg-text"])
        mgr.reset(SESSION_ID)

        mgr.handle_progress(
            SESSION_ID, make_tool_call_progress("tc-orphan", status="completed"), persona
        )
        assert mgr._sessions[SESSION_ID].current_message_type == "tool_call"

        # Next text chunk must create a new message, not reuse the tool call's message
        text_msg = mgr.get_or_create_text_message(SESSION_ID, persona)
        assert text_msg == "msg-text"
        assert persona.ychat.add_message.call_count == 2

    def test_empty_status_treated_as_none(self):
        """Empty string status must not overwrite existing status."""
        mgr = ToolCallManager()
        persona = make_persona(["msg-1"])
        mgr.reset(SESSION_ID)
        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1"), persona)

        update = make_tool_call_progress("tc-1", status="")
        mgr.handle_progress(SESSION_ID, update, persona)

        tc = mgr._sessions[SESSION_ID].tool_calls["tc-1"]
        assert tc.status == "in_progress"  # unchanged from handle_start

    def test_raw_input_passed_through(self):
        mgr = ToolCallManager()
        persona = make_persona()
        mgr.reset(SESSION_ID)
        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1"), persona)

        update = make_tool_call_progress("tc-1", status="completed", raw_input={"command": "ls"})
        mgr.handle_progress(SESSION_ID, update, persona)

        tc = mgr._sessions[SESSION_ID].tool_calls["tc-1"]
        assert tc.raw_input == {"command": "ls"}

    def test_non_serializable_raw_input_is_stringified(self):
        mgr = ToolCallManager()
        persona = make_persona()
        mgr.reset(SESSION_ID)
        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1"), persona)

        class CustomObj:
            def __str__(self):
                return "custom-input"

        update = make_tool_call_progress("tc-1", raw_input=CustomObj())
        mgr.handle_progress(SESSION_ID, update, persona)

        tc = mgr._sessions[SESSION_ID].tool_calls["tc-1"]
        assert tc.raw_input == "custom-input"


class TestFlushToolCall:
    def test_updates_specific_message_metadata(self):
        mgr = ToolCallManager()
        persona = make_persona(["msg-1"])
        mgr.reset(SESSION_ID)
        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1", "Reading"), persona)
        persona.ychat.update_message.reset_mock()

        mgr.flush_tool_call(SESSION_ID, "tc-1", persona)

        persona.ychat.update_message.assert_called_once()
        flushed_msg = persona.ychat.update_message.call_args[0][0]
        assert len(flushed_msg.metadata["tool_calls"]) == 1
        assert flushed_msg.metadata["tool_calls"][0]["tool_call_id"] == "tc-1"

    def test_noop_when_session_missing(self):
        mgr = ToolCallManager()
        persona = make_persona()
        mgr.flush_tool_call("nonexistent", "tc-1", persona)
        persona.ychat.get_message.assert_not_called()

    def test_noop_when_message_id_missing(self):
        mgr = ToolCallManager()
        persona = make_persona()
        mgr.reset(SESSION_ID)
        mgr.flush_tool_call(SESSION_ID, "unknown-tc", persona)
        persona.ychat.get_message.assert_not_called()

    def test_noop_when_yjs_message_deleted(self):
        mgr = ToolCallManager()
        persona = make_persona(["msg-1"])
        persona.ychat.get_message.return_value = None
        mgr.reset(SESSION_ID)
        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1"), persona)
        persona.ychat.update_message.reset_mock()

        mgr.flush_tool_call(SESSION_ID, "tc-1", persona)
        persona.ychat.update_message.assert_not_called()


class TestStateMachine:
    def test_text_then_tool_call_then_text(self):
        """text → tool_call → text creates 3 messages."""
        mgr = ToolCallManager()
        persona = make_persona(["msg-1", "msg-2", "msg-3"])
        mgr.reset(SESSION_ID)

        # Text
        mid1 = mgr.get_or_create_text_message(SESSION_ID, persona)
        assert mgr._sessions[SESSION_ID].current_message_type == "text"

        # Tool call
        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1"), persona)
        assert mgr._sessions[SESSION_ID].current_message_type == "tool_call"

        # Text again (should create new message)
        mid3 = mgr.get_or_create_text_message(SESSION_ID, persona)
        assert mgr._sessions[SESSION_ID].current_message_type == "text"

        assert persona.ychat.add_message.call_count == 3
        assert mid1 == "msg-1"
        assert mgr._sessions[SESSION_ID].tool_call_message_ids["tc-1"] == "msg-2"
        assert mid3 == "msg-3"
        assert mgr.get_all_message_ids(SESSION_ID) == ["msg-1", "msg-2", "msg-3"]

    def test_text_then_two_tool_calls(self):
        """text → tc1 → tc2 creates 3 messages (each TC separate)."""
        mgr = ToolCallManager()
        persona = make_persona(["msg-1", "msg-2", "msg-3"])
        mgr.reset(SESSION_ID)

        mgr.get_or_create_text_message(SESSION_ID, persona)
        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1"), persona)
        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-2"), persona)

        assert persona.ychat.add_message.call_count == 3
        session = mgr._sessions[SESSION_ID]
        assert session.tool_call_message_ids["tc-1"] == "msg-2"
        assert session.tool_call_message_ids["tc-2"] == "msg-3"

    def test_tool_call_only(self):
        """Tool call without preceding text creates 1 message with empty body."""
        mgr = ToolCallManager()
        persona = make_persona(["msg-1"])
        mgr.reset(SESSION_ID)

        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1"), persona)

        assert persona.ychat.add_message.call_count == 1
        # Verify message was created with empty body
        add_call = persona.ychat.add_message.call_args
        assert add_call[0][0].body == ""

    def test_consecutive_text_chunks_reuse_message(self):
        """Multiple text chunks in sequence append to the same message."""
        mgr = ToolCallManager()
        persona = make_persona(["msg-1"])
        mgr.reset(SESSION_ID)

        mid1 = mgr.get_or_create_text_message(SESSION_ID, persona)
        mid2 = mgr.get_or_create_text_message(SESSION_ID, persona)
        mid3 = mgr.get_or_create_text_message(SESSION_ID, persona)

        assert mid1 == mid2 == mid3 == "msg-1"
        persona.ychat.add_message.assert_called_once()

    def test_progress_targets_correct_message_after_state_change(self):
        """Progress for tc-1 arriving after tc-2 starts updates tc-1's message."""
        mgr = ToolCallManager()
        persona = make_persona(["msg-tc1", "msg-tc2"])
        mgr.reset(SESSION_ID)

        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1"), persona)
        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-2"), persona)

        # Current message is msg-tc2, but progress for tc-1 should target msg-tc1
        persona.ychat.get_message.reset_mock()
        mgr.handle_progress(
            SESSION_ID, make_tool_call_progress("tc-1", status="completed"), persona
        )

        persona.ychat.get_message.assert_called_with("msg-tc1")


class TestFullFlow:
    def test_start_then_progress(self):
        """ToolCallStart → ToolCallProgress → tool call state reflects final status."""
        mgr = ToolCallManager()
        persona = make_persona(["msg-1"])
        mgr.reset(SESSION_ID)

        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1", "Reading file.py", "read"), persona)
        mgr.handle_progress(SESSION_ID, make_tool_call_progress("tc-1", status="completed"), persona)

        tc = mgr._sessions[SESSION_ID].tool_calls["tc-1"]
        assert tc.status == "completed"

    def test_multiple_tool_calls_create_separate_messages(self):
        """Multiple tool calls in one session each get their own message."""
        mgr = ToolCallManager()
        persona = make_persona([f"msg-{i}" for i in range(3)])
        mgr.reset(SESSION_ID)

        for i in range(3):
            mgr.handle_start(SESSION_ID, make_tool_call_start(f"tc-{i}", f"Task {i}"), persona)
            mgr.handle_progress(SESSION_ID, make_tool_call_progress(f"tc-{i}", status="completed"), persona)

        assert persona.ychat.add_message.call_count == 3
        assert len(mgr.get_all_message_ids(SESSION_ID)) == 3
        # Each tool call maps to a different message
        session = mgr._sessions[SESSION_ID]
        message_ids = list(session.tool_call_message_ids.values())
        assert len(set(message_ids)) == 3  # all unique

    def test_yjs_write_count_per_tool_call(self):
        """Each handle_start/progress produces exactly one update_message call."""
        mgr = ToolCallManager()
        persona = make_persona(["msg-1", "msg-2"])
        mgr.reset(SESSION_ID)

        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1"), persona)
        assert persona.ychat.add_message.call_count == 1
        assert persona.ychat.update_message.call_count == 1

        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-2"), persona)
        assert persona.ychat.add_message.call_count == 2
        assert persona.ychat.update_message.call_count == 2
