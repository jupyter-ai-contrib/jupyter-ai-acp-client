from unittest.mock import MagicMock

from jupyter_ai_acp_client.tool_call_manager import SessionState, ToolCallManager


def make_persona(message_id: str = "msg-1"):
    """Return a mock persona whose ychat.add_message returns message_id."""
    persona = MagicMock()
    persona.id = "persona-id"
    persona.ychat.add_message.return_value = message_id
    persona.ychat.get_message.return_value = MagicMock()
    return persona


def make_tool_call_start(
    tool_call_id: str = "tc-1",
    title: str = "Reading file.py",
    kind: str = "read",
    locations=None,
):
    update = MagicMock()
    update.tool_call_id = tool_call_id
    update.title = title
    update.kind = kind
    update.locations = locations
    return update


def make_tool_call_progress(
    tool_call_id: str = "tc-1",
    title: str | None = None,
    kind: str | None = None,
    status: str | None = "completed",
    raw_output=None,
    locations=None,
):
    update = MagicMock()
    update.tool_call_id = tool_call_id
    update.title = title
    update.kind = kind
    update.status = status
    update.raw_output = raw_output
    update.locations = locations
    return update


SESSION_ID = "session-abc"


class TestEnsureSession:
    def test_creates_session_when_absent(self):
        mgr = ToolCallManager()
        state = mgr._ensure_session(SESSION_ID)
        assert isinstance(state, SessionState)
        assert SESSION_ID in mgr._sessions

    def test_returns_same_object_on_repeat_calls(self):
        mgr = ToolCallManager()
        a = mgr._ensure_session(SESSION_ID)
        b = mgr._ensure_session(SESSION_ID)
        assert a is b

    def test_does_not_overwrite_existing(self):
        mgr = ToolCallManager()
        mgr._sessions[SESSION_ID] = SessionState(message_id="existing")
        state = mgr._ensure_session(SESSION_ID)
        assert state.message_id == "existing"


class TestReset:
    def test_creates_fresh_session(self):
        mgr = ToolCallManager()
        mgr.reset(SESSION_ID)
        assert SESSION_ID in mgr._sessions
        assert mgr._sessions[SESSION_ID].tool_calls == {}
        assert mgr._sessions[SESSION_ID].message_id is None

    def test_clears_existing_state(self):
        mgr = ToolCallManager()
        persona = make_persona()
        mgr.handle_start(SESSION_ID, make_tool_call_start(), persona)
        assert mgr._sessions[SESSION_ID].tool_calls  # non-empty

        mgr.reset(SESSION_ID)
        assert mgr._sessions[SESSION_ID].tool_calls == {}
        assert mgr._sessions[SESSION_ID].message_id is None


class TestGetMessageId:
    def test_returns_none_for_unknown_session(self):
        mgr = ToolCallManager()
        assert mgr.get_message_id(SESSION_ID) is None

    def test_returns_none_before_message_created(self):
        mgr = ToolCallManager()
        mgr.reset(SESSION_ID)
        assert mgr.get_message_id(SESSION_ID) is None

    def test_returns_message_id_after_creation(self):
        mgr = ToolCallManager()
        persona = make_persona("msg-42")
        mgr.handle_start(SESSION_ID, make_tool_call_start(), persona)
        assert mgr.get_message_id(SESSION_ID) == "msg-42"


class TestGetOrCreateMessage:
    def test_creates_yjs_message_on_first_call(self):
        mgr = ToolCallManager()
        persona = make_persona("msg-1")
        mgr.reset(SESSION_ID)

        msg_id = mgr.get_or_create_message(SESSION_ID, persona)

        assert msg_id == "msg-1"
        persona.ychat.add_message.assert_called_once()

    def test_returns_existing_message_id_on_subsequent_calls(self):
        mgr = ToolCallManager()
        persona = make_persona("msg-1")
        mgr.reset(SESSION_ID)

        first = mgr.get_or_create_message(SESSION_ID, persona)
        second = mgr.get_or_create_message(SESSION_ID, persona)

        assert first == second == "msg-1"
        # add_message called only once — not on the second call
        persona.ychat.add_message.assert_called_once()

    def test_sets_awareness_on_creation(self):
        mgr = ToolCallManager()
        persona = make_persona("msg-1")
        mgr.reset(SESSION_ID)

        mgr.get_or_create_message(SESSION_ID, persona)

        persona.awareness.set_local_state_field.assert_called_with("isWriting", "msg-1")

    def test_does_not_flush_tool_calls_on_creation(self):
        """
        Dead early-flush was removed: get_or_create_message must NOT write
        tool calls to the message — callers (handle_start/progress) own that.
        """
        mgr = ToolCallManager()
        persona = make_persona("msg-1")
        mgr.reset(SESSION_ID)
        # Manually put a tool call in the session
        mgr._sessions[SESSION_ID].tool_calls["tc-x"] = MagicMock()

        mgr.get_or_create_message(SESSION_ID, persona)

        # update_message should NOT have been called at this point
        persona.ychat.update_message.assert_not_called()


class TestSerialize:
    def test_returns_empty_list_for_unknown_session(self):
        mgr = ToolCallManager()
        assert mgr.serialize("no-such-session") == []

    def test_returns_empty_list_before_any_tool_calls(self):
        mgr = ToolCallManager()
        mgr.reset(SESSION_ID)
        assert mgr.serialize(SESSION_ID) == []

    def test_returns_serialized_tool_calls_after_handle_start(self):
        mgr = ToolCallManager()
        persona = make_persona()
        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1", "Reading file.py", "read"), persona)

        result = mgr.serialize(SESSION_ID)

        assert len(result) == 1
        assert result[0]["tool_call_id"] == "tc-1"
        assert result[0]["status"] == "in_progress"


class TestHandleStart:
    def test_creates_session_if_absent(self):
        mgr = ToolCallManager()
        persona = make_persona()
        mgr.handle_start(SESSION_ID, make_tool_call_start(), persona)
        assert SESSION_ID in mgr._sessions

    def test_adds_tool_call_to_session(self):
        mgr = ToolCallManager()
        persona = make_persona()
        mgr.reset(SESSION_ID)

        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1", "Reading", "read"), persona)

        assert "tc-1" in mgr._sessions[SESSION_ID].tool_calls
        assert mgr._sessions[SESSION_ID].tool_calls["tc-1"].status == "in_progress"

    def test_calls_add_message_exactly_once(self):
        mgr = ToolCallManager()
        persona = make_persona("msg-1")
        mgr.reset(SESSION_ID)

        mgr.handle_start(SESSION_ID, make_tool_call_start(), persona)

        persona.ychat.add_message.assert_called_once()

    def test_second_handle_start_reuses_message(self):
        mgr = ToolCallManager()
        persona = make_persona("msg-1")
        mgr.reset(SESSION_ID)

        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1"), persona)
        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-2"), persona)

        # add_message only called once — second start reuses existing message
        persona.ychat.add_message.assert_called_once()
        assert "tc-1" in mgr._sessions[SESSION_ID].tool_calls
        assert "tc-2" in mgr._sessions[SESSION_ID].tool_calls

    def test_flushes_to_message_after_state_update(self):
        mgr = ToolCallManager()
        persona = make_persona("msg-1")
        mgr.reset(SESSION_ID)

        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1"), persona)

        # update_message called once — the _flush_to_message call
        persona.ychat.update_message.assert_called_once()

    def test_locations_extracted_from_update(self):
        mgr = ToolCallManager()
        persona = make_persona()
        loc = MagicMock()
        loc.path = "/some/file.py"
        update = make_tool_call_start("tc-1", "Reading", "read", locations=[loc])

        mgr.handle_start(SESSION_ID, update, persona)

        tc = mgr._sessions[SESSION_ID].tool_calls["tc-1"]
        assert tc.locations == ["/some/file.py"]

    def test_empty_kind_treated_as_none(self):
        mgr = ToolCallManager()
        persona = make_persona()
        update = make_tool_call_start("tc-1", "Working", kind="")

        mgr.handle_start(SESSION_ID, update, persona)

        tc = mgr._sessions[SESSION_ID].tool_calls["tc-1"]
        assert tc.kind is None

    def test_does_not_bleed_into_another_session(self):
        """Two concurrent sessions must be fully isolated."""
        mgr = ToolCallManager()
        persona_a = make_persona("msg-a")
        persona_b = make_persona("msg-b")
        session_a = "session-A"
        session_b = "session-B"

        mgr.handle_start(session_a, make_tool_call_start("tc-a", "Task A"), persona_a)
        mgr.handle_start(session_b, make_tool_call_start("tc-b", "Task B"), persona_b)

        assert "tc-a" in mgr._sessions[session_a].tool_calls
        assert "tc-b" not in mgr._sessions[session_a].tool_calls
        assert "tc-b" in mgr._sessions[session_b].tool_calls
        assert "tc-a" not in mgr._sessions[session_b].tool_calls
        assert mgr.get_message_id(session_a) == "msg-a"
        assert mgr.get_message_id(session_b) == "msg-b"


class TestHandleProgress:
    def test_updates_existing_tool_call_status(self):
        mgr = ToolCallManager()
        persona = make_persona()
        mgr.reset(SESSION_ID)
        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1"), persona)

        mgr.handle_progress(
            SESSION_ID, make_tool_call_progress("tc-1", status="completed"), persona
        )

        tc = mgr._sessions[SESSION_ID].tool_calls["tc-1"]
        assert tc.status == "completed"

    def test_creates_tool_call_if_not_seen_before(self):
        """handle_progress for an unseen tool_call_id must create one (protocol allows this)."""
        mgr = ToolCallManager()
        persona = make_persona()
        mgr.reset(SESSION_ID)

        mgr.handle_progress(
            SESSION_ID, make_tool_call_progress("tc-orphan", status="completed"), persona
        )

        assert "tc-orphan" in mgr._sessions[SESSION_ID].tool_calls

    def test_flushes_to_message_after_state_update(self):
        mgr = ToolCallManager()
        persona = make_persona("msg-1")
        mgr.reset(SESSION_ID)
        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1"), persona)
        persona.ychat.update_message.reset_mock()  # clear the flush from handle_start

        mgr.handle_progress(
            SESSION_ID, make_tool_call_progress("tc-1", status="completed"), persona
        )

        persona.ychat.update_message.assert_called_once()

    def test_non_serializable_raw_output_is_stringified(self):
        mgr = ToolCallManager()
        persona = make_persona()
        mgr.reset(SESSION_ID)
        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1"), persona)

        # Pass an object that is not str/int/float/bool/list/dict
        class CustomObj:
            def __str__(self):
                return "custom-string"

        update = make_tool_call_progress("tc-1", raw_output=CustomObj())
        mgr.handle_progress(SESSION_ID, update, persona)

        tc = mgr._sessions[SESSION_ID].tool_calls["tc-1"]
        assert tc.raw_output == "custom-string"

    def test_serializable_raw_output_is_preserved(self):
        mgr = ToolCallManager()
        persona = make_persona()
        mgr.reset(SESSION_ID)
        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1"), persona)

        update = make_tool_call_progress("tc-1", raw_output={"key": "value"})
        mgr.handle_progress(SESSION_ID, update, persona)

        tc = mgr._sessions[SESSION_ID].tool_calls["tc-1"]
        assert tc.raw_output == {"key": "value"}

    def test_empty_status_treated_as_none(self):
        """Empty string status must be converted to None, not stored as ''."""
        mgr = ToolCallManager()
        persona = make_persona()
        mgr.reset(SESSION_ID)
        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1"), persona)

        update = make_tool_call_progress("tc-1", status="")
        mgr.handle_progress(SESSION_ID, update, persona)

        # status="" is falsy → treated as None → update_tool_call_from_progress
        # receives status=None → does not overwrite existing status
        tc = mgr._sessions[SESSION_ID].tool_calls["tc-1"]
        assert tc.status == "in_progress"  # unchanged from handle_start


class TestFullFlow:
    def test_start_then_progress_then_serialize(self):
        """End-to-end: ToolCallStart → ToolCallProgress → serialize reflects final state."""
        mgr = ToolCallManager()
        persona = make_persona("msg-1")
        mgr.reset(SESSION_ID)

        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1", "Reading file.py", "read"), persona)
        mgr.handle_progress(SESSION_ID, make_tool_call_progress("tc-1", status="completed"), persona)

        result = mgr.serialize(SESSION_ID)
        assert len(result) == 1
        assert result[0]["tool_call_id"] == "tc-1"
        assert result[0]["status"] == "completed"

    def test_multiple_tool_calls_in_sequence(self):
        """Multiple tool calls in one session — all tracked independently."""
        mgr = ToolCallManager()
        persona = make_persona("msg-1")
        mgr.reset(SESSION_ID)

        for i in range(3):
            mgr.handle_start(SESSION_ID, make_tool_call_start(f"tc-{i}", f"Task {i}"), persona)
            mgr.handle_progress(SESSION_ID, make_tool_call_progress(f"tc-{i}", status="completed"), persona)

        result = mgr.serialize(SESSION_ID)
        assert len(result) == 3
        assert all(r["status"] == "completed" for r in result)
        # add_message called only once for the whole session
        persona.ychat.add_message.assert_called_once()

    def test_reset_between_turns_clears_state(self):
        """After reset, previous turn's tool calls are gone."""
        mgr = ToolCallManager()
        persona = make_persona("msg-1")
        mgr.reset(SESSION_ID)
        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1"), persona)

        # Simulate new turn
        persona = make_persona("msg-2")
        mgr.reset(SESSION_ID)

        assert mgr.get_message_id(SESSION_ID) is None
        assert mgr.serialize(SESSION_ID) == []

    def test_flush_skipped_when_yjs_returns_no_message(self):
        """
        _flush_to_message must not crash when ychat.get_message() returns None
        (e.g. the message was deleted between creation and flush).
        """
        mgr = ToolCallManager()
        persona = make_persona("msg-1")
        persona.ychat.get_message.return_value = None  # Yjs returns nothing
        mgr.reset(SESSION_ID)

        # Should not raise; the `if msg:` guard should silently skip the write
        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1"), persona)

        persona.ychat.update_message.assert_not_called()

    def test_yjs_write_count_for_two_tool_calls(self):
        """
        Verify there is exactly one update_message call per handle_start/progress,
        not two (the dead early-flush was removed).
        """
        mgr = ToolCallManager()
        persona = make_persona("msg-1")
        mgr.reset(SESSION_ID)

        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-1"), persona)
        # First tool call: add_message (1) + update_message from _flush (1)
        assert persona.ychat.add_message.call_count == 1
        assert persona.ychat.update_message.call_count == 1

        mgr.handle_start(SESSION_ID, make_tool_call_start("tc-2"), persona)
        # Second tool call: no new add_message, one more update_message
        assert persona.ychat.add_message.call_count == 1
        assert persona.ychat.update_message.call_count == 2
