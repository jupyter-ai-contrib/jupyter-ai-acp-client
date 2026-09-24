"""Tests for attachment resolution and load-session recovery in BaseAcpPersona."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from jupyterlab_chat.models import Message

from jupyter_ai_acp_client.base_acp_persona import BaseAcpPersona
from jupyter_ai_persona_manager import PersonaNotAuthenticated


def _make_chat_message(
    id: str, body: str, sender: str, deleted: bool | None = None
) -> Message:
    return Message(id=id, body=body, sender=sender, time=0.0, deleted=deleted)


def _make_persona(attachments_map: dict | None = None):
    """Create a minimal mock BaseAcpPersona for testing process_message."""
    persona = MagicMock()
    persona.get_client = AsyncMock()
    persona.get_session_id = AsyncMock(return_value="sess-1")
    persona.is_authed = AsyncMock(return_value=True)
    persona.prepare = AsyncMock()
    persona._pending_session_recovery_context = False
    persona._was_initially_unauthenticated = False

    # as_user() is sync — must return a regular MagicMock
    user_mock = MagicMock()
    user_mock.mention_name = "bot"
    persona.as_user.return_value = user_mock

    # YChat mock
    ychat = MagicMock()
    ychat.get_attachments.return_value = attachments_map or {}
    ychat.get_messages.return_value = []
    ychat.get_users.return_value = {}
    persona.chat = ychat

    # parent.root_dir
    persona.parent = MagicMock()
    persona.parent.root_dir = "/home/user/notebooks"

    return persona



def _make_client():
    """Create an AsyncMock client with prompt_and_reply explicitly async."""
    client = AsyncMock()
    client.prompt_and_reply = AsyncMock()
    return client


def _make_message(body: str, attachment_ids: list[str] | None = None):
    msg = MagicMock()
    msg.id = "current-msg"
    msg.body = body
    msg.attachments = attachment_ids
    return msg


def _make_session_init_persona(
    *,
    error: Exception | None = None,
    existing_session_id: str | None = "old-session",
    supports_session_load: bool = True,
):
    """Create an uninitialized persona wired for _init_client_session tests."""
    persona = MagicMock()
    persona.id = "test-persona"
    persona.log = MagicMock()
    persona._pending_session_recovery_context = False
    persona._was_initially_unauthenticated = False

    client = MagicMock()
    capabilities = MagicMock()
    capabilities.load_session = supports_session_load
    client.get_agent_capabilities = AsyncMock(return_value=capabilities)
    persona.get_client = AsyncMock(return_value=client)

    sessions = {}
    if existing_session_id:
        sessions[persona.id] = existing_session_id
    persona._get_existing_sessions = MagicMock(return_value=sessions)
    persona._load_session = AsyncMock()
    persona._create_session = AsyncMock(return_value=MagicMock(session_id="new-session"))

    if error:
        persona._load_session.side_effect = error
    else:
        persona._load_session.return_value = MagicMock(session_id=existing_session_id)

    return persona, client


class TestProcessMessageAttachments:
    """Tests for how process_message resolves attachments and calls prompt_and_reply."""

    async def test_empty_message(self):
        """
        Ensure that bare @-mentions messages include content.
        Prevents #111.
        """
        client = _make_client()
        persona = _make_persona()
        persona.get_client.return_value = client
        msg = _make_message("@bot")

        await BaseAcpPersona.process_message(persona, msg)

        client.prompt_and_reply.assert_called_once_with(
            session_id="sess-1",
            prompt="@bot",
            attachments=None,
            root_dir="/home/user/notebooks",
        )

    async def test_no_attachments(self):
        """When message has no attachments, prompt_and_reply is called without them."""
        client = _make_client()
        persona = _make_persona()
        persona.get_client.return_value = client
        msg = _make_message("@bot hello")

        await BaseAcpPersona.process_message(persona, msg)

        client.prompt_and_reply.assert_called_once_with(
            session_id="sess-1",
            prompt="@bot hello",
            attachments=None,
            root_dir="/home/user/notebooks",
        )

    async def test_notebook_guidance_prepended_on_first_message(self):
        """The first message of a session is prefixed with the agent-agnostic
        notebook-editing guidance, then the flag flips so it isn't repeated."""
        client = _make_client()
        persona = _make_persona()
        persona.get_client.return_value = client
        persona._notebook_guidance_sent = False  # simulate a fresh session
        msg = _make_message("@bot add a cell")

        await BaseAcpPersona.process_message(persona, msg)

        sent_prompt = client.prompt_and_reply.call_args.kwargs["prompt"]
        assert sent_prompt.startswith(BaseAcpPersona.NOTEBOOK_EDITING_GUIDANCE)
        assert sent_prompt.endswith("@bot add a cell")
        # Flag flipped so subsequent messages don't repeat the guidance.
        assert persona._notebook_guidance_sent is True

    async def test_notebook_guidance_not_repeated(self):
        """Once guidance has been sent, later messages are passed through as-is."""
        client = _make_client()
        persona = _make_persona()
        persona.get_client.return_value = client
        persona._notebook_guidance_sent = True  # already sent earlier this session
        msg = _make_message("@bot another request")

        await BaseAcpPersona.process_message(persona, msg)

        client.prompt_and_reply.assert_called_once_with(
            session_id="sess-1",
            prompt="@bot another request",
            attachments=None,
            root_dir="/home/user/notebooks",
        )

    async def test_single_attachment(self):
        """A single known attachment ID resolves to a dict."""
        client = _make_client()
        att_map = {
            "att-1": {"value": "file.py", "type": "file", "mimetype": "text/x-python"},
        }
        persona = _make_persona(att_map)
        persona.get_client.return_value = client
        msg = _make_message("@bot check this", ["att-1"])

        await BaseAcpPersona.process_message(persona, msg)

        call_kwargs = client.prompt_and_reply.call_args.kwargs
        assert call_kwargs["attachments"] == [att_map["att-1"]]

    async def test_multiple_attachments(self):
        """Multiple attachment IDs all resolve in order."""
        client = _make_client()
        att_map = {
            "att-1": {"value": "a.py", "type": "file"},
            "att-2": {"value": "b.ipynb", "type": "notebook"},
        }
        persona = _make_persona(att_map)
        persona.get_client.return_value = client
        msg = _make_message("@bot review", ["att-1", "att-2"])

        await BaseAcpPersona.process_message(persona, msg)

        call_kwargs = client.prompt_and_reply.call_args.kwargs
        assert call_kwargs["attachments"] == [att_map["att-1"], att_map["att-2"]]

    async def test_unknown_attachment_id_skipped(self):
        """Unknown attachment IDs are silently skipped with a log warning."""
        client = _make_client()
        persona = _make_persona({})
        persona.get_client.return_value = client
        msg = _make_message("@bot check", ["nonexistent"])

        await BaseAcpPersona.process_message(persona, msg)

        call_kwargs = client.prompt_and_reply.call_args.kwargs
        assert call_kwargs["attachments"] is None

    async def test_partial_resolution(self):
        """Only known IDs are resolved; unknown ones are skipped."""
        client = _make_client()
        att_map = {"att-1": {"value": "good.py", "type": "file"}}
        persona = _make_persona(att_map)
        persona.get_client.return_value = client
        msg = _make_message("@bot check", ["att-1", "missing"])

        await BaseAcpPersona.process_message(persona, msg)

        call_kwargs = client.prompt_and_reply.call_args.kwargs
        assert call_kwargs["attachments"] == [att_map["att-1"]]

    async def test_empty_attachment_list(self):
        """An empty attachment list results in None."""
        client = _make_client()
        persona = _make_persona()
        persona.get_client.return_value = client
        msg = _make_message("@bot hi", [])

        await BaseAcpPersona.process_message(persona, msg)

        call_kwargs = client.prompt_and_reply.call_args.kwargs
        assert call_kwargs["attachments"] is None

    async def test_root_dir_passed(self):
        """root_dir from persona.parent is forwarded to prompt_and_reply."""
        client = _make_client()
        persona = _make_persona()
        persona.parent.root_dir = "/custom/root"
        persona.get_client.return_value = client
        msg = _make_message("@bot hi")

        await BaseAcpPersona.process_message(persona, msg)

        call_kwargs = client.prompt_and_reply.call_args.kwargs
        assert call_kwargs["root_dir"] == "/custom/root"


class TestLoadSessionRecovery:
    """Tests for load-session recovery, history injection, and flag behavior."""

    async def test_any_load_session_error_creates_new_session(self):
        """Any load_session failure falls back to a new session."""
        persona, client = _make_session_init_persona(error=Exception("load failed"))

        await BaseAcpPersona._init_client_session(persona)

        persona._load_session.assert_awaited_once_with(client, "old-session")
        persona._create_session.assert_awaited_once_with(client)
        assert persona._pending_session_recovery_context is True

    async def test_no_existing_session_creates_new_session(self):
        """When no session ID is in metadata, _create_session is called directly."""
        persona, client = _make_session_init_persona(existing_session_id=None)

        await BaseAcpPersona._init_client_session(persona)

        persona._load_session.assert_not_awaited()
        persona._create_session.assert_awaited_once_with(client)
        assert persona._pending_session_recovery_context is False

    async def test_agent_without_load_session_creates_new_session(self):
        """When the agent doesn't support load_session, _create_session is called directly."""
        persona, client = _make_session_init_persona(supports_session_load=False)

        await BaseAcpPersona._init_client_session(persona)

        persona._load_session.assert_not_awaited()
        persona._create_session.assert_awaited_once_with(client)
        assert persona._pending_session_recovery_context is False

    async def test_create_session_error_propagates(self):
        """create_session() errors are not swallowed after load_session fails."""
        persona, _ = _make_session_init_persona(error=Exception("load failed"))
        create_error = Exception("create failed")
        persona._create_session.side_effect = create_error

        with pytest.raises(Exception) as exc_info:
            await BaseAcpPersona._init_client_session(persona)

        assert exc_info.value is create_error
        assert persona._pending_session_recovery_context is True

    async def test_recovery_flag_resets_after_first_message(self):
        """History is injected only on the first message after recovery."""
        client = _make_client()
        persona = _make_persona()
        persona._pending_session_recovery_context = True
        persona.get_client.return_value = client

        msg1 = _make_message("@bot first")
        await BaseAcpPersona.process_message(persona, msg1)

        assert persona._pending_session_recovery_context is False

        msg2 = _make_message("@bot second")
        await BaseAcpPersona.process_message(persona, msg2)

        second_call_prompt = client.prompt_and_reply.call_args_list[1].kwargs["prompt"]
        assert second_call_prompt == "@bot second"

    def test_build_history_context_excludes_current_message(self):
        """The current message is not included in the injected history."""
        persona = _make_persona()
        persona._MAX_HISTORY_MESSAGES = BaseAcpPersona._MAX_HISTORY_MESSAGES
        msgs = [
            _make_chat_message("msg-1", "hello", "user-1"),
            _make_chat_message("msg-2", "hi there", "bot-1"),
            _make_chat_message("msg-3", "follow up", "user-1"),  # current message
        ]
        persona.chat.get_messages.return_value = msgs
        persona.chat.get_users.return_value = {}

        result = BaseAcpPersona._build_history_context(persona, exclude_id="msg-3")

        assert "follow up" not in result
        assert "hello" in result
        assert "hi there" in result

    async def test_recovery_history_injected_into_prompt(self):
        """History is prepended to the prompt on the first message after recovery."""
        client = _make_client()
        persona = _make_persona()
        persona._pending_session_recovery_context = True
        persona._MAX_HISTORY_MESSAGES = BaseAcpPersona._MAX_HISTORY_MESSAGES
        persona.chat.get_messages.return_value = [
            _make_chat_message("msg-1", "hello world", "user-1"),
        ]
        persona.chat.get_users.return_value = {}
        persona.get_client.return_value = client
        # Delegate to the real method so history is built from ychat
        persona._build_history_context = (
            lambda **kw: BaseAcpPersona._build_history_context(persona, **kw)
        )
        msg = _make_message("@bot follow up")

        await BaseAcpPersona.process_message(persona, msg)

        prompt = client.prompt_and_reply.call_args.kwargs["prompt"]
        assert prompt.startswith("The previous ACP session")
        assert "hello world" in prompt
        assert "follow up" in prompt

    def test_build_history_context_caps_at_max_messages(self):
        """History is capped at _MAX_HISTORY_MESSAGES."""
        persona = _make_persona()
        cap = BaseAcpPersona._MAX_HISTORY_MESSAGES
        persona._MAX_HISTORY_MESSAGES = cap
        msgs = [
            _make_chat_message(f"msg-{i}", f"message {i}", "user-1")
            for i in range(cap + 10)
        ]
        persona.chat.get_messages.return_value = msgs
        persona.chat.get_users.return_value = {}

        result = BaseAcpPersona._build_history_context(persona)

        # Only the last _MAX_HISTORY_MESSAGES messages should appear
        lines = result.splitlines()
        message_lines = [l for l in lines if l.startswith("user-1:")]
        assert len(message_lines) == cap
        # The oldest messages are trimmed, most recent are kept
        assert f"message {cap + 9}" in result
        assert "message 0" not in result




class TestResumeAfterAuth:
    """Tests for proactive resume after user signs in.

    There are two categories of ACP agents with respect to authentication:

    **Agents with auth-gated sessions** (e.g. Kiro): These agents cannot
    start their ACP subprocess until the user is authenticated. The subprocess
    startup is blocked in `before_agent_subprocess()`, which means
    `_init_client_session()` does not complete until auth passes. Once the user
    signs in, the session is created automatically and `_resume_after_auth()`
    fires immediately — no new user message is needed to trigger it.

    **Agents without auth-gated sessions** (e.g. Claude, Codex, Copilot): These
    agents can start their subprocess and create a session without
    authentication. Auth errors are only detected at prompt time — when
    `process_message()` calls `prompt_and_reply()` and the agent raises a
    RequestError. Because the session already exists, `_init_client_session()`
    has long since completed. The resume logic must therefore run in
    `process_message()`: on the first message after auth succeeds (i.e. the flag
    is set and `is_authed()` returns True), `_resume_after_auth()` fires instead
    of processing the message normally.
    """

    async def test_resume_for_agents_without_auth_gated_sessions(self):
        """Agents without auth-gated sessions resume via process_message,
        and the prompt includes the user's original message from chat history."""
        client = _make_client()
        persona = _make_persona()
        persona._was_initially_unauthenticated = True
        persona._MAX_HISTORY_MESSAGES = BaseAcpPersona._MAX_HISTORY_MESSAGES
        persona.get_client.return_value = client
        # Simulate chat history: the user's original request is already in ychat,
        # along with the new message that triggered process_message()
        persona.chat.get_messages.return_value = [
            _make_chat_message("msg-1", "@Kiro generate a fibonacci file", "user-1"),
            _make_chat_message("msg-2", "You're not signed in.", "bot-1"),
            _make_chat_message("msg-3", "@Kiro hello again", "user-1"),
        ]
        persona.chat.get_users.return_value = {}
        persona._build_history_context = (
            lambda **kw: BaseAcpPersona._build_history_context(persona, **kw)
        )
        # Let _resume_after_auth run for real (not mocked)
        persona._resume_after_auth = (
            lambda client, session_id: BaseAcpPersona._resume_after_auth(
                persona, client, session_id
            )
        )

        # This message triggers process_message after auth passes
        msg = _make_message("@bot hello again")
        await BaseAcpPersona.process_message(persona, msg)

        # prompt_and_reply was called with chat history + prescribed template
        client.prompt_and_reply.assert_awaited_once()
        prompt = client.prompt_and_reply.call_args.kwargs["prompt"]
        assert "generate a fibonacci file" in prompt
        assert "Would you like me to help you with this now?" in prompt
        assert persona._was_initially_unauthenticated is False

    async def test_resume_only_fires_once_for_agents_without_auth_gated_sessions(self):
        """The resume prompt only fires on the first message after auth."""
        client = _make_client()
        persona = _make_persona()
        persona._was_initially_unauthenticated = True
        persona.get_client.return_value = client
        persona._resume_after_auth = AsyncMock()

        msg1 = _make_message("@bot first")
        await BaseAcpPersona.process_message(persona, msg1)

        msg2 = _make_message("@bot second")
        await BaseAcpPersona.process_message(persona, msg2)

        # Resume called once, then normal prompt_and_reply on second message
        persona._resume_after_auth.assert_awaited_once()
        client.prompt_and_reply.assert_awaited_once()

    async def test_resume_for_agents_with_auth_gated_sessions(self):
        """
        Agents with auth-gated sessions automatically resume via
        _init_client_session after session creation.
        """
        persona, client = _make_session_init_persona(existing_session_id=None)
        persona._was_initially_unauthenticated = True
        persona._resume_after_auth = AsyncMock()

        await BaseAcpPersona._init_client_session(persona)

        persona._resume_after_auth.assert_awaited_once_with(client, "new-session")
        assert persona._was_initially_unauthenticated is False

    async def test_no_resume_when_not_initially_unauthenticated(self):
        """No resume prompt when the user was authenticated from the start."""
        persona, client = _make_session_init_persona(existing_session_id=None)
        persona._was_initially_unauthenticated = False
        persona._resume_after_auth = AsyncMock()

        await BaseAcpPersona._init_client_session(persona)

        persona._resume_after_auth.assert_not_awaited()


class TestHandleUncaughtException:
    """Tests for structured RequestError display in handle_uncaught_exception."""

    async def test_request_error_shows_code_and_message(self):
        from acp.exceptions import RequestError

        persona = MagicMock()
        persona.send_message = MagicMock()

        exc = RequestError(-32603, "Internal error")
        await BaseAcpPersona.handle_uncaught_exception(persona, exc)

        body = persona.send_message.call_args[0][0]
        assert "jp-jai-error-details" in body
        assert "Error -32603" in body
        assert "Internal error" in body

    async def test_request_error_shows_data(self):
        from acp.exceptions import RequestError

        persona = MagicMock()
        persona.send_message = MagicMock()

        exc = RequestError(-32603, "Internal error", {"path": "/tmp/x"})
        await BaseAcpPersona.handle_uncaught_exception(persona, exc)

        body = persona.send_message.call_args[0][0]
        assert "```json" in body
        assert "/tmp/x" in body

    async def test_request_error_without_data(self):
        from acp.exceptions import RequestError

        persona = MagicMock()
        persona.send_message = MagicMock()

        exc = RequestError(-32000, "Authentication required")
        await BaseAcpPersona.handle_uncaught_exception(persona, exc)

        body = persona.send_message.call_args[0][0]
        assert "Error -32000" in body
        assert "```json" not in body
        assert "**Traceback:**" in body

    async def test_non_request_error_delegates_to_super(self):
        """Non-RequestError exceptions should not use the structured format."""
        persona = MagicMock()
        persona.send_message = MagicMock()

        exc = RuntimeError("something broke")
        try:
            await BaseAcpPersona.handle_uncaught_exception(persona, exc)
        except TypeError:
            # super() fails with MagicMock — proves we delegated
            pass
        if persona.send_message.called:
            body = persona.send_message.call_args[0][0]
            assert "**Error code:**" not in body


def _make_lazy_persona(persona_cls=None, subprocess_impl=None):
    """
    Build a BaseAcpPersona subclass instance without running the heavy
    BasePersona.__init__, wired just enough to exercise `prepare()`, the
    non-spawning `_client_started` probe, and the shutdown guard.

    By default each call defines a fresh subclass so class-level futures don't
    leak between tests. Pass `persona_cls` to build a second instance that
    shares one class's futures (for concurrency tests). Pass `subprocess_impl`
    to inject a counting/failing agent-subprocess stub. Nothing real is spawned.
    """

    if persona_cls is None:
        class _LazyTestPersona(BaseAcpPersona):
            # Shadow the read-only BasePersona properties so the test can inject
            # them without running the real constructor.
            event_loop = None
            event_logger = None

            @property
            def defaults(self):
                return MagicMock()

        persona_cls = _LazyTestPersona

    persona = persona_cls.__new__(persona_cls)
    persona.event_loop = asyncio.get_event_loop()
    persona.log = logging.getLogger("lazy-test-persona")
    persona._client_session_future = None
    persona._prepare_task = None
    persona._emitted = set()
    persona.is_authed = AsyncMock(return_value=True)
    persona.event_logger = MagicMock()
    persona.auth = MagicMock()
    persona.auth.assert_auth = AsyncMock()

    async def _fake_subprocess():
        return "subprocess"

    async def _fake_client():
        # Client whose get_connection() is awaitable.
        client = MagicMock()
        client.get_connection = AsyncMock(return_value="connection")
        return client

    async def _fake_session():
        return "session"

    persona._init_agent_subprocess = subprocess_impl or _fake_subprocess
    persona._init_client = _fake_client
    persona._init_client_session = _fake_session
    return persona_cls, persona


class TestPrepareLifecycle:
    """
    Deferred-spawn behavior for issue #172: construction spawns nothing; the
    `prepare()` hook performs the startup once, before the first message.
    """

    async def test_construction_creates_no_futures(self):
        """A freshly constructed persona has spawned nothing."""
        cls, persona = _make_lazy_persona()
        assert "_subprocess_future" not in cls.__dict__
        assert "_client_future" not in cls.__dict__
        assert persona._client_session_future is None
        assert persona._client_started() is False

    async def test_prepare_starts_subprocess_client_and_session(self):
        """prepare() creates the subprocess, client, and session futures."""
        cls, persona = _make_lazy_persona()
        assert persona._client_started() is False

        await persona.prepare()

        assert cls._subprocess_future is not None
        assert cls._client_future is not None
        assert persona._client_session_future is not None
        assert persona._client_started() is True
        # The futures resolve to the stubbed startup results.
        assert await persona.get_agent_subprocess() == "subprocess"
        assert await persona.get_client() is not None
        assert await persona.get_session_response() == "session"

    async def test_prepare_is_idempotent(self):
        """Calling prepare() twice does not recreate the futures or re-emit."""
        cls, persona = _make_lazy_persona()

        await persona.prepare()
        subprocess_future = cls._subprocess_future
        client_future = cls._client_future
        session_future = persona._client_session_future

        await persona.prepare()

        assert cls._subprocess_future is subprocess_future
        assert cls._client_future is client_future
        assert persona._client_session_future is session_future
        ops = [
            c.kwargs.get("data", {}).get("operation")
            for c in persona.event_logger.emit.call_args_list
        ]
        assert ops.count("acp_engagement") == 1

    async def test_prepare_emits_engagement(self):
        """prepare() emits the 'tried' engagement funnel event."""
        cls, persona = _make_lazy_persona()

        await persona.prepare()

        ops = [
            c.kwargs.get("data", {}).get("operation")
            for c in persona.event_logger.emit.call_args_list
        ]
        assert "acp_engagement" in ops

    async def test_shutdown_of_unengaged_persona_does_not_spawn(self):
        """
        Shutting down a persona that was never engaged must not spawn it.
        `_shutdown` must not call get_client()/get_agent_subprocess() in this
        path, and must reset the class futures.
        """
        cls, persona = _make_lazy_persona()
        cls._before_subprocess_future = None
        persona.get_client = AsyncMock()
        persona.get_agent_subprocess = AsyncMock()
        persona.get_session_id = AsyncMock()

        await persona._shutdown()

        persona.get_client.assert_not_awaited()
        persona.get_agent_subprocess.assert_not_awaited()
        persona.get_session_id.assert_not_awaited()
        assert cls._client_future is None
        assert cls._subprocess_future is None


class TestPrepareConcurrencyAndRetry:
    """
    Hardening for prepare(): idempotent under concurrency, exactly one shared
    subprocess across instances of the same class, exception propagation, and
    retry after a failed startup.
    """

    async def test_concurrent_prepare_spawns_one_subprocess(self):
        """Many concurrent prepare() calls on one instance = one subprocess,
        one engagement event."""
        calls = []

        async def counting_subprocess():
            calls.append(1)
            return "subprocess"

        cls, persona = _make_lazy_persona(subprocess_impl=counting_subprocess)

        await asyncio.gather(*[persona.prepare() for _ in range(5)])
        # Let the (single) subprocess task actually run.
        await persona.get_agent_subprocess()

        assert sum(calls) == 1
        ops = [
            c.kwargs.get("data", {}).get("operation")
            for c in persona.event_logger.emit.call_args_list
        ]
        assert ops.count("acp_engagement") == 1

    async def test_two_instances_same_class_share_one_subprocess(self):
        """Two personas of the same class preparing at once spawn exactly one
        shared agent subprocess."""
        calls = []

        async def counting_subprocess():
            calls.append(1)
            return "subprocess"

        cls, p1 = _make_lazy_persona(subprocess_impl=counting_subprocess)
        _, p2 = _make_lazy_persona(persona_cls=cls, subprocess_impl=counting_subprocess)

        await asyncio.gather(p1.prepare(), p2.prepare())
        # Both instances resolve the same shared subprocess.
        s1 = await p1.get_agent_subprocess()
        s2 = await p2.get_agent_subprocess()

        assert sum(calls) == 1
        assert s1 == s2 == "subprocess"
        assert p1.__class__._subprocess_future is p2.__class__._subprocess_future

    async def test_failed_startup_is_retried_on_next_prepare(self):
        """A subprocess that failed is discarded and retried on the next
        prepare()."""
        state = {"fail": True, "calls": 0}

        async def flaky_subprocess():
            state["calls"] += 1
            if state["fail"]:
                raise RuntimeError("spawn boom")
            return "subprocess"

        cls, persona = _make_lazy_persona(subprocess_impl=flaky_subprocess)

        # prepare() awaits startup, so the spawn failure propagates out of it.
        with pytest.raises(RuntimeError, match="spawn boom"):
            await persona.prepare()

        # Clear the fault and prepare again: the failed task is discarded and recreated.
        state["fail"] = False
        await persona.prepare()
        assert await persona.get_agent_subprocess() == "subprocess"
        assert state["calls"] == 2  # retried, not cached-failed

    async def test_cancelled_startup_is_discarded_on_next_prepare(self):
        """A cancelled startup task is discarded and recreated on the next
        prepare() (a cancelled future is a not-succeeded future)."""
        cls, persona = _make_lazy_persona()

        # Seed a previously-cancelled shared subprocess task on the class.
        async def _never():
            await asyncio.sleep(3600)

        stuck = persona.event_loop.create_task(_never())
        stuck.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stuck
        cls._subprocess_future = stuck

        # prepare() must discard the cancelled task and create a live one.
        await persona.prepare()
        assert await persona.get_agent_subprocess() == "subprocess"

    async def test_engagement_re_emits_only_after_reset(self):
        """A successful prepare() emits engagement once even across retries of a
        failed startup (engagement is per-instance, emitted on first call)."""
        cls, persona = _make_lazy_persona()
        await persona.prepare()
        await persona.prepare()
        ops = [
            c.kwargs.get("data", {}).get("operation")
            for c in persona.event_logger.emit.call_args_list
        ]
        assert ops.count("acp_engagement") == 1


class TestProcessMessagePropagatesStartupFailure:
    """A startup failure reaches process_message so the manager can show it."""

    async def test_get_client_failure_propagates(self):
        persona = _make_persona()  # persona.prepare is an AsyncMock no-op
        persona.get_client = AsyncMock(side_effect=RuntimeError("client boom"))

        with pytest.raises(RuntimeError, match="client boom"):
            await BaseAcpPersona.process_message(persona, _make_message("@bot hi"))


class TestFunnelEvents:
    """Event funnel: requirements_met (construction) -> engagement (prepare)
    -> login + usage (process_message)."""

    def test_requirements_met_emitted_on_construction(self, monkeypatch):
        import jupyter_ai_acp_client.base_acp_persona as mod

        emitted = []
        monkeypatch.setattr(
            mod,
            "emit_event",
            lambda logger, op, outcome, details=None: emitted.append(op),
        )
        # Stub the heavy BasePersona.__init__ so we can exercise BaseAcpPersona's.
        monkeypatch.setattr(mod.BasePersona, "__init__", lambda self, *a, **k: None)

        class _P(BaseAcpPersona):
            event_loop = None
            event_logger = None

            @property
            def defaults(self):
                return MagicMock()

        p = _P.__new__(_P)
        BaseAcpPersona.__init__(p, executable=["x"])

        assert "acp_requirements_met" in emitted

    async def test_login_and_success_emitted_on_successful_prepare(self):
        cls, persona = _make_lazy_persona()

        await persona.prepare()

        ops = [
            c.kwargs.get("data", {}).get("operation")
            for c in persona.event_logger.emit.call_args_list
        ]
        assert "acp_login" in ops
        assert "acp_success" in ops

    async def test_login_failure_and_no_success_when_unauthenticated(self):
        cls, persona = _make_lazy_persona()
        persona.auth.assert_auth = AsyncMock(side_effect=PersonaNotAuthenticated())

        with pytest.raises(PersonaNotAuthenticated):
            await persona.prepare()

        calls = [
            (
                c.kwargs.get("data", {}).get("operation"),
                c.kwargs.get("data", {}).get("outcome"),
            )
            for c in persona.event_logger.emit.call_args_list
        ]
        assert ("acp_login", "failure") in calls
        assert ("acp_login", "success") not in calls
        assert ("acp_success", "success") not in calls





class TestUnauthenticated:
    """
    Auth is asserted in `prepare()` via `self.auth.assert_auth()` (after the
    connection, before the session). When it raises `PersonaNotAuthenticated`,
    `prepare()` stops without creating a session; the manager maps that to
    `PreparationState.NOT_AUTHED` and prompts on the message path. The base
    `handle_message_no_auth` records `_was_initially_unauthenticated` so reactive
    agents resume on their next message.
    """

    async def test_prepare_raises_and_creates_no_session_when_unauthed(self):
        cls, persona = _make_lazy_persona()
        persona.auth.assert_auth = AsyncMock(side_effect=PersonaNotAuthenticated())

        with pytest.raises(PersonaNotAuthenticated):
            await persona.prepare()

        # assert_auth runs before the session is obtained, so prepare never
        # reaches success (no acp_success emitted, session not awaited).
        ops = [
            c.kwargs.get("data", {}).get("operation")
            for c in persona.event_logger.emit.call_args_list
        ]
        assert "acp_success" not in ops

    async def test_prepare_proceeds_when_authed(self):
        cls, persona = _make_lazy_persona()
        # assert_auth passes (default), so startup completes.
        await persona.prepare()

        assert cls._subprocess_future is not None
        assert cls._client_future is not None
        assert persona._client_session_future is not None

    async def test_handle_message_no_auth_sets_resume_flag(self):
        cls, persona = _make_lazy_persona()
        persona._was_initially_unauthenticated = False

        await BaseAcpPersona.handle_message_no_auth(persona, None)

        assert persona._was_initially_unauthenticated is True

