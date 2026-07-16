"""Tests for content block building and session management in JaiAcpClient."""

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from acp.exceptions import RequestError
from acp.schema import (
    AvailableCommand,
    AvailableCommandsUpdate,
    ConfigOptionUpdate,
    CurrentModeUpdate,
    ResourceContentBlock,
    TextContentBlock,
    Usage,
    UsageUpdate,
)

from jupyterlab_chat.models import User
from jupyterlab_chat.ychat import YChat
from pycrdt import Awareness

from jupyter_ai_persona_manager import PersonaAwareness

from jupyter_ai_acp_client.base_acp_persona import BaseAcpPersona
from jupyter_ai_acp_client.default_acp_client import JaiAcpClient


SESSION_ID = "sess-1"


def _awareness() -> PersonaAwareness:
    """A real PersonaAwareness over a fresh in-memory YChat. Constructed outside
    an event loop, so the heartbeat is skipped — everything else is real."""
    ychat = YChat()
    ychat.awareness = Awareness(ydoc=ychat._ydoc)
    user = User(username="test-persona", name="Test", display_name="Test")
    return PersonaAwareness(
        ychat=ychat, log=logging.getLogger("test"), user=user, id="test-persona"
    )


def _make_client_and_persona():
    """Create a minimal mock JaiAcpClient with a persona wired for testing."""
    client = object.__new__(JaiAcpClient)
    client._prompt_locks_by_session = {}
    client._cancel_requested = {}
    client._legacy_models_by_session = {}
    client._permission_manager = MagicMock()

    # Mock connection
    conn = AsyncMock()
    conn.prompt = AsyncMock(return_value=MagicMock())
    client.get_connection = AsyncMock(return_value=conn)

    # Mock persona
    persona = MagicMock()
    persona.log = MagicMock()
    persona.awareness = MagicMock()
    persona.ychat = MagicMock()
    persona.ychat.get_message.return_value = None

    # Mock tool call manager
    client._tool_call_manager = MagicMock()

    client._personas_by_session = {SESSION_ID: persona}

    return client, conn, persona


class TestPromptAndReplyContentBlocks:
    """Tests for how prompt_and_reply builds ACP content blocks."""

    async def test_text_only(self):
        """Without attachments, sends a single TextContentBlock."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(session_id=SESSION_ID, prompt="hello")

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextContentBlock)
        assert blocks[0].text == "hello"

    async def test_file_attachment_produces_resource_block(self):
        """A file attachment produces a ResourceContentBlock with file:// URI."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="check this",
            attachments=[{"value": "src/main.py", "type": "file", "mimetype": "text/x-python"}],
            root_dir="/home/user/notebooks",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert len(blocks) == 2
        assert isinstance(blocks[1], ResourceContentBlock)
        assert blocks[1].uri == Path("/home/user/notebooks/src/main.py").resolve().as_uri()
        assert blocks[1].name == "main.py"
        assert blocks[1].mime_type == "text/x-python"

    async def test_notebook_attachment_default_mime_type(self):
        """Notebook attachments get application/x-ipynb+json when mimetype is None."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="review",
            attachments=[{"value": "analysis.ipynb", "type": "notebook"}],
            root_dir="/home/user",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].mime_type == "application/x-ipynb+json"

    async def test_notebook_explicit_mimetype_preserved(self):
        """When notebook has explicit mimetype, it is preserved."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="review",
            attachments=[{"value": "nb.ipynb", "type": "notebook", "mimetype": "custom/type"}],
            root_dir="/home/user",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].mime_type == "custom/type"

    async def test_multiple_attachments_in_order(self):
        """Multiple attachments produce ResourceContentBlocks in order after text."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="review all",
            attachments=[
                {"value": "a.py", "type": "file"},
                {"value": "b.ipynb", "type": "notebook"},
            ],
            root_dir="/tmp",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert len(blocks) == 3
        assert blocks[0].text == "review all"
        assert blocks[1].name == "a.py"
        assert blocks[2].name == "b.ipynb"
        assert blocks[2].mime_type == "application/x-ipynb+json"

    async def test_none_attachments(self):
        """None attachments produces only the text block."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="hello",
            attachments=None,
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert len(blocks) == 1

    async def test_empty_list_attachments(self):
        """Empty attachment list produces only the text block."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="hello",
            attachments=[],
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert len(blocks) == 1

    async def test_empty_value_fallback_name(self):
        """When attachment value is empty, name falls back to '<attachment>'."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="check",
            attachments=[{"value": "", "type": "file"}],
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].name == "<attachment>"

    async def test_mimetype_none_for_file(self):
        """File attachment with no mimetype gets None mime_type."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="check",
            attachments=[{"value": "data.csv", "type": "file"}],
            root_dir="/tmp",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].mime_type is None

    async def test_no_root_dir_uses_relative_path(self):
        """When root_dir is None, URI is the raw relative path."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="check",
            attachments=[{"value": "subdir/file.py", "type": "file"}],
            root_dir=None,
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].uri == "subdir/file.py"

    async def test_file_uri_format(self):
        """file:// URI has correct RFC 8089 format with three slashes."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="check",
            attachments=[{"value": "test.py", "type": "file"}],
            root_dir="/home/user",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].uri.startswith("file:///")

    async def test_path_traversal_blocked(self):
        """Attachment path escaping root_dir falls back to raw relative path."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="check",
            attachments=[{"value": "../../../etc/passwd", "type": "file"}],
            root_dir="/home/user/notebooks",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].uri == "../../../etc/passwd"


def _real_usage_persona():
    """
    A `BaseAcpPersona` created without `__init__` (no subprocess or session),
    carrying the real usage setters and properties so tests cover the actual
    store-then-read round trip. Collaborators the client touches are mocked.
    """

    class _ConcreteAcpPersona(BaseAcpPersona):
        @property
        def defaults(self):  # pragma: no cover - never called in these tests
            return None

    persona = _ConcreteAcpPersona.__new__(_ConcreteAcpPersona)
    persona._acp_context_usage = None
    persona._acp_session_usage = None
    persona._acp_context_percent = None
    persona.log = logging.getLogger("test")
    # A real awareness slot so `_sync_awareness_usage` -> `report_usage`
    # round-trips through the real typed properties.
    persona.awareness = _awareness()
    persona.ychat = MagicMock()
    return persona


class TestUsageStorage:
    """A usage report received by the client ends up readable on the persona."""

    async def test_usage_update_is_stored_as_context_usage(self):
        client, _, _ = _make_client_and_persona()
        client._loading_sessions = {}
        persona = _real_usage_persona()
        client._personas_by_session[SESSION_ID] = persona
        update = UsageUpdate(sessionUpdate="usage_update", used=41_000, size=200_000)

        await client.session_update(SESSION_ID, update)

        assert persona.acp_context_usage is update

    async def test_prompt_response_usage_is_stored_as_session_usage(self):
        client, conn, _ = _make_client_and_persona()
        persona = _real_usage_persona()
        client._personas_by_session[SESSION_ID] = persona
        usage = Usage(inputTokens=900, outputTokens=340, totalTokens=1_240)
        conn.prompt = AsyncMock(return_value=MagicMock(usage=usage))

        await client.prompt_and_reply(session_id=SESSION_ID, prompt="hello")

        assert persona.acp_session_usage is usage

    async def test_prompt_response_without_usage_stores_nothing(self):
        client, conn, _ = _make_client_and_persona()
        persona = _real_usage_persona()
        client._personas_by_session[SESSION_ID] = persona
        conn.prompt = AsyncMock(return_value=MagicMock(usage=None))

        await client.prompt_and_reply(session_id=SESSION_ID, prompt="hello")

        assert persona.acp_session_usage is None

    async def test_kiro_metadata_notification_records_context_percent(self):
        client, _, _ = _make_client_and_persona()
        persona = _real_usage_persona()
        client._personas_by_session[SESSION_ID] = persona

        await client.ext_notification(
            "kiro.dev/metadata",
            {"sessionId": SESSION_ID, "contextUsagePercentage": 1.252000093460083},
        )

        assert persona.acp_context_percent == pytest.approx(1.252000093460083)
        # The percent is also published over awareness for the toolbar.
        assert persona.get_usage().context_percent == pytest.approx(
            1.252000093460083
        )

    async def test_kiro_metadata_for_unknown_session_is_ignored(self):
        client, _, _ = _make_client_and_persona()
        persona = _real_usage_persona()
        client._personas_by_session[SESSION_ID] = persona

        await client.ext_notification(
            "kiro.dev/metadata",
            {"sessionId": "nope", "contextUsagePercentage": 1.48},
        )

        assert persona.acp_context_percent is None

    async def test_kiro_metadata_without_percentage_records_nothing(self):
        client, _, _ = _make_client_and_persona()
        persona = _real_usage_persona()
        client._personas_by_session[SESSION_ID] = persona

        await client.ext_notification(
            "kiro.dev/metadata",
            {
                "sessionId": SESSION_ID,
                "meteringUsage": [
                    {"value": 0.031, "unit": "credit", "unitPlural": "credits"}
                ],
                "turnDurationMs": 2178,
            },
        )

        assert persona.acp_context_percent is None

    async def test_kiro_metadata_with_malformed_percentage_records_nothing(self):
        client, _, _ = _make_client_and_persona()
        persona = _real_usage_persona()
        client._personas_by_session[SESSION_ID] = persona

        for malformed in ("1.25", True, None, [1.25]):
            await client.ext_notification(
                "kiro.dev/metadata",
                {"sessionId": SESSION_ID, "contextUsagePercentage": malformed},
            )
            assert persona.acp_context_percent is None, repr(malformed)


class TestLegacyModels:
    """Kiro-style legacy models: captured off the raw session/new response,
    handed over per session, applied via the legacy `session/set_model`."""

    MODELS = {
        "currentModelId": "claude-sonnet-5",
        "availableModels": [
            {"modelId": "auto", "name": "auto", "description": "Chosen by task"},
            {"modelId": "claude-sonnet-5", "name": "claude-sonnet-5"},
        ],
    }

    def _event(self, message):
        return SimpleNamespace(message=message)

    def test_session_new_result_models_are_captured(self):
        client, _, _ = _make_client_and_persona()

        client._capture_legacy_models(
            self._event(
                {"id": 1, "result": {"sessionId": "sess-9", "models": self.MODELS}}
            )
        )

        assert client.pop_legacy_models("sess-9") == self.MODELS
        # pop is one-shot: the captured payload is handed over once.
        assert client.pop_legacy_models("sess-9") is None

    def test_non_matching_messages_are_ignored(self):
        client, _, _ = _make_client_and_persona()

        for message in (
            {"method": "session/update", "params": {"sessionId": "sess-9"}},
            {"id": 2, "result": {"sessionId": "sess-9"}},
            {"id": 3, "result": {"sessionId": "sess-9", "models": ["not-a-dict"]}},
            {"id": 4, "result": "ok"},
            None,
        ):
            client._capture_legacy_models(self._event(message))

        assert client.pop_legacy_models("sess-9") is None

    async def test_set_session_model_sends_legacy_request(self):
        client, conn, _ = _make_client_and_persona()

        await client.set_session_model("auto", SESSION_ID)

        conn.send_raw_request.assert_awaited_once_with(
            "session/set_model", {"sessionId": SESSION_ID, "modelId": "auto"}
        )


class TestKiroCommands:
    """Kiro's vendor commands notification is published as slash commands."""

    async def test_commands_notification_publishes_slash_commands(self):
        client, _, persona = _make_client_and_persona()
        persona.report_slash_commands = MagicMock()

        await client.ext_notification(
            "kiro.dev/commands/available",
            {
                "sessionId": SESSION_ID,
                "commands": [
                    {"name": "/model", "description": "Select a model"},
                    {
                        "name": "compact",
                        "description": "Compact context",
                        "meta": {"local": True},
                    },
                    {"name": "/clear"},
                ],
                "tools": [],
            },
        )

        commands = persona.report_slash_commands.call_args[0][0]
        # Names are leading-slash normalized, like the standard update path.
        assert [(c.name, c.description) for c in commands] == [
            ("/model", "Select a model"),
            ("/compact", "Compact context"),
            ("/clear", None),
        ]

    async def test_commands_for_unknown_session_are_ignored(self):
        client, _, persona = _make_client_and_persona()
        persona.report_slash_commands = MagicMock()

        await client.ext_notification(
            "kiro.dev/commands/available",
            {"sessionId": "nope", "commands": [{"name": "/model"}]},
        )

        persona.report_slash_commands.assert_not_called()

    async def test_malformed_command_entries_are_skipped(self):
        client, _, persona = _make_client_and_persona()
        persona.report_slash_commands = MagicMock()

        await client.ext_notification(
            "kiro.dev/commands/available",
            {
                "sessionId": SESSION_ID,
                "commands": ["not-a-dict", {}, {"name": 5}, {"name": "/ok"}],
            },
        )

        commands = persona.report_slash_commands.call_args[0][0]
        assert [(c.name, c.description) for c in commands] == [("/ok", None)]

    async def test_empty_commands_keep_previous_advertisement(self):
        client, _, persona = _make_client_and_persona()
        persona.report_slash_commands = MagicMock()

        await client.ext_notification(
            "kiro.dev/commands/available",
            {"sessionId": SESSION_ID, "commands": []},
        )

        persona.report_slash_commands.assert_not_called()


class TestAwarenessPush:
    """The client pushes ACP updates onto the persona's awareness API too."""

    async def test_available_commands_update_advertises_over_awareness(self):
        client, _, persona = _make_client_and_persona()
        client._loading_sessions = {}
        persona.report_slash_commands = MagicMock()
        update = AvailableCommandsUpdate(
            sessionUpdate="available_commands_update",
            availableCommands=[
                AvailableCommand(name="compact", description="Compact context"),
                AvailableCommand(name="/clear", description="Clear"),
            ],
        )

        await client.session_update(SESSION_ID, update)

        commands = persona.report_slash_commands.call_args[0][0]
        # Names are leading-slash normalized.
        assert [(c.name, c.description) for c in commands] == [
            ("/compact", "Compact context"),
            ("/clear", "Clear"),
        ]

    async def test_current_mode_update_rebuilds_awareness_config(self):
        client, _, persona = _make_client_and_persona()
        client._loading_sessions = {}
        persona._sync_awareness_config = MagicMock()
        update = CurrentModeUpdate(sessionUpdate="current_mode_update", currentModeId="code")

        await client.session_update(SESSION_ID, update)

        persona.update_acp_current_mode.assert_called_once_with("code")
        persona._sync_awareness_config.assert_called_once()

    async def test_config_option_update_rebuilds_awareness_config(self):
        client, _, persona = _make_client_and_persona()
        client._loading_sessions = {}
        persona._sync_awareness_config = MagicMock()
        update = ConfigOptionUpdate(
            sessionUpdate="config_option_update", configOptions=[]
        )

        await client.session_update(SESSION_ID, update)

        persona.update_acp_config_options.assert_called_once()
        persona._sync_awareness_config.assert_called_once()

    async def test_usage_update_pushes_awareness_usage(self):
        client, _, persona = _make_client_and_persona()
        client._loading_sessions = {}
        persona._sync_awareness_usage = MagicMock()
        update = UsageUpdate(sessionUpdate="usage_update", used=1, size=2)

        await client.session_update(SESSION_ID, update)

        persona.update_acp_context_usage.assert_called_once_with(update)
        persona._sync_awareness_usage.assert_called_once()


class TestLoadSessionCleanup:
    """Tests for _loading_sessions cleanup on failure."""

    async def test_failed_load_session_removes_task_from_loading_sessions(self):
        """A failed load_session cleans up its task so retries can start fresh."""
        client = object.__new__(JaiAcpClient)
        client.event_loop = asyncio.get_running_loop()
        client._loading_sessions = {}

        persona = MagicMock()
        error = RequestError(-32002, "Resource not found")

        async def _failing_rpc(*args, **kwargs):
            raise error

        client._load_session_rpc = _failing_rpc

        with pytest.raises(RequestError):
            await client.load_session(persona, "stale-session-id")

        assert "stale-session-id" not in client._loading_sessions
