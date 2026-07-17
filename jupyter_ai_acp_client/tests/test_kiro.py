"""
Tests for the Kiro-scoped ACP client and persona.

kiro-cli's ACP surface is non-standard, so all of its handling lives in
`KiroAcpClient` (`kiro_client.py`) and `KiroAcpPersona`
(`acp_personas/kiro.py`) rather than the generic `JaiAcpClient` /
`BaseAcpPersona`. These tests cover:

- the defensive, all-optional Pydantic models for Kiro's payloads;
- the client capturing the legacy `models` block off raw session responses and
  pushing it onto the persona;
- the `session/set_model` legacy RPC;
- the vendor `_kiro.dev/metadata` (usage) and `_kiro.dev/commands/available`
  notifications, including **per-session isolation** (a usage report touches
  only its own session's persona) and the **Kiro-credit cost** (extra credit);
- the persona sourcing its model picker from the captured models and routing
  `update_model` through `session/set_model`;
- the client injection wiring (`KiroAcpPersona.acp_client_class`).
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from acp.exceptions import RequestError
from acp.schema import LoadSessionResponse, NewSessionResponse
from jupyterlab_chat.models import User
from jupyterlab_chat.ychat import YChat
from pycrdt import Awareness

from jupyter_ai_persona_manager import PersonaAwareness

# `kiro.py` runs a `kiro-cli` install/version guard at import time, which raises
# where kiro-cli isn't installed (e.g. CI). Satisfy the guard with mocks so the
# persona can be imported and unit-tested without the binary present.
with patch("shutil.which", return_value="/usr/bin/kiro-cli"), patch(
    "subprocess.run",
    return_value=MagicMock(returncode=0, stdout="kiro-cli 2.12.3", stderr=""),
):
    from jupyter_ai_acp_client.acp_personas.kiro import KiroAcpPersona
from jupyter_ai_acp_client.kiro_client import (
    KiroAcpClient,
    KiroCommand,
    KiroCommands,
    KiroMeteringUsage,
    KiroMetadata,
    KiroModelOption,
    KiroModels,
)

_ALL_KIRO_MODELS = [
    KiroModels,
    KiroModelOption,
    KiroMetadata,
    KiroMeteringUsage,
    KiroCommands,
    KiroCommand,
]


SESSION_ID = "sess-1"

MODELS = {
    "currentModelId": "claude-sonnet-5",
    "availableModels": [
        {"modelId": "auto", "name": "auto", "description": "Chosen by task"},
        {"modelId": "claude-sonnet-5", "name": "claude-sonnet-5"},
    ],
}


def _awareness(persona_id: str = "test-persona") -> PersonaAwareness:
    """A real PersonaAwareness over a fresh in-memory YChat. Constructed outside
    an event loop, so the heartbeat is skipped — everything else is real."""
    ychat = YChat()
    ychat.awareness = Awareness(ydoc=ychat._ydoc)
    user = User(username=persona_id, name="Test", display_name="Test")
    return PersonaAwareness(
        ychat=ychat, log=logging.getLogger("test"), user=user, id=persona_id
    )


def _kiro_persona(persona_id: str = "test-persona") -> KiroAcpPersona:
    """
    A real `KiroAcpPersona` built without `__init__` (no subprocess/session),
    carrying a real awareness slot so `report_usage`/`report_model_configuration`
    round-trip through the real typed properties. Collaborators are mocked.
    """
    persona = KiroAcpPersona.__new__(KiroAcpPersona)
    persona.log = logging.getLogger("test")
    persona.awareness = _awareness(persona_id)
    persona.ychat = MagicMock()
    persona._acp_modes = []
    persona._acp_current_mode_id = None
    persona._acp_config_options = []
    persona._acp_context_usage = None
    persona._acp_session_usage = None
    persona._kiro_models = None
    return persona


def _kiro_client(send_request_return=None):
    """
    A `KiroAcpClient` built without `__init__`, with a mocked connection whose
    inner `_conn.send_request` is an AsyncMock. `_personas_by_session` starts
    empty; tests populate it.
    """
    client = object.__new__(KiroAcpClient)
    client._personas_by_session = {}
    client._loading_sessions = {}
    client._metering_totals = {}
    client._metering_units = {}

    conn = MagicMock()
    conn._conn = MagicMock()
    conn._conn.send_request = AsyncMock(return_value=send_request_return)
    client.get_connection = AsyncMock(return_value=conn)
    client._get_mcp_servers = AsyncMock(return_value=[])
    return client, conn


# ---------------------------------------------------------------------------
# Pydantic models: defensive parsing
# ---------------------------------------------------------------------------


class TestKiroModels:
    @pytest.mark.parametrize("model", _ALL_KIRO_MODELS)
    def test_missing_fields_never_raise(self, model):
        # Every field is optional: an empty payload validates to all-None/defaults.
        model.model_validate({})

    @pytest.mark.parametrize("model", _ALL_KIRO_MODELS)
    def test_added_unknown_fields_never_raise(self, model):
        # Fields Kiro might add in the future are ignored, never rejected.
        model.model_validate(
            {"brandNewFieldKiroAdded": {"nested": [1, 2, 3]}, "another": 5}
        )

    def test_wrong_typed_fields_never_raise(self):
        # Every known field present but the wrong type: each degrades to
        # None/empty rather than raising, so a shape change can't crash parsing.
        assert KiroModels.model_validate(
            {"currentModelId": 123, "availableModels": "not-a-list"}
        ).available_models is None
        meta = KiroMetadata.model_validate(
            {
                "sessionId": [1],
                "contextUsagePercentage": {},
                "meteringUsage": 7,
                "turnDurationMs": "not-a-number",
            }
        )
        assert meta.session_id is None
        assert meta.context_usage_percentage is None
        assert meta.metering_usage is None
        assert meta.turn_duration_ms is None
        cmds = KiroCommands.model_validate({"sessionId": 5, "commands": "not-a-list"})
        assert cmds.session_id is None
        assert cmds.commands is None
        cmd = KiroCommand.model_validate(
            {"name": 5, "description": ["x"], "meta": "not-a-dict"}
        )
        assert cmd.name is None and cmd.description is None and cmd.meta is None

    def test_metadata_percentage_coercion(self):
        # A genuine number is kept; anything else is dropped rather than coerced.
        assert KiroMetadata.model_validate(
            {"contextUsagePercentage": 1.252}
        ).context_usage_percentage == pytest.approx(1.252)
        assert (
            KiroMetadata.model_validate(
                {"contextUsagePercentage": 5}
            ).context_usage_percentage
            == 5.0
        )
        for malformed in ("1.25", True, None, [1.25]):
            assert (
                KiroMetadata.model_validate(
                    {"contextUsagePercentage": malformed}
                ).context_usage_percentage
                is None
            ), repr(malformed)
        assert KiroMetadata.model_validate({}).context_usage_percentage is None

    def test_metadata_parses_metering_and_ignores_extra(self):
        meta = KiroMetadata.model_validate(
            {
                "sessionId": "s1",
                "meteringUsage": [
                    {"value": 0.031, "unit": "credit", "unitPlural": "credits"},
                    "not-a-dict",
                ],
                "turnDurationMs": 2178,
                "unknownField": "ignored",
            }
        )
        assert meta.session_id == "s1"
        # Non-dict metering entries are dropped; the valid one parses.
        assert len(meta.metering_usage) == 1
        assert meta.metering_usage[0].value == pytest.approx(0.031)
        assert meta.metering_usage[0].unit == "credit"
        assert meta.turn_duration_ms == 2178

    def test_metering_value_only_real_numbers(self):
        m = KiroMetadata.model_validate(
            {"meteringUsage": [{"value": "0.5", "unit": "credit"}]}
        )
        # A string value is ignored, not coerced.
        assert m.metering_usage[0].value is None
        assert m.metering_usage[0].unit == "credit"

    def test_models_tolerate_null_and_mistyped_fields(self):
        # Explicit null list stays None (the persona treats it as no options).
        m = KiroModels.model_validate(
            {"currentModelId": "m1", "availableModels": None}
        )
        assert m.current_model_id == "m1"
        assert m.available_models is None

        # Mistyped name/description degrade to None without raising.
        m = KiroModels.model_validate(
            {"availableModels": [{"modelId": "m1", "name": 7, "description": ["x"]}]}
        )
        assert m.available_models[0].model_id == "m1"
        assert m.available_models[0].name is None
        assert m.available_models[0].description is None

    def test_models_drop_non_dict_and_idless_entries(self):
        m = KiroModels.model_validate(
            {
                "availableModels": [
                    "not-a-dict",
                    {"name": "no-id"},
                    {"modelId": "m1", "name": "M1"},
                ]
            }
        )
        # Non-dicts dropped at parse; the id-less entry survives parsing but is
        # filtered by the persona (see TestKiroPersonaModelConfig).
        ids = [o.model_id for o in m.available_models]
        assert ids == [None, "m1"]

    def test_commands_drop_non_dict_and_nameless_entries(self):
        cmds = KiroCommands.model_validate(
            {
                "sessionId": "s1",
                "commands": ["not-a-dict", {}, {"name": 5}, {"name": "/ok"}],
            }
        )
        assert [c.name for c in cmds.commands] == ["/ok"]


# ---------------------------------------------------------------------------
# Client: session model capture
# ---------------------------------------------------------------------------


class TestKiroClientSession:
    async def test_create_session_captures_and_pushes_models(self):
        client, conn = _kiro_client({"sessionId": "sess-9", "models": MODELS})
        persona = MagicMock()
        persona.get_chat_dir.return_value = "/tmp"

        response = await client.create_session(persona)

        # Raw session/new was issued and the typed response validated.
        assert isinstance(response, NewSessionResponse)
        assert response.session_id == "sess-9"
        # Parsed models pushed straight onto the persona.
        pushed = persona.set_kiro_models.call_args[0][0]
        assert isinstance(pushed, KiroModels)
        assert pushed.current_model_id == "claude-sonnet-5"
        # Session registered against the persona.
        assert client._personas_by_session["sess-9"] is persona

    async def test_load_session_rpc_keys_models_by_requested_id(self):
        # A session/load response carries no sessionId; key by the requested id.
        client, conn = _kiro_client({"models": MODELS})
        persona = MagicMock()
        persona.get_chat_dir.return_value = "/tmp"

        response = await client._load_session_rpc(persona, "sess-9")

        assert isinstance(response, LoadSessionResponse)
        assert isinstance(persona.set_kiro_models.call_args[0][0], KiroModels)
        assert client._personas_by_session["sess-9"] is persona

    async def test_session_without_models_pushes_none(self):
        client, conn = _kiro_client({"sessionId": "sess-9", "modes": None})
        persona = MagicMock()
        persona.get_chat_dir.return_value = "/tmp"

        await client.create_session(persona)

        persona.set_kiro_models.assert_called_once_with(None)

    async def test_create_session_persona_without_set_kiro_models(self):
        # The client backs any BaseAcpPersona, not just KiroAcpPersona. A persona
        # that doesn't surface a model picker (e.g. the usage-only E2E fixture)
        # has no `set_kiro_models`; the client must not require it.
        client, conn = _kiro_client({"sessionId": "sess-9", "models": MODELS})

        class _NoModelSetter:
            def get_chat_dir(self):
                return "/tmp"

        persona = _NoModelSetter()

        response = await client.create_session(persona)

        assert response.session_id == "sess-9"
        assert client._personas_by_session["sess-9"] is persona

    async def test_set_session_model_sends_raw_request(self):
        client, conn = _kiro_client()

        await client.set_session_model("auto", SESSION_ID)

        conn._conn.send_request.assert_awaited_once_with(
            "session/set_model", {"sessionId": SESSION_ID, "modelId": "auto"}
        )


# ---------------------------------------------------------------------------
# Client: vendor command notification
# ---------------------------------------------------------------------------


class TestKiroClientCommands:
    async def test_commands_notification_publishes_slash_commands(self):
        client, _ = _kiro_client()
        persona = MagicMock()
        client._personas_by_session[SESSION_ID] = persona

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
        # Names are leading-slash normalized; `meta` is ignored.
        assert [(c.name, c.description) for c in commands] == [
            ("/model", "Select a model"),
            ("/compact", "Compact context"),
            ("/clear", None),
        ]

    async def test_commands_for_unknown_session_are_ignored(self):
        client, _ = _kiro_client()
        persona = MagicMock()
        client._personas_by_session[SESSION_ID] = persona

        await client.ext_notification(
            "kiro.dev/commands/available",
            {"sessionId": "nope", "commands": [{"name": "/model"}]},
        )

        persona.report_slash_commands.assert_not_called()

    async def test_malformed_command_entries_are_skipped(self):
        client, _ = _kiro_client()
        persona = MagicMock()
        client._personas_by_session[SESSION_ID] = persona

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
        client, _ = _kiro_client()
        persona = MagicMock()
        client._personas_by_session[SESSION_ID] = persona

        await client.ext_notification(
            "kiro.dev/commands/available",
            {"sessionId": SESSION_ID, "commands": []},
        )

        persona.report_slash_commands.assert_not_called()

    async def test_unknown_ext_notification_delegates_to_super(self):
        client, _ = _kiro_client()

        with pytest.raises(RequestError) as exc_info:
            await client.ext_notification("other.vendor/thing", {"sessionId": SESSION_ID})

        # Falls through to the generic handler -> JSON-RPC method-not-found.
        assert exc_info.value.code == -32601


# ---------------------------------------------------------------------------
# Client: vendor usage notification (percent + credit cost + isolation)
# ---------------------------------------------------------------------------


class TestKiroUsage:
    async def test_metadata_records_context_percent(self):
        client, _ = _kiro_client()
        persona = _kiro_persona()
        client._personas_by_session[SESSION_ID] = persona

        await client.ext_notification(
            "kiro.dev/metadata",
            {"sessionId": SESSION_ID, "contextUsagePercentage": 1.252},
        )

        assert persona.get_usage().context_percent == pytest.approx(1.252)

    async def test_metadata_unknown_session_ignored(self):
        client, _ = _kiro_client()
        persona = _kiro_persona()
        client._personas_by_session[SESSION_ID] = persona

        await client.ext_notification(
            "kiro.dev/metadata",
            {"sessionId": "nope", "contextUsagePercentage": 1.48},
        )

        assert persona.get_usage().context_percent is None

    async def test_metadata_without_percentage_records_no_percent(self):
        client, _ = _kiro_client()
        persona = _kiro_persona()
        client._personas_by_session[SESSION_ID] = persona

        await client.ext_notification(
            "kiro.dev/metadata",
            {"sessionId": SESSION_ID, "turnDurationMs": 2178},
        )

        assert persona.get_usage().context_percent is None

    async def test_metadata_malformed_percentage_ignored(self):
        client, _ = _kiro_client()
        persona = _kiro_persona()
        client._personas_by_session[SESSION_ID] = persona

        for malformed in ("1.25", True, None, [1.25]):
            await client.ext_notification(
                "kiro.dev/metadata",
                {"sessionId": SESSION_ID, "contextUsagePercentage": malformed},
            )
            assert persona.get_usage().context_percent is None, repr(malformed)

    async def test_metadata_credit_cost_reported(self):
        """EXTRA CREDIT: the Kiro-credit metered cost is surfaced as cost."""
        client, _ = _kiro_client()
        persona = _kiro_persona()
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

        usage = persona.get_usage()
        assert usage.cost_amount == pytest.approx(0.031)
        # Deliberate ISO-4217 stretch: the unit's plural name, not a currency code.
        assert usage.cost_currency == "credits"
        assert usage.context_percent is None

    async def test_metadata_sums_multiple_metering_entries(self):
        client, _ = _kiro_client()
        persona = _kiro_persona()
        client._personas_by_session[SESSION_ID] = persona

        await client.ext_notification(
            "kiro.dev/metadata",
            {
                "sessionId": SESSION_ID,
                "meteringUsage": [
                    {"value": 0.03, "unit": "credit"},
                    {"value": 0.01, "unit": "credit"},
                ],
            },
        )

        assert persona.get_usage().cost_amount == pytest.approx(0.04)

    async def test_metadata_cost_accumulates_across_turns(self):
        # Kiro meters per turn (deltas), unlike the standard channel's
        # cumulative cost, so the session total must accumulate client-side.
        client, _ = _kiro_client()
        persona = _kiro_persona()
        client._personas_by_session[SESSION_ID] = persona

        for credits in (0.05, 0.041):
            await client.ext_notification(
                "kiro.dev/metadata",
                {
                    "sessionId": SESSION_ID,
                    "contextUsagePercentage": 1.7,
                    "meteringUsage": [
                        {"value": credits, "unit": "credit", "unitPlural": "credits"}
                    ],
                    "turnDurationMs": 1500,
                },
            )

        usage = persona.get_usage()
        assert usage.cost_amount == pytest.approx(0.091)
        assert usage.cost_currency == "credits"

    async def test_metadata_unit_persists_when_later_report_omits_it(self):
        client, _ = _kiro_client()
        persona = _kiro_persona()
        client._personas_by_session[SESSION_ID] = persona

        for payload in (
            [{"value": 0.05, "unitPlural": "credits"}],
            [{"value": 0.03}],
        ):
            await client.ext_notification(
                "kiro.dev/metadata",
                {"sessionId": SESSION_ID, "meteringUsage": payload},
            )

        usage = persona.get_usage()
        assert usage.cost_amount == pytest.approx(0.08)
        # The first named unit sticks; a unit-less later report never resets it.
        assert usage.cost_currency == "credits"

    async def test_metadata_mixed_units_in_one_payload_keep_the_first(self):
        client, _ = _kiro_client()
        persona = _kiro_persona()
        client._personas_by_session[SESSION_ID] = persona

        await client.ext_notification(
            "kiro.dev/metadata",
            {
                "sessionId": SESSION_ID,
                "meteringUsage": [
                    {"value": 0.05, "unitPlural": "credits"},
                    {"value": 99.0, "unitPlural": "tokens"},
                ],
            },
        )

        usage = persona.get_usage()
        # Amounts in a different unit are skipped, never summed together.
        assert usage.cost_amount == pytest.approx(0.05)
        assert usage.cost_currency == "credits"

    async def test_metadata_wholesale_bad_shape_does_not_raise(self):
        # A wholesale surprise (non-object params) is backstopped by _safe_parse:
        # no persona resolves and nothing raises.
        client, _ = _kiro_client()
        persona = _kiro_persona()
        client._personas_by_session[SESSION_ID] = persona

        await client.ext_notification("kiro.dev/metadata", ["not", "an", "object"])

        assert persona.get_usage().context_percent is None

    async def test_two_sessions_usage_isolation(self):
        """A usage notification for one Kiro session updates only that session's
        reported usage; a second concurrent Kiro session is untouched."""
        client, _ = _kiro_client()
        persona_a = _kiro_persona("persona-a")
        persona_b = _kiro_persona("persona-b")
        client._personas_by_session["sess-a"] = persona_a
        client._personas_by_session["sess-b"] = persona_b

        await client.ext_notification(
            "kiro.dev/metadata",
            {
                "sessionId": "sess-a",
                "contextUsagePercentage": 42.0,
                "meteringUsage": [{"value": 0.5, "unit": "credit"}],
            },
        )

        # Session A got the update.
        assert persona_a.get_usage().context_percent == pytest.approx(42.0)
        assert persona_a.get_usage().cost_amount == pytest.approx(0.5)
        # Session B is completely untouched.
        assert persona_b.get_usage().context_percent is None
        assert persona_b.get_usage().cost_amount is None


# ---------------------------------------------------------------------------
# Persona: model configuration + update_model
# ---------------------------------------------------------------------------


class TestKiroPersonaModelConfig:
    def test_build_awareness_config_fills_model_from_kiro_models(self):
        persona = _kiro_persona()
        persona.set_kiro_models(KiroModels.model_validate(MODELS))

        model, settings = persona._build_awareness_config()

        assert model.current == "claude-sonnet-5"
        assert [(o.id, o.name, o.description) for o in model.options] == [
            ("auto", "auto", "Chosen by task"),
            ("claude-sonnet-5", "claude-sonnet-5", None),
        ]
        assert settings == []

    def test_config_option_model_wins_over_kiro_models(self):
        from acp.schema import SessionConfigOptionSelect, SessionConfigSelectOption

        persona = _kiro_persona()
        persona._acp_config_options = [
            SessionConfigOptionSelect(
                id="model",
                name="Model",
                type="select",
                category="model",
                currentValue="opus",
                options=[
                    SessionConfigSelectOption(value="opus", name="Opus"),
                    SessionConfigSelectOption(value="haiku", name="Haiku"),
                ],
            )
        ]
        persona.set_kiro_models(KiroModels.model_validate(MODELS))

        model, _ = persona._build_awareness_config()

        # A genuine ACP model config option takes precedence over legacy models.
        assert model.current == "opus"
        assert [o.id for o in model.options] == ["opus", "haiku"]

    def test_idless_kiro_model_entries_dropped(self):
        persona = _kiro_persona()
        persona.set_kiro_models(
            KiroModels.model_validate(
                {
                    "currentModelId": "m1",
                    "availableModels": [{"name": "no-id"}, {"modelId": "m1", "name": "M1"}],
                }
            )
        )

        model, _ = persona._build_awareness_config()

        assert [(o.id, o.name) for o in model.options] == [("m1", "M1")]

    def test_null_available_models_yields_empty_options(self):
        persona = _kiro_persona()
        persona.set_kiro_models(
            KiroModels.model_validate(
                {"currentModelId": "m1", "availableModels": None}
            )
        )

        model, _ = persona._build_awareness_config()

        assert model.current == "m1"
        assert model.options == []

    def test_no_kiro_models_yields_empty_model(self):
        persona = _kiro_persona()

        model, settings = persona._build_awareness_config()

        assert model.current is None
        assert model.options == []


class TestKiroPersonaUpdateModel:
    async def test_update_model_routes_to_set_session_model(self):
        persona = _kiro_persona()
        persona.set_kiro_models(KiroModels.model_validate(MODELS))
        client = MagicMock()
        client.set_session_model = AsyncMock()
        persona.get_client = AsyncMock(return_value=client)
        persona.get_session_id = AsyncMock(return_value="sess-9")
        persona.set_acp_config_option = AsyncMock()

        await persona.update_model("auto")

        client.set_session_model.assert_awaited_once_with("auto", "sess-9")
        # Not routed through the config-option path.
        persona.set_acp_config_option.assert_not_awaited()
        # The stored payload tracks the new current model for future rebuilds.
        assert persona._kiro_models.current_model_id == "auto"

    async def test_update_model_config_option_wins(self):
        from acp.schema import SessionConfigOptionSelect, SessionConfigSelectOption

        persona = _kiro_persona()
        persona._acp_config_options = [
            SessionConfigOptionSelect(
                id="model",
                name="Model",
                type="select",
                category="model",
                currentValue="opus",
                options=[
                    SessionConfigSelectOption(value="opus", name="Opus"),
                    SessionConfigSelectOption(value="haiku", name="Haiku"),
                ],
            )
        ]
        persona.set_kiro_models(KiroModels.model_validate(MODELS))
        persona.set_acp_config_option = AsyncMock()
        client = MagicMock()
        client.set_session_model = AsyncMock()
        persona.get_client = AsyncMock(return_value=client)

        await persona.update_model("haiku")

        # With a real model config option present, defer to the standard path.
        persona.set_acp_config_option.assert_awaited_once_with("model", "haiku")
        client.set_session_model.assert_not_awaited()


class TestKiroClientInjection:
    def test_kiro_persona_uses_kiro_acp_client(self):
        assert KiroAcpPersona.acp_client_class is KiroAcpClient
