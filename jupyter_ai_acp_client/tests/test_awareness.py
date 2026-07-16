"""Tests for the persona-manager awareness mapping on BaseAcpPersona.

These cover the ACP-state -> awareness-schema translation
(`_build_awareness_config`, `_sync_awareness_usage`) and the `update_*` methods
dispatching to the right ACP setter. Following the style of
`test_base_acp_persona.py`, the persona is a MagicMock (or a real
`BaseAcpPersona` built without `__init__`) and unbound methods are called
directly as `BaseAcpPersona.<method>(persona, ...)`.
"""

import logging
from unittest.mock import AsyncMock, MagicMock

from acp.schema import (
    Cost,
    SessionConfigOptionBoolean,
    SessionConfigOptionSelect,
    SessionConfigSelectGroup,
    SessionConfigSelectOption,
    SessionMode,
    Usage,
    UsageUpdate,
)
from jupyterlab_chat.models import User
from jupyterlab_chat.ychat import YChat
from pycrdt import Awareness

from jupyter_ai_persona_manager import PersonaAwareness

from jupyter_ai_acp_client.base_acp_persona import MODE_CONTROL_ID, BaseAcpPersona


def _awareness() -> PersonaAwareness:
    """A real PersonaAwareness over a fresh in-memory YChat. Constructed outside
    an event loop, so the heartbeat is skipped — everything else is real."""
    ychat = YChat()
    ychat.awareness = Awareness(ydoc=ychat._ydoc)
    user = User(username="test-persona", name="Test", display_name="Test")
    return PersonaAwareness(
        ychat=ychat, log=logging.getLogger("test"), user=user, id="test-persona"
    )


def _select_option(
    option_id: str,
    current: str,
    values: list[str],
    *,
    category: str | None = None,
) -> SessionConfigOptionSelect:
    return SessionConfigOptionSelect(
        id=option_id,
        name=option_id.title(),
        type="select",
        category=category,
        currentValue=current,
        options=[SessionConfigSelectOption(value=v, name=v.title()) for v in values],
    )


def _bool_option(
    option_id: str, current: bool, *, category: str | None = None
) -> SessionConfigOptionBoolean:
    return SessionConfigOptionBoolean(
        id=option_id,
        name=option_id.title(),
        type="boolean",
        category=category,
        currentValue=current,
    )


def _awareness_persona(
    *,
    modes=None,
    current_mode=None,
    config_options=None,
    context=None,
    session_usage=None,
    context_percent=None,
    legacy_models=None,
):
    """A real `BaseAcpPersona` built without `__init__`, carrying the raw ACP
    state the awareness mapping reads plus a real awareness state to broadcast
    into. Collaborators are mocked.

    Models are ordinary config options now (`session/set_model` was removed from
    ACP), so there is no dedicated model state to seed — pass a `category="model"`
    select in `config_options`. Agents on the legacy channel (kiro-cli) pass the
    captured payload as `legacy_models` instead. `modes`/`current_mode` seed the
    dedicated `session/set_mode` state; a `category="mode"` config option is the
    other (preferred) way to advertise a mode."""

    class _Concrete(BaseAcpPersona):
        @property
        def defaults(self):  # pragma: no cover - never called
            return None

    persona = _Concrete.__new__(_Concrete)
    persona.log = logging.getLogger("test")
    persona.awareness = _awareness()
    persona.ychat = MagicMock()
    persona._acp_modes = modes or []
    persona._acp_current_mode_id = current_mode
    persona._acp_config_options = config_options or []
    persona._acp_context_usage = context
    persona._acp_session_usage = session_usage
    persona._acp_context_percent = context_percent
    persona._acp_legacy_models = legacy_models
    return persona


class TestBuildAwarenessConfig:
    """ACP model/mode/config options -> ModelConfiguration + settings list."""

    def test_model_category_option_becomes_model_configuration(self):
        persona = _awareness_persona(
            config_options=[
                _select_option(
                    "model", "opus", ["opus", "haiku"], category="model"
                ),
            ]
        )

        model, settings = persona._build_awareness_config()

        assert model.current == "opus"
        assert [o.id for o in model.options] == ["opus", "haiku"]
        # The model option is consumed by the model picker, not general settings.
        assert settings == []

    def test_model_sourced_by_id_when_uncategorized(self):
        # An option literally named "model" backs the picker even without the
        # category, for agents that predate the category convention.
        persona = _awareness_persona(
            config_options=[_select_option("model", "gpt", ["gpt", "claude"])]
        )

        model, settings = persona._build_awareness_config()

        assert model.current == "gpt"
        assert [o.id for o in model.options] == ["gpt", "claude"]
        assert settings == []

    def test_model_config_category_becomes_model_settings(self):
        persona = _awareness_persona(
            config_options=[
                _select_option("model", "opus", ["opus"], category="model"),
                _select_option(
                    "context_size", "large", ["small", "large"], category="model_config"
                ),
            ],
        )

        model, settings = persona._build_awareness_config()

        # model_config category rides on ModelConfiguration.settings, not the
        # general settings list.
        assert settings == []
        assert len(model.settings) == 1
        ctx = model.settings[0]
        assert ctx.id == "context_size"
        assert ctx.current == "large"
        assert [o.id for o in ctx.options] == ["small", "large"]

    def test_mode_and_other_options_become_general_settings(self):
        persona = _awareness_persona(
            modes=[SessionMode(id="plan", name="Plan"), SessionMode(id="code", name="Code")],
            current_mode="plan",
            config_options=[
                _select_option("effort", "high", ["low", "high"]),
            ],
        )

        model, settings = persona._build_awareness_config()

        assert model.current is None
        by_id = {s.id: s for s in settings}
        # Mode is surfaced as a general setting keyed by MODE_CONTROL_ID.
        assert MODE_CONTROL_ID in by_id
        mode = by_id[MODE_CONTROL_ID]
        assert mode.current == "plan"
        assert [o.id for o in mode.options] == ["plan", "code"]
        # A non-model_config option is also a general setting.
        assert by_id["effort"].current == "high"
        assert [o.id for o in by_id["effort"].options] == ["low", "high"]

    def test_boolean_option_encoded_as_two_option_select(self):
        persona = _awareness_persona(config_options=[_bool_option("allow_all", False)])

        _, settings = persona._build_awareness_config()

        allow = next(s for s in settings if s.id == "allow_all")
        assert allow.current == "false"
        assert [o.id for o in allow.options] == ["true", "false"]

    def test_boolean_option_true_stringified(self):
        persona = _awareness_persona(config_options=[_bool_option("allow_all", True)])

        _, settings = persona._build_awareness_config()

        assert next(s for s in settings if s.id == "allow_all").current == "true"

    def test_mode_config_option_becomes_mode_setting(self):
        # A mode advertised as a config option (category="mode") backs the mode
        # setting — no dedicated set_mode state needed.
        persona = _awareness_persona(
            config_options=[
                _select_option("mode", "ask", ["ask", "code"], category="mode"),
            ]
        )

        _, settings = persona._build_awareness_config()

        by_id = {s.id: s for s in settings}
        assert MODE_CONTROL_ID in by_id
        assert by_id[MODE_CONTROL_ID].current == "ask"
        assert [o.id for o in by_id[MODE_CONTROL_ID].options] == ["ask", "code"]
        # The mode option is consumed by the mode setting, not shown twice.
        assert "mode" not in by_id

    def test_mode_via_both_channels_not_duplicated(self):
        # An agent may advertise a mode through both the dedicated set_mode state
        # and a config option. The config option is preferred and the mode
        # appears exactly once.
        persona = _awareness_persona(
            modes=[SessionMode(id="plan", name="Plan")],
            current_mode="plan",
            config_options=[
                _select_option("mode", "code", ["ask", "code"], category="mode"),
            ],
        )

        _, settings = persona._build_awareness_config()

        mode_settings = [s for s in settings if s.id == MODE_CONTROL_ID]
        assert len(mode_settings) == 1
        # The config option wins over the dedicated state.
        assert mode_settings[0].current == "code"
        assert [o.id for o in mode_settings[0].options] == ["ask", "code"]

    def test_category_ties_resolved_by_array_order(self):
        # When several options share a category, the earliest wins the prominent
        # slot; later same-category options fall through to general settings.
        persona = _awareness_persona(
            config_options=[
                _select_option("model_a", "a1", ["a1", "a2"], category="model"),
                _select_option("model_b", "b1", ["b1", "b2"], category="model"),
                _select_option("mode_a", "ask", ["ask", "code"], category="mode"),
                _select_option("mode_b", "fast", ["fast", "slow"], category="mode"),
            ]
        )

        model, settings = persona._build_awareness_config()

        # First model-category option backs the picker.
        assert model.current == "a1"
        assert [o.id for o in model.options] == ["a1", "a2"]
        by_id = {s.id: s for s in settings}
        # First mode-category option backs the mode setting.
        assert by_id[MODE_CONTROL_ID].current == "ask"
        # The runners-up remain visible as ordinary general settings.
        assert "model_b" in by_id and by_id["model_b"].current == "b1"
        assert "mode_b" in by_id and by_id["mode_b"].current == "fast"

    def test_grouped_select_options_flattened(self):
        opt = SessionConfigOptionSelect(
            id="effort",
            name="Effort",
            type="select",
            currentValue="high",
            options=[
                SessionConfigSelectGroup(
                    group="g1",
                    name="Group 1",
                    options=[
                        SessionConfigSelectOption(value="low", name="Low"),
                        SessionConfigSelectOption(value="high", name="High"),
                    ],
                )
            ],
        )
        persona = _awareness_persona(config_options=[opt])

        _, settings = persona._build_awareness_config()

        effort = next(s for s in settings if s.id == "effort")
        assert [o.id for o in effort.options] == ["low", "high"]

    def test_nothing_advertised_yields_empty_config(self):
        persona = _awareness_persona()

        model, settings = persona._build_awareness_config()

        assert model.current is None
        assert model.options == []
        assert model.settings == []
        assert settings == []

    def test_sync_broadcasts_model_and_settings(self):
        persona = _awareness_persona(
            config_options=[
                _select_option("model", "opus", ["opus"], category="model"),
            ],
            modes=[SessionMode(id="plan", name="Plan")],
            current_mode="plan",
        )

        persona._sync_awareness_config()

        assert persona.get_model() == "opus"
        assert MODE_CONTROL_ID in {s.id for s in persona.get_setting_configurations()}

    def test_legacy_models_fill_model_configuration(self):
        # kiro-cli advertises models on the raw session/new response instead of
        # a config option; the captured payload backs the model picker.
        persona = _awareness_persona(
            legacy_models={
                "currentModelId": "claude-sonnet-5",
                "availableModels": [
                    {"modelId": "auto", "name": "auto", "description": "Chosen by task"},
                    {"modelId": "claude-sonnet-5", "name": "claude-sonnet-5"},
                ],
            }
        )

        model, settings = persona._build_awareness_config()

        assert model.current == "claude-sonnet-5"
        assert [(o.id, o.name, o.description) for o in model.options] == [
            ("auto", "auto", "Chosen by task"),
            ("claude-sonnet-5", "claude-sonnet-5", None),
        ]
        assert settings == []

    def test_config_option_model_wins_over_legacy_models(self):
        persona = _awareness_persona(
            config_options=[
                _select_option("model", "opus", ["opus", "haiku"], category="model")
            ],
            legacy_models={
                "currentModelId": "auto",
                "availableModels": [{"modelId": "auto", "name": "auto"}],
            },
        )

        model, _ = persona._build_awareness_config()

        assert model.current == "opus"
        assert [o.id for o in model.options] == ["opus", "haiku"]

    def test_malformed_legacy_model_entries_are_skipped(self):
        persona = _awareness_persona(
            legacy_models={
                "currentModelId": "m1",
                "availableModels": ["not-a-dict", {"name": "no-id"}, {"modelId": "m1"}],
            }
        )

        model, _ = persona._build_awareness_config()

        assert [(o.id, o.name) for o in model.options] == [("m1", "m1")]


class TestSyncAwarenessUsage:
    """ACP usage state -> awareness Usage model."""

    def test_context_and_cost_mapped(self):
        persona = _awareness_persona(
            context=UsageUpdate(
                sessionUpdate="usage_update",
                used=41_000,
                size=200_000,
                cost=Cost(amount=0.41, currency="USD"),
            )
        )

        persona._sync_awareness_usage()

        usage = persona.get_usage()
        assert usage.context_tokens == 41_000
        assert usage.context_size == 200_000
        assert usage.cost_amount == 0.41
        assert usage.cost_currency == "USD"

    def test_session_tokens_mapped(self):
        persona = _awareness_persona(
            session_usage=Usage(
                inputTokens=900,
                outputTokens=340,
                totalTokens=1_240,
                cachedReadTokens=100,
                thoughtTokens=50,
            )
        )

        persona._sync_awareness_usage()

        usage = persona.get_usage()
        assert usage.input_tokens == 900
        assert usage.output_tokens == 340
        assert usage.total_tokens == 1_240
        assert usage.cached_read_tokens == 100
        assert usage.thought_tokens == 50

    def test_nothing_reported_leaves_usage_empty(self):
        persona = _awareness_persona()

        persona._sync_awareness_usage()

        usage = persona.get_usage()
        assert usage.context_tokens is None
        assert usage.total_tokens is None

    def test_percent_only_context_mapped(self):
        persona = _awareness_persona(context_percent=1.48)

        persona._sync_awareness_usage()

        usage = persona.get_usage()
        assert usage.context_percent == 1.48
        assert usage.context_tokens is None
        assert usage.context_size is None


# These `update_*` methods are thin: they tell the ACP session to switch and
# nothing more. `BasePersona` decides what changed (passing only changed keys),
# records the new current values, and rebroadcasts — so these no longer filter,
# sync awareness, or swallow errors.

class TestUpdateModel:
    """update_model applies the choice through the backing model config option
    (models are config options now; session/set_model was removed)."""

    async def test_applies_model_category_option(self):
        persona = _awareness_persona(
            config_options=[
                _select_option("chosen_model", "gpt", ["gpt", "claude"], category="model")
            ]
        )
        persona.set_acp_config_option = AsyncMock()

        await BaseAcpPersona.update_model(persona, "claude")

        # Routes to the model option by its actual id, not a hardcoded "model".
        persona.set_acp_config_option.assert_awaited_once_with("chosen_model", "claude")

    async def test_falls_back_to_model_id_when_none_advertised(self):
        persona = _awareness_persona(config_options=[])
        persona.set_acp_config_option = AsyncMock()

        await BaseAcpPersona.update_model(persona, "claude")

        persona.set_acp_config_option.assert_awaited_once_with("model", "claude")

    async def test_uses_first_model_option_when_multiple(self):
        # Several category="model" options: apply through the FIRST (the one the
        # tie-break surfaced as the prominent model picker), never a later one.
        persona = _awareness_persona(
            config_options=[
                _select_option("model_a", "a1", ["a1", "a2"], category="model"),
                _select_option("model_b", "b1", ["b1", "b2"], category="model"),
            ]
        )
        persona.set_acp_config_option = AsyncMock()

        await BaseAcpPersona.update_model(persona, "a2")

        persona.set_acp_config_option.assert_awaited_once_with("model_a", "a2")

    async def test_legacy_models_use_set_session_model(self):
        persona = _awareness_persona(
            legacy_models={
                "currentModelId": "claude-sonnet-5",
                "availableModels": [{"modelId": "auto", "name": "auto"}],
            }
        )
        client = MagicMock()
        client.set_session_model = AsyncMock()
        persona.get_client = AsyncMock(return_value=client)
        persona.get_session_id = AsyncMock(return_value="sess-9")
        persona.set_acp_config_option = AsyncMock()

        await BaseAcpPersona.update_model(persona, "auto")

        client.set_session_model.assert_awaited_once_with("auto", "sess-9")
        # The choice goes through the legacy request, not a config option.
        persona.set_acp_config_option.assert_not_awaited()
        # The stored payload tracks the new current model for future rebuilds.
        assert persona._acp_legacy_models["currentModelId"] == "auto"

    async def test_model_config_option_wins_over_legacy_models(self):
        persona = _awareness_persona(
            config_options=[
                _select_option("model", "opus", ["opus", "haiku"], category="model")
            ],
            legacy_models={
                "currentModelId": "auto",
                "availableModels": [{"modelId": "auto", "name": "auto"}],
            },
        )
        persona.set_acp_config_option = AsyncMock()
        persona.get_client = AsyncMock()

        await BaseAcpPersona.update_model(persona, "haiku")

        persona.set_acp_config_option.assert_awaited_once_with("model", "haiku")
        persona.get_client.assert_not_awaited()


class TestUpdateModelSettings:
    """update_model_settings applies each given setting as a config option."""

    async def test_applies_each_given_setting(self):
        persona = _awareness_persona(
            config_options=[
                _select_option(
                    "context_size", "small", ["small", "large"], category="model_config"
                ),
            ]
        )
        persona.set_acp_config_option = AsyncMock()

        await BaseAcpPersona.update_model_settings(persona, {"context_size": "large"})

        persona.set_acp_config_option.assert_awaited_once_with("context_size", "large")


class TestUpdateSettings:
    """update_settings routes mode and config options to the right setters."""

    # The mode pseudo-setting has three routing branches, exercised below in the
    # order of increasing ambiguity. The rule: prefer a `category="mode"` config
    # option (the FIRST one) over the dedicated `session/set_mode` state.

    async def test_mode_only_set_mode_state_calls_set_acp_mode(self):
        # Branch 1 — mode advertised only via the dedicated set_mode state:
        # route to session/set_mode.
        persona = _awareness_persona(
            modes=[SessionMode(id="plan", name="Plan")], current_mode="plan"
        )
        persona.set_acp_mode = AsyncMock()
        persona.set_acp_config_option = AsyncMock()

        await BaseAcpPersona.update_settings(persona, {MODE_CONTROL_ID: "code"})

        persona.set_acp_mode.assert_awaited_once_with("code")
        persona.set_acp_config_option.assert_not_awaited()

    async def test_mode_config_option_only_calls_set_config_option(self):
        # Mode advertised only as a config option -> session/set_config_option on
        # that option's id, not session/set_mode.
        persona = _awareness_persona(
            config_options=[
                _select_option("mode", "ask", ["ask", "code"], category="mode")
            ]
        )
        persona.set_acp_mode = AsyncMock()
        persona.set_acp_config_option = AsyncMock()

        await BaseAcpPersona.update_settings(persona, {MODE_CONTROL_ID: "code"})

        persona.set_acp_config_option.assert_awaited_once_with("mode", "code")
        persona.set_acp_mode.assert_not_awaited()

    async def test_mode_both_channels_prefers_config_option(self):
        # Branch 2 — set_mode state AND a config option: prefer the config option,
        # routing to session/set_config_option (never session/set_mode).
        persona = _awareness_persona(
            modes=[SessionMode(id="plan", name="Plan")],
            current_mode="plan",
            config_options=[
                _select_option("mode_cfg", "ask", ["ask", "code"], category="mode")
            ],
        )
        persona.set_acp_mode = AsyncMock()
        persona.set_acp_config_option = AsyncMock()

        await BaseAcpPersona.update_settings(persona, {MODE_CONTROL_ID: "code"})

        persona.set_acp_config_option.assert_awaited_once_with("mode_cfg", "code")
        persona.set_acp_mode.assert_not_awaited()

    async def test_mode_both_channels_multiple_options_uses_first(self):
        # Branch 3 (worst case) — set_mode state AND several category="mode"
        # config options: prefer the FIRST config option and route there.
        persona = _awareness_persona(
            modes=[SessionMode(id="plan", name="Plan")],
            current_mode="plan",
            config_options=[
                _select_option("mode_a", "ask", ["ask", "code"], category="mode"),
                _select_option("mode_b", "fast", ["fast", "slow"], category="mode"),
            ],
        )
        persona.set_acp_mode = AsyncMock()
        persona.set_acp_config_option = AsyncMock()

        await BaseAcpPersona.update_settings(persona, {MODE_CONTROL_ID: "code"})

        persona.set_acp_config_option.assert_awaited_once_with("mode_a", "code")
        persona.set_acp_mode.assert_not_awaited()

    async def test_config_option_calls_set_acp_config_option(self):
        persona = _awareness_persona(
            config_options=[_select_option("effort", "high", ["low", "high"])]
        )
        persona.set_acp_config_option = AsyncMock()

        await BaseAcpPersona.update_settings(persona, {"effort": "low"})

        persona.set_acp_config_option.assert_awaited_once_with("effort", "low")

    async def test_boolean_value_coerced_to_bool(self):
        persona = _awareness_persona(config_options=[_bool_option("allow_all", False)])
        persona.set_acp_config_option = AsyncMock()

        await BaseAcpPersona.update_settings(persona, {"allow_all": "true"})

        persona.set_acp_config_option.assert_awaited_once_with("allow_all", True)
