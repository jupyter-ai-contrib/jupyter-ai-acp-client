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
    ModelInfo,
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
    models=None,
    current_model=None,
    modes=None,
    current_mode=None,
    config_options=None,
    context=None,
    session_usage=None,
):
    """A real `BaseAcpPersona` built without `__init__`, carrying the raw ACP
    state the awareness mapping reads plus a real awareness state to broadcast
    into. Collaborators are mocked."""

    class _Concrete(BaseAcpPersona):
        @property
        def defaults(self):  # pragma: no cover - never called
            return None

    persona = _Concrete.__new__(_Concrete)
    persona.log = logging.getLogger("test")
    persona.awareness = _awareness()
    persona.ychat = MagicMock()
    persona._acp_models = models or []
    persona._acp_current_model_id = current_model
    persona._acp_modes = modes or []
    persona._acp_current_mode_id = current_mode
    persona._acp_config_options = config_options or []
    persona._acp_context_usage = context
    persona._acp_session_usage = session_usage
    return persona


class TestBuildAwarenessConfig:
    """ACP model/mode/config options -> ModelConfiguration + settings list."""

    def test_dedicated_model_becomes_model_configuration(self):
        persona = _awareness_persona(
            models=[
                ModelInfo(modelId="opus", name="Opus", description="big"),
                ModelInfo(modelId="haiku", name="Haiku"),
            ],
            current_model="opus",
        )

        model, settings = persona._build_awareness_config()

        assert model.current == "opus"
        assert [(o.id, o.name, o.description) for o in model.options] == [
            ("opus", "Opus", "big"),
            ("haiku", "Haiku", None),
        ]
        assert settings == []

    def test_model_config_category_becomes_model_settings(self):
        persona = _awareness_persona(
            models=[ModelInfo(modelId="opus", name="Opus")],
            current_model="opus",
            config_options=[
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

    def test_config_model_and_mode_dropped_when_dedicated_fields_present(self):
        persona = _awareness_persona(
            models=[ModelInfo(modelId="opus", name="Opus")],
            current_model="opus",
            modes=[SessionMode(id="plan", name="Plan")],
            current_mode="plan",
            config_options=[
                _select_option("model", "opus", ["opus"]),
                _select_option("mode", "plan", ["plan"]),
                _select_option("effort", "high", ["low", "high"]),
            ],
        )

        model, settings = persona._build_awareness_config()

        # dedicated model wins; config "model"/"mode" duplicates dropped.
        assert [o.id for o in model.options] == ["opus"]
        assert {s.id for s in settings} == {MODE_CONTROL_ID, "effort"}

    def test_config_model_fallback_when_no_dedicated_models(self):
        persona = _awareness_persona(
            config_options=[
                _select_option("model", "gpt", ["gpt", "claude"]),
            ]
        )

        model, settings = persona._build_awareness_config()

        # Falls back to the "model" config option for the model configuration.
        assert model.current == "gpt"
        assert [o.id for o in model.options] == ["gpt", "claude"]
        # And it is not also duplicated into general settings.
        assert settings == []

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
            models=[ModelInfo(modelId="opus", name="Opus")],
            current_model="opus",
            modes=[SessionMode(id="plan", name="Plan")],
            current_mode="plan",
        )

        persona._sync_awareness_config()

        assert persona.get_model() == "opus"
        assert MODE_CONTROL_ID in {s.id for s in persona.get_setting_configurations()}


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


# These `update_*` methods are thin: they tell the ACP session to switch and
# nothing more. `BasePersona` decides what changed (passing only changed keys),
# records the new current values, and rebroadcasts — so these no longer filter,
# sync awareness, or swallow errors.

class TestUpdateModel:
    """update_model dispatches to the ACP model setter."""

    async def test_calls_set_acp_model(self):
        persona = MagicMock()
        persona.set_acp_model = AsyncMock()

        await BaseAcpPersona.update_model(persona, "opus")

        persona.set_acp_model.assert_awaited_once_with("opus")


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

    async def test_mode_pseudo_setting_calls_set_acp_mode(self):
        persona = _awareness_persona(
            modes=[SessionMode(id="plan", name="Plan")], current_mode="plan"
        )
        persona.set_acp_mode = AsyncMock()
        persona.set_acp_config_option = AsyncMock()

        await BaseAcpPersona.update_settings(persona, {MODE_CONTROL_ID: "code"})

        persona.set_acp_mode.assert_awaited_once_with("code")
        persona.set_acp_config_option.assert_not_awaited()

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
