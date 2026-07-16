"""
Kiro-specific ACP client and the typed models for the vendor payloads it reads.

kiro-cli's ACP surface (its v2 engine, the default) diverges from the ACP v1
spec in two ways this module handles, keeping the generic `JaiAcpClient`
agent-agnostic:

1. **Models.** Kiro advertises its selectable models through a top-level
   ``models`` field on the raw ``session/new`` / ``session/load`` response
   (``currentModelId`` + ``availableModels``) instead of ACP v1 config options,
   and it still serves the deprecated ``session/set_model`` request to switch
   them. The typed SDK response models drop unknown fields, so `KiroAcpClient`
   issues those two session RPCs itself and reads ``models`` off the raw dict.

2. **Usage & commands.** Kiro reports context fill as a bare percentage (plus a
   Kiro-credit cost) over the ``_kiro.dev/metadata`` vendor notification, and its
   slash commands over ``_kiro.dev/commands/available`` — neither is a standard
   ACP ``session/update``. `KiroAcpClient.ext_notification` maps both onto the
   persona-manager awareness API. (The ACP SDK strips the leading underscore, so
   the methods arrive here as ``kiro.dev/metadata`` / ``kiro.dev/commands/available``.)

The Pydantic models below type these payloads defensively: **every field is
optional** so a shape change never raises, and ``mode="before"`` validators drop
values of the wrong type (rather than letting Pydantic coerce e.g. the string
``"1.25"`` or ``True`` into a float) so malformed input is ignored, not
misread.
"""

from __future__ import annotations

from typing import Any, Optional

from acp.meta import AGENT_METHODS
from acp.schema import (
    LoadSessionRequest,
    LoadSessionResponse,
    NewSessionRequest,
    NewSessionResponse,
)
from acp.utils import serialize_params, validate_model, validate_model_from_dict
from jupyter_ai_persona_manager import CommandOption
from jupyter_ai_persona_manager import Usage as AwarenessUsage
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..default_acp_client import JaiAcpClient


def _str_or_none(value: Any) -> Any:
    """Keep a value only if it's a string; otherwise drop it to ``None`` so a
    mistyped field never raises during validation."""
    return value if isinstance(value, str) else None


class KiroModelOption(BaseModel):
    """One selectable model in Kiro's legacy ``models`` payload."""

    model_config = ConfigDict(
        populate_by_name=True, extra="ignore", protected_namespaces=()
    )

    model_id: Optional[str] = Field(default=None, alias="modelId")
    name: Optional[str] = None
    description: Optional[str] = None

    _coerce_strings = field_validator(
        "model_id", "name", "description", mode="before"
    )(_str_or_none)


class KiroModels(BaseModel):
    """
    The ``models`` block Kiro attaches to a raw session response: the current
    model plus the list of available models.
    """

    model_config = ConfigDict(
        populate_by_name=True, extra="ignore", protected_namespaces=()
    )

    current_model_id: Optional[str] = Field(default=None, alias="currentModelId")
    available_models: Optional[list[KiroModelOption]] = Field(
        default=None, alias="availableModels"
    )

    _coerce_current = field_validator("current_model_id", mode="before")(_str_or_none)

    @field_validator("available_models", mode="before")
    @classmethod
    def _keep_dict_entries(cls, value: Any) -> Any:
        """Drop the whole list if it isn't one; drop any non-dict entries."""
        if not isinstance(value, list):
            return None
        return [item for item in value if isinstance(item, dict)]


class KiroMeteringUsage(BaseModel):
    """One metered-cost entry in a ``_kiro.dev/metadata`` notification."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    value: Optional[float] = None
    unit: Optional[str] = None
    unit_plural: Optional[str] = Field(default=None, alias="unitPlural")

    _coerce_strings = field_validator("unit", "unit_plural", mode="before")(
        _str_or_none
    )

    @field_validator("value", mode="before")
    @classmethod
    def _real_number_only(cls, value: Any) -> Any:
        """Accept only a genuine int/float (not bool/str); else ignore it."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)


class KiroMetadata(BaseModel):
    """
    A ``_kiro.dev/metadata`` vendor notification: context fill as a percentage
    and an optional per-turn metered cost (in Kiro credits).
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    session_id: Optional[str] = Field(default=None, alias="sessionId")
    context_usage_percentage: Optional[float] = Field(
        default=None, alias="contextUsagePercentage"
    )
    metering_usage: Optional[list[KiroMeteringUsage]] = Field(
        default=None, alias="meteringUsage"
    )
    turn_duration_ms: Optional[float] = Field(default=None, alias="turnDurationMs")

    _coerce_session_id = field_validator("session_id", mode="before")(_str_or_none)

    @field_validator("context_usage_percentage", mode="before")
    @classmethod
    def _real_number_only(cls, value: Any) -> Any:
        """
        Accept only a genuine int/float. Guards against Pydantic coercing a
        string like ``"1.25"`` or a ``bool`` into a float, so a malformed
        percentage is ignored rather than misread.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    @field_validator("metering_usage", mode="before")
    @classmethod
    def _keep_dict_entries(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return None
        return [item for item in value if isinstance(item, dict)]


class KiroCommand(BaseModel):
    """One slash command in a ``_kiro.dev/commands/available`` notification."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    name: Optional[str] = None
    description: Optional[str] = None
    meta: Optional[dict] = None

    _coerce_strings = field_validator("name", "description", mode="before")(
        _str_or_none
    )


class KiroCommands(BaseModel):
    """A ``_kiro.dev/commands/available`` vendor notification."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    session_id: Optional[str] = Field(default=None, alias="sessionId")
    commands: Optional[list[KiroCommand]] = None

    _coerce_session_id = field_validator("session_id", mode="before")(_str_or_none)

    @field_validator("commands", mode="before")
    @classmethod
    def _keep_named_entries(cls, value: Any) -> Any:
        """Keep only dict entries carrying a non-empty string ``name``."""
        if not isinstance(value, list):
            return None
        return [
            item
            for item in value
            if isinstance(item, dict)
            and isinstance(item.get("name"), str)
            and item.get("name")
        ]


class KiroAcpClient(JaiAcpClient):
    """
    A `JaiAcpClient` scoped to `KiroAcpPersona` that handles kiro-cli's
    non-standard model, usage, and command surfaces. Only `KiroAcpPersona`
    constructs this (via its ``acp_client_class``); the generic client stays
    agent-agnostic.
    """

    async def create_session(self, persona) -> NewSessionResponse:
        """
        Create a session and capture Kiro's legacy ``models`` payload.

        Sends ``session/new`` as a raw request (rather than the typed
        ``conn.new_session``) because the typed `NewSessionResponse` drops the
        top-level ``models`` field. The parsed models are pushed straight onto
        the persona, which reads them when it builds its awareness config.
        """
        conn = await self.get_connection()
        mcp_servers = await self._get_mcp_servers(persona)
        raw = await conn._conn.send_request(
            AGENT_METHODS["session_new"],
            serialize_params(
                NewSessionRequest(
                    cwd=persona.get_chat_dir(), mcp_servers=mcp_servers or []
                )
            ),
        )
        persona.set_kiro_models(self._parse_models(raw))
        response = validate_model(raw, NewSessionResponse)
        self._personas_by_session[response.session_id] = persona
        return response

    async def _load_session_rpc(self, persona, session_id: str) -> LoadSessionResponse:
        """
        Perform the raw ``session/load`` RPC and capture Kiro's legacy
        ``models``. Overrides the raw call site, not the public
        ``load_session`` (whose in-flight dedup wrapper is generic). A
        ``session/load`` response carries no ``sessionId``, so the persona is
        keyed under the requested ID.
        """
        conn = await self.get_connection()
        mcp_servers = await self._get_mcp_servers(persona)
        raw = await conn._conn.send_request(
            AGENT_METHODS["session_load"],
            serialize_params(
                LoadSessionRequest(
                    cwd=persona.get_chat_dir(),
                    session_id=session_id,
                    mcp_servers=mcp_servers or [],
                )
            ),
        )
        persona.set_kiro_models(self._parse_models(raw))
        response = validate_model_from_dict(raw, LoadSessionResponse)
        self._personas_by_session[session_id] = persona
        return response

    @staticmethod
    def _parse_models(raw: Any) -> Optional[KiroModels]:
        """Parse the legacy ``models`` block off a raw session response, if any."""
        if isinstance(raw, dict) and isinstance(raw.get("models"), dict):
            return KiroModels.model_validate(raw["models"])
        return None

    async def set_session_model(self, model_id: str, session_id: str) -> None:
        """
        Switch the session's model via the deprecated ``session/set_model``
        request. Removed from ACP v1 but still served by kiro-cli's v2 engine.
        """
        conn = await self.get_connection()
        await conn._conn.send_request(
            "session/set_model", {"sessionId": session_id, "modelId": model_id}
        )

    async def ext_notification(self, method: str, params: dict) -> None:
        """
        Handle Kiro's vendor notifications, mapping them onto the awareness API
        of the notification's own session persona only (per-session isolation),
        and delegate everything else to the generic handler.
        """
        if method == "kiro.dev/metadata":
            meta = KiroMetadata.model_validate(params)
            persona = self._personas_by_session.get(meta.session_id)
            if persona is None:
                return
            usage = AwarenessUsage()
            if meta.context_usage_percentage is not None:
                usage.context_percent = meta.context_usage_percentage
            if meta.metering_usage:
                # Kiro reports cost in credits, so `cost_currency` carries the
                # unit token ("credit") rather than an ISO-4217 code — a
                # deliberate stretch of the awareness `Usage` field so the
                # metered cost can surface. Sum values in case several arrive.
                amounts = [m.value for m in meta.metering_usage if m.value is not None]
                if amounts:
                    usage.cost_amount = sum(amounts)
                    usage.cost_currency = meta.metering_usage[0].unit
            usage_provided = usage.model_dump(exclude_none=True)
            if usage_provided:
                persona.report_usage(usage)
            return
        if method == "kiro.dev/commands/available":
            cmds = KiroCommands.model_validate(params)
            persona = self._personas_by_session.get(cmds.session_id)
            if persona is None:
                return
            options = [
                CommandOption(
                    name=c.name if c.name.startswith("/") else "/" + c.name,
                    description=c.description,
                )
                for c in (cmds.commands or [])
                if c.name
            ]
            if options:
                persona.report_slash_commands(options)
            return
        return await super().ext_notification(method, params)
