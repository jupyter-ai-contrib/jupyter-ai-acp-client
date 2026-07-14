from __future__ import annotations

from typing import TYPE_CHECKING

from jupyter_server.base.handlers import APIHandler
import tornado
from pydantic import BaseModel
from acp.schema import SessionConfigOptionBoolean, SessionConfigOptionSelect
from .base_acp_persona import (
    MODE_CONTROL_ID,
    BaseAcpPersona,
    _flatten_select_options,
)

if TYPE_CHECKING:
    from jupyter_server_fileid.manager import BaseFileIdManager
    from jupyter_ai_persona_manager import PersonaManager

class AcpSlashCommand(BaseModel):
    name: str
    description: str

class AcpSlashCommandsResponse(BaseModel):
    commands: list[AcpSlashCommand] = []

class AcpSlashCommandsHandler(APIHandler):
    @property
    def file_id_manager(self) -> BaseFileIdManager:
        manager = self.serverapp.web_app.settings["file_id_manager"]
        assert manager
        return manager
    
    @tornado.web.authenticated
    def get(self, persona_mention_name: str = ""):
        # get chat path
        chat_path = self.get_argument('chat_path', None)
        if not chat_path:
            # raise HTTP error: chat_path is required URL query arg
            raise tornado.web.HTTPError(400, "chat_path is required as a URL query parameter")
        
        # get chat room ID using file ID manager
        file_id = self.file_id_manager.get_id(chat_path)
        if not file_id:
            raise tornado.web.HTTPError(404, f"Chat not found: {chat_path}")
        room_id = f"text:chat:{file_id}"
        
        # get persona manager
        persona_manager: PersonaManager | None = self.serverapp.web_app.settings.get("jupyter-ai", {}).get("persona-managers", {}).get(room_id, None)
        if not persona_manager:
            raise tornado.web.HTTPError(404, f"Chat not initialized: {chat_path}")

        persona = None
        if persona_mention_name:
            for p in persona_manager.personas.values():
                if p.as_user().mention_name == persona_mention_name:
                    persona = p
                    break
            if not persona:
                # raise HTTP error: persona not found
                raise tornado.web.HTTPError(404, f"Persona not found: @{persona_mention_name}")
        else:
            persona = persona_manager.default_persona

        # Return early with empty response if either:
        # 1. no default persona, or
        # 2. default persona is not ACP persona, or
        # 3. mentioned persona is not ACP persona.
        if not isinstance(persona, BaseAcpPersona):
            self.finish(AcpSlashCommandsResponse().model_dump())
            return
        
        # Otherwise get the ACP slash commands from the persona.
        # Convert slash commands to the response format
        commands = []
        for cmd in persona.acp_slash_commands:
            name = cmd.name if cmd.name.startswith("/") else "/" + cmd.name
            commands.append(
                AcpSlashCommand(
                    name=name,
                    description=cmd.description
                )
            )

        response = AcpSlashCommandsResponse(commands=commands)
        self.finish(response.model_dump())


MODEL_CONTROL_ID = "__model__"

class AcpControlChoice(BaseModel):
    value: str
    label: str
    description: str | None = None

class AcpControl(BaseModel):
    # Stable control id: "__model__", "__mode__", or the config option id.
    id: str
    # Which ACP mechanism backs this control: "model" | "mode" | "config_option".
    source: str
    # "select" (a dropdown of choices) or "boolean" (a toggle).
    kind: str
    # Human-readable label shown on the control.
    label: str
    # Currently selected value: a choice value (select) or a bool (boolean).
    current_value: str | bool | None = None
    # Available choices, for selects.
    choices: list[AcpControlChoice] = []

def build_controls(persona: BaseAcpPersona) -> list[AcpControl]:
    """
    Normalize an ACP persona's model, mode, and config options into one uniform
    list of toolbar controls. Model and mode are surfaced as selects; config
    options keep their declared select/boolean kind.
    """
    controls: list[AcpControl] = []

    if persona.acp_models:
        controls.append(AcpControl(
            id=MODEL_CONTROL_ID,
            source="model",
            kind="select",
            label="Model",
            current_value=persona.acp_current_model_id,
            choices=[
                AcpControlChoice(value=m.model_id, label=m.name, description=m.description)
                for m in persona.acp_models
            ],
        ))

    if persona.acp_modes:
        controls.append(AcpControl(
            id=MODE_CONTROL_ID,
            source="mode",
            kind="select",
            label="Mode",
            current_value=persona.acp_current_mode_id,
            choices=[
                AcpControlChoice(value=m.id, label=m.name, description=m.description)
                for m in persona.acp_modes
            ],
        ))

    # Some agents (e.g. Copilot) advertise model and mode both as dedicated
    # fields and as config options. Skip the config-option duplicates when the
    # dedicated control already covers them, but keep them for agents (e.g.
    # OpenCode) that only expose model/mode through config options.
    has_model = bool(persona.acp_models)
    has_mode = bool(persona.acp_modes)

    for opt in persona.acp_config_options:
        if opt.id == "model" and has_model:
            continue
        if opt.id == "mode" and has_mode:
            continue
        if isinstance(opt, SessionConfigOptionSelect):
            controls.append(AcpControl(
                id=opt.id,
                source="config_option",
                kind="select",
                label=opt.name,
                current_value=opt.current_value,
                choices=[
                    AcpControlChoice(value=c.value, label=c.name, description=c.description)
                    for c in _flatten_select_options(opt.options)
                ],
            ))
        elif isinstance(opt, SessionConfigOptionBoolean):
            controls.append(AcpControl(
                id=opt.id,
                source="config_option",
                kind="boolean",
                label=opt.name,
                current_value=opt.current_value,
            ))

    return controls

class _ChatScopedHandler(APIHandler):
    """
    Base for handlers scoped to a single chat. Resolves the chat's
    `PersonaManager` from the required `chat_path` query argument.
    """

    @property
    def file_id_manager(self) -> BaseFileIdManager:
        manager = self.serverapp.web_app.settings["file_id_manager"]
        assert manager
        return manager

    def _get_persona_manager(self) -> "PersonaManager":
        chat_path = self.get_argument('chat_path', None)
        if not chat_path:
            raise tornado.web.HTTPError(400, "chat_path is required as a URL query parameter")
        file_id = self.file_id_manager.get_id(chat_path)
        if not file_id:
            raise tornado.web.HTTPError(404, f"Chat not found: {chat_path}")
        room_id = f"text:chat:{file_id}"
        persona_manager = self.serverapp.web_app.settings.get("jupyter-ai", {}).get("persona-managers", {}).get(room_id, None)
        if not persona_manager:
            raise tornado.web.HTTPError(404, f"Chat not initialized: {chat_path}")
        return persona_manager

class PersonaInfo(BaseModel):
    id: str
    name: str
    mention_name: str
    is_acp: bool
    avatar_url: str | None = None

class AcpContextUsage(BaseModel):
    # Tokens currently in the agent's context window.
    used: int
    # Total context window size in tokens.
    size: int

class AcpTokenUsage(BaseModel):
    # All values are cumulative across the session, not per turn.
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_read_tokens: int | None = None
    cached_write_tokens: int | None = None
    thought_tokens: int | None = None

class AcpCostUsage(BaseModel):
    # Cumulative session cost, as reported by the agent.
    amount: float
    currency: str

class AcpUsage(BaseModel):
    # Each field is None when the agent has not reported that quantity.
    context: AcpContextUsage | None = None
    tokens: AcpTokenUsage | None = None
    cost: AcpCostUsage | None = None

def build_usage(persona: BaseAcpPersona) -> AcpUsage:
    """
    Collect the persona's reported usage: context fill and cost from the latest
    `usage_update`, cumulative session tokens from the latest prompt response.
    Fields the agent has not reported stay None.
    """
    usage = AcpUsage()
    context = persona.acp_context_usage
    if context is not None:
        usage.context = AcpContextUsage(used=context.used, size=context.size)
        if context.cost is not None:
            usage.cost = AcpCostUsage(
                amount=context.cost.amount, currency=context.cost.currency
            )
    tokens = persona.acp_session_usage
    if tokens is not None:
        usage.tokens = AcpTokenUsage(
            input_tokens=tokens.input_tokens,
            output_tokens=tokens.output_tokens,
            total_tokens=tokens.total_tokens,
            cached_read_tokens=tokens.cached_read_tokens,
            cached_write_tokens=tokens.cached_write_tokens,
            thought_tokens=tokens.thought_tokens,
        )
    return usage

class PersonasResponse(BaseModel):
    # Every persona in the chat, for the selector.
    personas: list[PersonaInfo] = []
    # The session controls (model, mode, config options) of the persona named in
    # the `persona_id` query arg, normalized for the input toolbar, when it is an
    # ACP persona. Empty otherwise.
    controls: list[AcpControl] = []
    # The selected persona's reported usage (context fill, session tokens, cost),
    # when it is an ACP persona. Quantities the agent has not reported are None.
    usage: AcpUsage = AcpUsage()

class PersonasHandler(_ChatScopedHandler):
    """
    REST endpoint backing the persona selector and its session controls. GET
    returns the chat's personas and — for the persona named by the optional
    `persona_id` query arg (defaulting to the chat's default persona) — that
    persona's session controls, when it is an ACP persona.

    The endpoint no longer tracks an "active" persona: which persona a message
    is routed to now lives in each message's metadata (stamped by the frontend),
    so this endpoint only advertises what's available and configurable.
    """

    def _resolve_persona(self, persona_manager: "PersonaManager"):
        """The persona whose controls to serve: the one named by `persona_id`,
        else the chat's default persona. Returns None if neither resolves.

        A `persona_id` naming a persona that isn't installed here (e.g. a default
        advertised by the server but not present in this environment) is not an
        error: it just means "no controls to show". We must still return the
        persona list so the selector can render, so this never raises."""
        persona_id = self.get_argument("persona_id", None)
        if persona_id:
            return persona_manager.personas.get(persona_id)
        return persona_manager.default_persona

    @tornado.web.authenticated
    def get(self, _unused: str = ""):
        persona_manager = self._get_persona_manager()
        personas = [
            PersonaInfo(
                id=p.id,
                name=p.name,
                mention_name=p.as_user().mention_name,
                is_acp=isinstance(p, BaseAcpPersona),
                avatar_url=p.as_user().avatar_url,
            )
            for p in persona_manager.personas.values()
        ]
        persona = self._resolve_persona(persona_manager)
        controls: list[AcpControl] = []
        usage = AcpUsage()
        if isinstance(persona, BaseAcpPersona):
            controls = build_controls(persona)
            usage = build_usage(persona)
        response = PersonasResponse(
            personas=personas, controls=controls, usage=usage
        )
        self.finish(response.model_dump())


class AcpControlHandler(_ChatScopedHandler):
    """
    REST endpoint that sets a single session control on an ACP persona. POST
    body: {persona_id, control_id, source, value}. `source` selects the ACP
    mechanism: "model" and "mode" take the chosen id as `value`; "config_option"
    takes the option `control_id` and its new select value or boolean.
    """

    @tornado.web.authenticated
    async def post(self, _unused: str = ""):
        persona_manager = self._get_persona_manager()
        body = self.get_json_body() or {}
        persona_id = body.get("persona_id")
        control_id = body.get("control_id")
        source = body.get("source")
        value = body.get("value")
        if not persona_id or not control_id or not source or value is None:
            raise tornado.web.HTTPError(
                400,
                "Missing required fields: persona_id, control_id, source, value",
            )

        persona = persona_manager.personas.get(persona_id)
        if not isinstance(persona, BaseAcpPersona):
            raise tornado.web.HTTPError(
                404, f"No ACP persona to configure: {persona_id}"
            )

        if source == "model":
            await persona.set_acp_model(value)
        elif source == "mode":
            await persona.set_acp_mode(value)
        elif source == "config_option":
            await persona.set_acp_config_option(control_id, value)
        else:
            raise tornado.web.HTTPError(400, f"Unknown control source: {source}")

        self.finish({"status": "ok", "control_id": control_id, "value": value})


class StopStreamingHandler(APIHandler):
    @property
    def file_id_manager(self) -> BaseFileIdManager:
        manager = self.serverapp.web_app.settings["file_id_manager"]
        assert manager
        return manager

    @tornado.web.authenticated
    async def post(self, persona_mention_name: str = ""):
        chat_path = self.get_argument('chat_path', None)
        if not chat_path:
            raise tornado.web.HTTPError(400, "chat_path is required as a URL query parameter")

        file_id = self.file_id_manager.get_id(chat_path)
        if not file_id:
            raise tornado.web.HTTPError(404, f"Chat not found: {chat_path}")
        room_id = f"text:chat:{file_id}"

        persona_manager: PersonaManager | None = self.serverapp.web_app.settings.get("jupyter-ai", {}).get("persona-managers", {}).get(room_id, None)
        if not persona_manager:
            raise tornado.web.HTTPError(404, f"Chat not initialized: {chat_path}")

        # Stop all ACP personas in this chat
        stopped = []
        for p in persona_manager.personas.values():
            if not isinstance(p, BaseAcpPersona):
                continue
            try:
                client = await p.get_client()
                session_id = await p.get_session_id()
                if session_id:
                    await client.stop_streaming(session_id)
                    stopped.append(p.as_user().mention_name)
            except Exception:
                pass
        self.finish({"status": "stopped", "stopped": stopped})


class PermissionHandler(APIHandler):
    """
    REST endpoint for permission decisions. The frontend POSTs the user's
    button click here, and this handler finds the right JaiAcpClient and
    resolves the pending Future so the suspended request_permission coroutine
    can resume.
    """

    async def _find_client_for_session(self, session_id: str):
        """
        Iterate all persona managers → personas to find the JaiAcpClient
        that has a pending permission for the given session_id.
        Returns the client or None.
        """
        import logging
        logger = logging.getLogger(__name__)

        persona_managers = (
            self.serverapp.web_app.settings
            .get("jupyter-ai", {})
            .get("persona-managers", {})
        )
        logger.debug(f"_find_client_for_session: looking for session_id={session_id}, "
                     f"persona_managers count={len(persona_managers)}")
        for room_id, pm in persona_managers.items():
            logger.debug(f"  checking room={room_id}, personas={list(pm.personas.keys())}")
            for persona_id, persona in pm.personas.items():
                if not isinstance(persona, BaseAcpPersona):
                    logger.debug(f"    {persona_id}: not BaseAcpPersona, skipping")
                    continue
                client = await persona.get_client()
                if client.includes_session(session_id):
                    logger.debug(f"    FOUND client for session {session_id}")
                    return client
        logger.debug(f"_find_client_for_session: no client found for session_id={session_id}")
        return None

    @tornado.web.authenticated
    async def post(self):
        body = self.get_json_body()
        if body is None:
            raise tornado.web.HTTPError(400, "Request body required")
        session_id = body.get("session_id")
        tool_call_id = body.get("tool_call_id")
        option_id = body.get("option_id")

        if not all([session_id, tool_call_id, option_id]):
            raise tornado.web.HTTPError(400, "Missing required fields: session_id, tool_call_id, option_id")

        client = await self._find_client_for_session(session_id)
        if not client:
            raise tornado.web.HTTPError(404, "No pending permission request found for this session")

        resolved = client.resolve_permission(session_id, tool_call_id, option_id)
        if not resolved:
            raise tornado.web.HTTPError(404, "No pending permission request found")

        self.finish({"status": "ok"})