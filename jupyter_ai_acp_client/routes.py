from __future__ import annotations

from typing import TYPE_CHECKING

from jupyter_server.base.handlers import APIHandler
import tornado
from pydantic import BaseModel
from .base_acp_persona import BaseAcpPersona

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
            persona = persona_manager.active_persona

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


class AcpModel(BaseModel):
    model_id: str
    name: str
    description: str | None = None

class AcpModelsResponse(BaseModel):
    # Display name of the persona the models belong to, or `None` when no ACP
    # persona is addressed.
    persona: str | None = None
    models: list[AcpModel] = []
    current_model_id: str | None = None

class AcpModelsHandler(APIHandler):
    """
    REST endpoint for the per-persona model selector. GET returns the addressed
    ACP persona's available models and current model; POST sets the model. The
    persona is resolved the same way slash commands are: by mention name, else
    the last-mentioned or default persona.
    """

    @property
    def file_id_manager(self) -> BaseFileIdManager:
        manager = self.serverapp.web_app.settings["file_id_manager"]
        assert manager
        return manager

    def _resolve_persona(self, persona_mention_name: str):
        """
        Resolve the addressed persona from the `chat_path` query arg and an
        optional mention name. Returns the persona, or `None` when the chat
        holds no matching ACP persona. Raises `HTTPError` for bad input.
        """
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

        if persona_mention_name:
            for p in persona_manager.personas.values():
                if p.as_user().mention_name == persona_mention_name:
                    return p if isinstance(p, BaseAcpPersona) else None
            raise tornado.web.HTTPError(404, f"Persona not found: @{persona_mention_name}")

        persona = persona_manager.active_persona
        return persona if isinstance(persona, BaseAcpPersona) else None

    @tornado.web.authenticated
    def get(self, persona_mention_name: str = ""):
        persona = self._resolve_persona(persona_mention_name)
        if persona is None:
            self.finish(AcpModelsResponse().model_dump())
            return

        models = [
            AcpModel(model_id=m.model_id, name=m.name, description=m.description)
            for m in persona.acp_models
        ]
        response = AcpModelsResponse(
            persona=persona.name,
            models=models,
            current_model_id=persona.acp_current_model_id,
        )
        self.finish(response.model_dump())

    @tornado.web.authenticated
    async def post(self, persona_mention_name: str = ""):
        body = self.get_json_body() or {}
        model_id = body.get("model_id")
        if not model_id:
            raise tornado.web.HTTPError(400, "Missing required field: model_id")

        persona = self._resolve_persona(persona_mention_name)
        if persona is None:
            raise tornado.web.HTTPError(404, "No ACP persona to set a model on")

        await persona.set_acp_model(model_id)
        self.finish({"status": "ok", "current_model_id": model_id})


class ActivePersonaInfo(BaseModel):
    id: str
    name: str
    mention_name: str
    is_acp: bool

class ActivePersonaResponse(BaseModel):
    # Every persona in the chat, for the selector.
    personas: list[ActivePersonaInfo] = []
    # The active persona (who replies), or None for "no one".
    active_id: str | None = None
    active_name: str | None = None
    # The active persona's models, when it is an ACP persona.
    models: list[AcpModel] = []
    current_model_id: str | None = None

class ActivePersonaHandler(APIHandler):
    """
    REST endpoint for the active-persona selector. GET returns the chat's
    personas, which one is active, and the active persona's models in one call.
    POST sets the active persona (persona_id null means "no one replies").
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

    @tornado.web.authenticated
    def get(self, _unused: str = ""):
        persona_manager = self._get_persona_manager()
        personas = [
            ActivePersonaInfo(
                id=p.id,
                name=p.name,
                mention_name=p.as_user().mention_name,
                is_acp=isinstance(p, BaseAcpPersona),
            )
            for p in persona_manager.personas.values()
        ]
        active = persona_manager.active_persona
        models: list[AcpModel] = []
        current_model_id = None
        if isinstance(active, BaseAcpPersona):
            models = [
                AcpModel(model_id=m.model_id, name=m.name, description=m.description)
                for m in active.acp_models
            ]
            current_model_id = active.acp_current_model_id
        response = ActivePersonaResponse(
            personas=personas,
            active_id=active.id if active else None,
            active_name=active.name if active else None,
            models=models,
            current_model_id=current_model_id,
        )
        self.finish(response.model_dump())

    @tornado.web.authenticated
    def post(self, _unused: str = ""):
        persona_manager = self._get_persona_manager()
        body = self.get_json_body() or {}
        persona_id = body.get("persona_id")
        if persona_id:
            persona = persona_manager.personas.get(persona_id)
            if not persona:
                raise tornado.web.HTTPError(404, f"Persona not found: {persona_id}")
            persona_manager.set_active_persona(persona)
        else:
            persona_manager.set_active_persona(None)
        self.finish({"status": "ok", "active_id": persona_id or None})


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