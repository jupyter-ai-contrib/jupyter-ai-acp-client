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

        if persona_mention_name:
            for p in persona_manager.personas.values():
                if p.as_user().mention_name == persona_mention_name:
                    persona = p
                    break
            if not persona:
                # raise HTTP error: persona not found
                raise tornado.web.HTTPError(404, f"Persona not found: @{persona_mention_name}")
        else:
            persona = persona_manager.last_mentioned_persona or persona_manager.default_persona

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


class PermissionHandler(APIHandler):
    """
    REST endpoint for permission decisions. The frontend POSTs the user's
    button click here, and this handler finds the right JaiAcpClient and
    resolves the pending Future so the suspended request_permission coroutine
    can resume.
    """

    def _find_client_for_session(self, session_id: str):
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
        logger.info(f"_find_client_for_session: looking for session_id={session_id}, "
                     f"persona_managers count={len(persona_managers)}")
        for room_id, pm in persona_managers.items():
            logger.info(f"  checking room={room_id}, personas={list(pm.personas.keys())}")
            for persona_id, persona in pm.personas.items():
                if not isinstance(persona, BaseAcpPersona):
                    logger.info(f"    {persona_id}: not BaseAcpPersona, skipping")
                    continue
                client_future = persona.__class__._client_future
                if client_future is None:
                    logger.info(f"    {persona_id}: _client_future is None")
                    continue
                if not client_future.done():
                    logger.info(f"    {persona_id}: _client_future not done yet")
                    continue
                client = client_future.result()
                known_sessions = list(client._personas_by_session.keys())
                pending_keys = list(client._pending_permissions.keys())
                logger.info(f"    {persona_id}: known_sessions={known_sessions}, "
                            f"pending_permissions={pending_keys}")
                if session_id in client._personas_by_session:
                    logger.info(f"    FOUND client for session {session_id}")
                    return client
        logger.info(f"_find_client_for_session: no client found for session_id={session_id}")
        return None

    @tornado.web.authenticated
    async def post(self):
        body = self.get_json_body()
        session_id = body.get("session_id")
        tool_call_id = body.get("tool_call_id")
        option_id = body.get("option_id")

        if not all([session_id, tool_call_id, option_id]):
            raise tornado.web.HTTPError(400, "Missing required fields: session_id, tool_call_id, option_id")

        client = self._find_client_for_session(session_id)
        if not client:
            raise tornado.web.HTTPError(404, "No pending permission request found for this session")

        resolved = client.resolve_permission(session_id, tool_call_id, option_id)
        if not resolved:
            raise tornado.web.HTTPError(404, "No pending permission request found")

        self.finish({"status": "ok"})