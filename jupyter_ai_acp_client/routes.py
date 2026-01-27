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
