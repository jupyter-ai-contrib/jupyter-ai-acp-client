from __future__ import annotations

from typing import TYPE_CHECKING

from jupyter_server.base.handlers import APIHandler
import tornado
from .base_acp_persona import BaseAcpPersona

if TYPE_CHECKING:
    from jupyter_server_fileid.manager import BaseFileIdManager
    from jupyter_ai_persona_manager import PersonaManager


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