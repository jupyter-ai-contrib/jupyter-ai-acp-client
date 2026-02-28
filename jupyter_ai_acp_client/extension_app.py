from __future__ import annotations
from typing import TYPE_CHECKING
from jupyter_server.extension.application import ExtensionApp
from .routes import AcpSlashCommandsHandler, PermissionHandler


class JaiAcpClientExtension(ExtensionApp):
    """
    Jupyter AI ACP client extension.
    """

    name = "jupyter_ai_acp_client"
    handlers = [
        (r"ai/acp/slash_commands/?([^/]*)?", AcpSlashCommandsHandler),
        (r"ai/acp/permissions", PermissionHandler),
    ]

    def initialize_settings(self):
        """Initialize router settings and event listeners."""
        # # Ensure 'jupyter-ai' dictionary is in `self.settings`, which gets
        # # copied to `self.serverapp.web_app.settings` after this method returns
        # if 'jupyter-ai' not in self.settings:
        #     self.settings['jupyter-ai'] = {}
        
        # self.settings['jupyter-ai']['acp-client']
        return

    async def stop_extension(self):
        """Clean up router when extension stops."""
        return
