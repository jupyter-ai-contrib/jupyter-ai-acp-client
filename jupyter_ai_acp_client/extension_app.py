from __future__ import annotations
from typing import TYPE_CHECKING
from jupyter_server.extension.application import ExtensionApp
from .routes import AcpSlashCommandsHandler


class JaiAcpClientExtension(ExtensionApp):
    """
    Jupyter AI ACP client extension.
    """

    name = "jupyter_ai_acp_client"
    handlers = [
        (r"ai/acp/slash_commands/?([^/]*)?", AcpSlashCommandsHandler),
    ]

    def initialize_settings(self):
        """Initialize router settings and event listeners."""
        return

    async def stop_extension(self):
        """Clean up router when extension stops."""
        return
