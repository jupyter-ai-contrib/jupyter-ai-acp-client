from __future__ import annotations
from typing import TYPE_CHECKING
from jupyter_server.extension.application import ExtensionApp
from .routes import AcpSlashCommandsHandler, PermissionHandler, StopStreamingHandler


class JaiAcpClientExtension(ExtensionApp):
    """
    Jupyter AI ACP client extension.
    """

    name = "jupyter_ai_acp_client"
    handlers = [
        (r"ai/acp/slash_commands/?([^/]*)?", AcpSlashCommandsHandler),
        (r"ai/acp/permissions", PermissionHandler),
        (r"ai/acp/stop/?([^/]*)?", StopStreamingHandler),
    ]

    def initialize_settings(self):
        """Initialize router settings and register ACP event schema."""
        from .telemetry import register_telemetry_schemas, SCHEMA_ID

        try:
            event_logger = self.serverapp.event_logger
            register_telemetry_schemas(event_logger)

            ext_log = self.log

            async def _log_event(logger, schema_id, data):
                ext_log.debug("[ACP event] schema=%s data=%s", schema_id, data)

            event_logger.add_listener(
                schema_id=SCHEMA_ID,
                listener=_log_event,
            )
            self.log.info("Registered ACP event schema and listener with EventLogger.")
        except Exception:
            self.log.error(
                "Failed to register ACP event schema or listener with EventLogger.",
                exc_info=True,
            )

    async def stop_extension(self):
        """Clean up router when extension stops."""
        return
