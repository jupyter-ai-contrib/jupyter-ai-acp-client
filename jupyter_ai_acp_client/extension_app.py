from __future__ import annotations
from typing import TYPE_CHECKING
from jupyter_server.extension.application import ExtensionApp
from .routes import PermissionHandler
from .telemetry import register_telemetry_schemas, SCHEMA_ID, emit_event


class JaiAcpClientExtension(ExtensionApp):
    """
    Jupyter AI ACP client extension.
    """

    name = "jupyter_ai_acp_client"
    handlers = [
        (r"ai/acp/permissions", PermissionHandler),
    ]

    def initialize_settings(self):
        """Initialize router settings and register ACP event schema."""
        

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

            # Defer emit until all extensions are initialized.
            self.serverapp.io_loop.add_callback(self._emit_agents_status, event_logger)
        except Exception:
            self.log.error(
                "Failed to register ACP event schema or listener with EventLogger.",
                exc_info=True,
            )

    def _check_installed_agents(self) -> tuple[list[str], list[str]]:
        """Import all vendored ACP personas and check their requirements.

        Returns a tuple of (installed, not_installed) persona name lists.
        """
        persona_classes = []
        try:
            from .acp_personas.claude import ClaudeAcpPersona
            persona_classes.append(ClaudeAcpPersona)
        except Exception:
            pass
        try:
            from .acp_personas.kiro import KiroAcpPersona
            persona_classes.append(KiroAcpPersona)
        except Exception:
            pass
        try:
            from .acp_personas.codex import CodexAcpPersona
            persona_classes.append(CodexAcpPersona)
        except Exception:
            pass
        try:
            from .acp_personas.goose import GooseAcpPersona
            persona_classes.append(GooseAcpPersona)
        except Exception:
            pass
        try:
            from .acp_personas.copilot import CopilotAcpPersona
            persona_classes.append(CopilotAcpPersona)
        except Exception:
            pass
        try:
            from .acp_personas.mistral_vibe import MistralVibeAcpPersona
            persona_classes.append(MistralVibeAcpPersona)
        except Exception:
            pass
        try:
            from .acp_personas.opencode import OpenCodeAcpPersona
            persona_classes.append(OpenCodeAcpPersona)
        except Exception:
            pass
        try:
            from .acp_personas.kilo import KiloAcpPersona
            persona_classes.append(KiloAcpPersona)
        except Exception:
            pass

        installed = []
        not_installed = []

        for cls in persona_classes:
            try:
                result = cls.check_requirements(None)
                name = cls.__name__.replace("AcpPersona", "")
                if result is None:
                    installed.append(name)
                else:
                    not_installed.append(name)
            except Exception:
                name = cls.__name__.replace("AcpPersona", "")
                not_installed.append(name)

        return installed, not_installed

    def _emit_agents_status(self, event_logger) -> None:
        """Emit a single acp_agents_status telemetry event at startup."""
        from .telemetry import emit_event

        installed, not_installed = self._check_installed_agents()

        emit_event(
            event_logger,
            "acp_agents_status",
            "success",
            {
                "installed": ", ".join(installed) if installed else "",
                "not_installed": ", ".join(not_installed) if not_installed else "",
                "total_registered": str(len(installed) + len(not_installed)),
            },
        )

        self.log.info(
            "ACP agents status: %d installed (%s), %d not installed (%s)",
            len(installed),
            ", ".join(installed) or "none",
            len(not_installed),
            ", ".join(not_installed) or "none",
        )

    async def stop_extension(self):
        """Clean up router when extension stops."""
        return
