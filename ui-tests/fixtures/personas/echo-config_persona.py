"""
Fixture persona for E2E tests: an ACP persona whose agent echoes its current
session config options as YAML (see ../agents/echo_config_agent.py).

This file is not part of the shipped package. A suite that requests the
`echo-config` fixture installs it at runtime: its `beforeAll` calls
`installPersonas` (see `tests/test-helpers.ts`), which uploads this file to the
suite's `<dir>/.jupyter/personas/`, where the PersonaManager auto-loads any
`*persona*.py`. Because the loader only keeps classes defined in this module
(`obj.__module__ == module stem`), the persona class must be declared here
rather than imported.

The path to the fake agent script is read from `JAI_TEST_AGENTS_DIR`, which the
server config exports, so this file works regardless of where it is copied.
"""

import os
import sys

import jupyter_ai_acp_client
from jupyter_ai_acp_client.base_acp_persona import BaseAcpPersona
from jupyter_ai_persona_manager import PersonaDefaults
from jupyterlab_chat.models import Message

_AGENTS_DIR = os.environ["JAI_TEST_AGENTS_DIR"]
_AGENT_SCRIPT = os.path.join(_AGENTS_DIR, "echo_config_agent.py")
# Reuse a shipped avatar so the fixture needs no image asset of its own.
_AVATAR_PATH = os.path.join(
    os.path.dirname(jupyter_ai_acp_client.__file__), "static", "goose.svg"
)


class EchoConfigTestPersona(BaseAcpPersona):
    """Test-only ACP persona that echoes its config options as YAML."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, executable=[sys.executable, _AGENT_SCRIPT], **kwargs
        )

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Echo Config Agent",
            description="Test-only ACP persona that echoes its config options.",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def process_message(self, message: Message) -> None:
        await super().process_message(message)
