"""
Fixture persona for E2E tests: an ACP persona whose agent always replies "hello".

This file is not part of the shipped package. The server config copies it into
`<root>/.jupyter/personas/` when a suite requests the "hello" persona, and the
PersonaManager auto-loads any `*persona*.py` there. Because the loader only keeps
classes defined in this module (`obj.__module__ == module stem`), the persona
class must be declared here rather than imported.

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
_AGENT_SCRIPT = os.path.join(_AGENTS_DIR, "hello_agent.py")
# Reuse a shipped avatar so the fixture needs no image asset of its own.
_AVATAR_PATH = os.path.join(
    os.path.dirname(jupyter_ai_acp_client.__file__), "static", "goose.svg"
)


class HelloTestPersona(BaseAcpPersona):
    """Test-only ACP persona that always replies 'hello'."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, executable=[sys.executable, _AGENT_SCRIPT], **kwargs
        )

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Hello Test Agent",
            description="Test-only ACP persona that always replies 'hello'.",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def process_message(self, message: Message) -> None:
        await super().process_message(message)
