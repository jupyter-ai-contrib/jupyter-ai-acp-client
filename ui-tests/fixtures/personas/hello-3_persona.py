"""
Fixture persona for E2E tests: a third ACP persona backed by the same
`hello_agent.py`, but replying with a distinct word ("hola") so a test can tell
which persona answered. Used by persona-control.spec.ts to verify that switching
the picker routes each message to the selected persona.

Not part of the shipped package. A suite that requests the `hello-3` fixture
installs it at runtime: its `beforeAll` calls `installPersonas` (see
`tests/test-helpers.ts`), which uploads this file to the suite's
`<dir>/.jupyter/personas/`, where the PersonaManager auto-loads any
`*persona*.py`. Because the loader only keeps classes defined in this module
(`obj.__module__ == module stem`), the persona class must be declared here
rather than imported.
"""

import os
import sys

import jupyter_ai_acp_client
from jupyter_ai_acp_client.base_acp_persona import BaseAcpPersona
from jupyter_ai_persona_manager import PersonaDefaults
from jupyterlab_chat.models import Message

_AGENTS_DIR = os.environ["JAI_TEST_AGENTS_DIR"]
_AGENT_SCRIPT = os.path.join(_AGENTS_DIR, "hello_agent.py")
_AVATAR_PATH = os.path.join(
    os.path.dirname(jupyter_ai_acp_client.__file__), "static", "goose.svg"
)


class Hello3TestPersona(BaseAcpPersona):
    """Test-only ACP persona that always replies 'hola'."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            executable=[sys.executable, _AGENT_SCRIPT, "--reply", "hola"],
            **kwargs,
        )

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Hello Agent Three",
            description="Test-only ACP persona that always replies 'hola'.",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def process_message(self, message: Message) -> None:
        await super().process_message(message)
