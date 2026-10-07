"""
Fixture persona: an ACP persona whose agent advertises its mode through *both*
channels at once — the dedicated `session/set_mode` state and a config option
with category `"mode"`. Used to check the client de-duplicates it into a single
mode control. See ../agents/mode_agent.py.

Not part of the shipped package. `installPersonas` copies it into a suite's
`<dir>/.jupyter/personas/`, and the PersonaManager auto-loads it there. The
loader only keeps classes defined in this module, so the class must be declared
here. The fake agent path comes from `JAI_TEST_AGENTS_DIR` (exported by the
server config).
"""

import os
import sys

import jupyter_ai_acp_client
from jupyter_ai_acp_client.base_acp_persona import BaseAcpPersona
from jupyter_ai_persona_manager import PersonaDefaults
from jupyterlab_chat.models import Message

_AGENTS_DIR = os.environ["JAI_TEST_AGENTS_DIR"]
_AGENT_SCRIPT = os.path.join(_AGENTS_DIR, "mode_agent.py")
_AVATAR_PATH = os.path.join(
    os.path.dirname(jupyter_ai_acp_client.__file__), "static", "goose.svg"
)


class BothModeTestPersona(BaseAcpPersona):
    """Test-only ACP persona whose agent advertises mode via both channels."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            executable=[sys.executable, _AGENT_SCRIPT, "--channel", "both"],
            **kwargs,
        )

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Both Mode Agent",
            description="Test-only ACP persona advertising mode via both channels.",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def process_message(self, message: Message) -> None:
        await super().process_message(message)
