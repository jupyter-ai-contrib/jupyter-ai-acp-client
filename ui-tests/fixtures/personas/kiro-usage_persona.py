"""
Fixture persona: an ACP persona whose agent reports context usage only as a
bare percentage in a `_kiro.dev/metadata` extension notification, the way
kiro-cli's v2 engine does, with nothing on the standard usage channels. See
../agents/usage_agent.py.

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
from jupyter_ai_acp_client.kiro_client import KiroAcpClient
from jupyter_ai_persona_manager import PersonaDefaults
from jupyterlab_chat.models import Message

_AGENTS_DIR = os.environ["JAI_TEST_AGENTS_DIR"]
_AGENT_SCRIPT = os.path.join(_AGENTS_DIR, "usage_agent.py")
_AVATAR_PATH = os.path.join(
    os.path.dirname(jupyter_ai_acp_client.__file__), "static", "goose.svg"
)


class KiroUsageTestPersona(BaseAcpPersona):
    """Test-only ACP persona reporting a kiro-style context percentage only.

    Uses `KiroAcpClient` so the `_kiro.dev/metadata` vendor notification is
    mapped onto awareness the same way the real Kiro persona does (the generic
    client no longer handles it)."""

    acp_client_class = KiroAcpClient

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            executable=[sys.executable, _AGENT_SCRIPT, "--mode", "kiro"],
            **kwargs,
        )

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Kiro Usage Agent",
            description="Test-only ACP persona reporting a context percentage only.",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def process_message(self, message: Message) -> None:
        await super().process_message(message)
