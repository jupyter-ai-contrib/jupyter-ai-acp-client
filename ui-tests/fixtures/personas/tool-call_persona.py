"""
Fixture persona: an ACP persona whose agent requests permission for a file
edit. See ../agents/tool_call_agent.py.
"""

import os
import sys

import jupyter_ai_acp_client
from jupyter_ai_acp_client.base_acp_persona import BaseAcpPersona
from jupyter_ai_persona_manager import PersonaDefaults
from jupyterlab_chat.models import Message

_AGENTS_DIR = os.environ["JAI_TEST_AGENTS_DIR"]
_AGENT_SCRIPT = os.path.join(_AGENTS_DIR, "tool_call_agent.py")
_AVATAR_PATH = os.path.join(
    os.path.dirname(jupyter_ai_acp_client.__file__), "static", "goose.svg"
)


class ToolCallTestPersona(BaseAcpPersona):
    """Test-only ACP persona whose agent requests permission for a file edit."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, executable=[sys.executable, _AGENT_SCRIPT], **kwargs
        )

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Tool Call Agent",
            description="Test-only ACP persona that requests an edit permission.",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def process_message(self, message: Message) -> None:
        await super().process_message(message)
