import os
import sys
from ..base_acp_persona import BaseAcpPersona
from jupyter_ai_persona_manager import PersonaDefaults


class TestAcpPersona(BaseAcpPersona):
    def __init__(self, *args, **kwargs):
        # Get absolute path to agent.py
        agent_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "examples", "agent.py")
        )
        executable = [sys.executable, agent_path]
        super().__init__(*args, executable=executable, **kwargs)

    @property
    def defaults(self) -> PersonaDefaults:
        avatar_path = str(
            os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "static", "test.svg")
            )
        )

        return PersonaDefaults(
            name="Test-ACP",
            description="A test ACP persona",
            avatar_path=avatar_path,
            system_prompt="unused",
        )
