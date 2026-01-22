import shutil
from jupyter_ai_persona_manager import PersonaRequirementsUnmet
if shutil.which("claude-code-acp") is None:
    raise PersonaRequirementsUnmet(
        "This persona requires the Claude Code ACP adapter to be installed."
        " Install it via `npm install -g @zed-industries/claude-code-acp`"
        " then restart."
    )

import os
from ..base_acp_persona import BaseAcpPersona
from jupyter_ai_persona_manager import PersonaDefaults

class ClaudeAcpPersona(BaseAcpPersona):
    def __init__(self, *args, **kwargs):
        executable = ["claude-code-acp"]
        super().__init__(*args, executable=executable, **kwargs)
    
    @property
    def defaults(self) -> PersonaDefaults:
        avatar_path = str(os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "static", "claude.svg")
        ))

        return PersonaDefaults(
            name="Claude-ACP",
            description="Claude Code as an ACP agent persona.",
            avatar_path=avatar_path,
            system_prompt="unused"
        )