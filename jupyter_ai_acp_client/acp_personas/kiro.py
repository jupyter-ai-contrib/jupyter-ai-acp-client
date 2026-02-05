import shutil
from jupyter_ai_persona_manager import PersonaRequirementsUnmet
if shutil.which("kiro-cli") is None:
    raise PersonaRequirementsUnmet(
        "This persona requires `kiro-cli` to be installed."
        " See https://kiro.dev for installation instructions."
    )

import os
from ..base_acp_persona import BaseAcpPersona
from jupyter_ai_persona_manager import PersonaDefaults

class KiroAcpPersona(BaseAcpPersona):
    def __init__(self, *args, **kwargs):
        executable = ["kiro-cli", "acp"]
        super().__init__(*args, executable=executable, **kwargs)
    
    @property
    def defaults(self) -> PersonaDefaults:
        avatar_path = str(os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "static", "kiro.svg")
        ))

        return PersonaDefaults(
            name="Kiro",
            description="Kiro in Jupyter AI!",
            avatar_path=avatar_path,
            system_prompt="unused"
        )