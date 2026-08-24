import os
import re
import shutil
import subprocess

from jupyter_ai_persona_manager import PersonaDefaults
from jupyterlab_chat.models import Message
from ..base_acp_persona import BaseAcpPersona
from acp.exceptions import RequestError


class KiloAcpPersona(BaseAcpPersona):
    def __init__(self, *args, **kwargs):
        executable = ["kilo", "acp"]
        super().__init__(*args, executable=executable, **kwargs)

    def check_requirements(self) -> str | None:
        """Check that kilo is installed and meets version requirements."""
        if shutil.which("kilo") is None:
            return (
                "This persona requires `kilo` to be installed."
                " See https://kilo.ai/docs/getting-started for installation instructions."
            )

        try:
            result = subprocess.run(
                ["kilo", "acp", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                stderr = result.stderr.strip()
                error_msg = (
                    f"kilo --version returned non-zero exit code {result.returncode}."
                    " Please ensure kilo is properly installed."
                )
                if stderr:
                    error_msg += f"\nStderr output: {stderr}"
                return error_msg

            version_match = re.search(r'(\d+\.\d+\.\d+)', result.stdout)
            if not version_match:
                return (
                    "Could not extract version number from kilo --version output."
                    f" Got: {result.stdout.strip()}"
                )

            version_str = version_match.group(1)
            version_parts = [int(x) for x in version_str.split('.')]
            required_version = (7, 0, 0)
            current_version = tuple(version_parts)

            if current_version < required_version:
                return (
                    f"kilo version {version_str} is installed, but version >=7.0.0 is required."
                    " Please upgrade kilo. See https://kilo.ai/ for instructions."
                )

        except subprocess.TimeoutExpired:
            return (
                "kilo acp --version command timed out."
                " Please ensure kilo is properly installed."
            )
        except FileNotFoundError:
            return (
                "kilo command not found."
                " Please ensure kilo is properly installed."
            )

        return None

    @property
    def defaults(self) -> PersonaDefaults:
        avatar_path = str(os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "static", "kilo.svg")
        ))

        return PersonaDefaults(
            name="Kilo",
            description="Kilo in Jupyter AI!",
            avatar_path=avatar_path,
            system_prompt="unused"
        )
    
    async def before_agent_subprocess(self) -> None:
        # The Kilo ACP agent server will start successfully even if it is not configured properly 
        return None

    async def is_authed(self) -> bool:
        # Be optimistic and assume the user is authenticated. 
        # If they are not, the `process_message()` method will raise an error which we can catch to 
        # trigger the auth flow.
        return True
