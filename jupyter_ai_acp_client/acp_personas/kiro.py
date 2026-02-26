import asyncio
import os
import re
import shutil
import subprocess

from jupyter_ai_persona_manager import PersonaDefaults, PersonaRequirementsUnmet
from jupyterlab_chat.models import Message
from ..base_acp_persona import BaseAcpPersona

# Raise `PersonaRequirementsUnmet` if `kiro-cli` not installed
if shutil.which("kiro-cli") is None:
    raise PersonaRequirementsUnmet(
        "This persona requires `kiro-cli` to be installed."
        " See https://kiro.dev for installation instructions."
    )

# Raise `PersonaRequirementsUnmet` if `kiro-cli<1.25.0`
try:
    result = subprocess.run(
        ["kiro-cli", "--version"],
        capture_output=True,
        text=True,
        timeout=5
    )

    # Check for non-zero exit code
    if result.returncode != 0:
        stderr = result.stderr.strip()
        error_msg = (
            f"kiro-cli --version returned non-zero exit code {result.returncode}."
            " Please ensure kiro-cli is properly installed."
        )
        if stderr:
            error_msg += f"\nStderr output: {stderr}"

        raise PersonaRequirementsUnmet(error_msg)

    # Extract semver from stdout using regex
    version_match = re.search(r'(\d+\.\d+\.\d+)', result.stdout)
    if not version_match:
        raise PersonaRequirementsUnmet(
            "Could not extract version number from kiro-cli --version output."
            f" Got: {result.stdout.strip()}"
        )

    version_str = version_match.group(1)
    version_parts = [int(x) for x in version_str.split('.')]

    # Check if version >= 1.25.0
    required_version = (1, 25, 0)
    current_version = tuple(version_parts)

    if current_version < required_version or current_version[0] >= 2:
        raise PersonaRequirementsUnmet(
            f"kiro-cli version {version_str} is installed, but version >=1.25.0,<2 is required."
            " Please upgrade kiro-cli. See https://kiro.dev for instructions."
        )

except subprocess.TimeoutExpired:
    raise PersonaRequirementsUnmet(
        "kiro-cli --version command timed out."
        " Please ensure kiro-cli is properly installed."
    )
except FileNotFoundError:
    # This shouldn't happen since we checked with shutil.which, but handle it anyway
    raise PersonaRequirementsUnmet(
        "kiro-cli command not found."
        " Please ensure kiro-cli is properly installed."
    )

class KiroAcpPersona(BaseAcpPersona):
    _terminal_opened: bool
    def __init__(self, *args, **kwargs):
        executable = ["kiro-cli", "acp"]
        super().__init__(*args, executable=executable, **kwargs)
        self._terminal_opened = False
    
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
    
    async def ensure_auth(self) -> None:
        self.log.info("[Kiro] User is not logged in.")
        while True:
            # If authenticated with Kiro, return
            if await self._check_kiro_auth():
                break

            # Otherwise, check every 2 seconds
            await asyncio.sleep(2)
    
    async def handle_no_auth(self, message: Message) -> None:
        # Return canned reply
        self.send_message("Please sign in via `kiro-cli login`.")

        # Open the terminal to help the user login
        if not self._terminal_opened:
            # TODO
            self._terminal_opened = True

    async def _check_kiro_auth(self) -> bool:
        """
        Helper method that checks if the client is authenticated with Kiro.
        """
        import asyncio
        process = await asyncio.create_subprocess_exec(
            "kiro-cli", "whoami",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        await process.wait()
        return process.returncode == 0
    