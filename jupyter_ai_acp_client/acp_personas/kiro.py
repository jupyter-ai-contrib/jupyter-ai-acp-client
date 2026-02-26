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
        self.log.info("[Kiro] User is not signed in.")
        while True:
            # If authenticated with Kiro, return
            if await self._check_kiro_auth():
                break

            # Otherwise, check every 2 seconds
            await asyncio.sleep(2)
        
        # Reaching this point := user is authenticated
        # Send a message letting them know
        self.log.info("[Kiro] User is signed in.")
        self.send_message("Thanks for signing in! I'm ready to help.")
    
    async def handle_no_auth(self, message: Message) -> None:
        # Return canned reply
        self.send_message("You're not signed in to Kiro yet. Please run `kiro-cli login` in a terminal to sign in.")

        # Open the terminal to help the user login
        if not self._terminal_opened:
            self._terminal_opened = await self._open_kiro_login_terminal()
            if self._terminal_opened:
                self.send_message("I've opened a new terminal to help with that.")

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
    
    async def _open_kiro_login_terminal(self) -> bool:
        """
        Attempt to open a terminal to log in with Kiro.

        Returns `True` if successful, `False` otherwise.
        """
        try:
            from jupyterlab_commands_toolkit.tools import execute_command
        except:
            return False

        response = await execute_command("terminal:create-new")
        return response.get("success", False)