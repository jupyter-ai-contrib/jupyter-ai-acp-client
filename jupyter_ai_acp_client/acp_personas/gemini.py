import asyncio
import os
import re
import shutil
import subprocess

from jupyter_ai_persona_manager import PersonaDefaults, PersonaRequirementsUnmet
from jupyterlab_chat.models import Message
from ..base_acp_persona import BaseAcpPersona

# Raise `PersonaRequirementsUnmet` if `gemini` not installed
if shutil.which("gemini") is None:
    raise PersonaRequirementsUnmet(
        "This persona requires `gemini` CLI to be installed."
        " See https://geminicli.com/ for installation instructions."
    )

# Raise `PersonaRequirementsUnmet` if `gemini` version check fails
try:
    result = subprocess.run(
        ["gemini", "--version"],
        capture_output=True,
        text=True,
        timeout=5
    )

    # Check for non-zero exit code
    if result.returncode != 0:
        stderr = result.stderr.strip()
        error_msg = (
            f"gemini --version returned non-zero exit code {result.returncode}."
            " Please ensure gemini CLI is properly installed."
        )
        if stderr:
            error_msg += f"\nStderr output: {stderr}"

        raise PersonaRequirementsUnmet(error_msg)

    # Extract semver from stdout using regex
    version_match = re.search(r'(\d+\.\d+\.\d+)', result.stdout)
    if not version_match:
        raise PersonaRequirementsUnmet(
            "Could not extract version number from gemini --version output."
            f" Got: {result.stdout.strip()}"
        )

    version_str = version_match.group(1)
    version_parts = [int(x) for x in version_str.split('.')]

    # Check if version >= 0.34.0
    required_version = (0, 34, 0)
    current_version = tuple(version_parts)

    if current_version < required_version:
        required_version_str = ".".join(map(str, required_version))
        raise PersonaRequirementsUnmet(
            f"gemini CLI version {version_str} is installed, but version >={required_version_str} is required."
            " Please upgrade gemini CLI. See https://ai.google.dev for instructions."
        )

except subprocess.TimeoutExpired:
    raise PersonaRequirementsUnmet(
        "gemini --version command timed out."
        " Please ensure gemini CLI is properly installed."
    )
except FileNotFoundError:
    # This shouldn't happen since we checked with shutil.which, but handle it anyway
    raise PersonaRequirementsUnmet(
        "gemini command not found."
        " Please ensure gemini CLI is properly installed."
    )

class GeminiAcpPersona(BaseAcpPersona):
    _terminal_opened: bool
    def __init__(self, *args, **kwargs):
        executable = ["gemini", "--acp"] # For a specific model, use additional entries "-m", "<model_id>"
        super().__init__(*args, executable=executable, **kwargs)
        self._terminal_opened = False

    @property
    def defaults(self) -> PersonaDefaults:
        avatar_path = str(os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "static", "gemini.svg")
        ))

        return PersonaDefaults(
            name="Gemini",
            description="Gemini in Jupyter AI!",
            avatar_path=avatar_path,
            system_prompt="unused"
        )

    async def before_agent_subprocess(self) -> None:
        # The Gemini ACP agent subprocess fails to start if the user is not signed
        # in. Therefore we must implement this method to wait until the user is
        # signed in. The ACP agent server does not start until this is complete.
        failed_auth_check = False
        while True:
            # If authenticated with Gemini, return
            if await self._check_gemini_auth():
                break

            # Reaching here := user is not signed in
            if not failed_auth_check:
                self.log.info("[Gemini] User is not signed in.")
                failed_auth_check = True

            # Re-check every 2 seconds
            await asyncio.sleep(2)

        # Reaching this point := user is authenticated
        self.log.info("[Gemini] User is signed in.")

        # If initially signed out, send a message letting the user know they are
        # now signed in.
        if failed_auth_check:
            self.send_message("Thanks for signing in! I'm ready to help.")

    async def is_authed(self) -> bool:
        # Check if the before_subprocess task is done (subprocess has started)
        if not self._before_subprocess_future.done():
            return False

        # In Gemini, configuration can change at runtime (e.g., if settings.json
        # is deleted), so we need to verify that Gemini is still properly
        # configured before processing each message. Use a fast file check.
        return await self._check_gemini_auth_fast()

    async def handle_no_auth(self, message: Message) -> None:
        # Return canned reply with setup instructions
        self.send_message("You're not configured to use Gemini yet. Please run `gemini` in a terminal to complete the setup.")

        # Open the terminal to help the user with setup
        if not self._terminal_opened:
            self._terminal_opened = await self._open_gemini_login_terminal()
            if self._terminal_opened:
                self.send_message("I've opened a new terminal to help with that.")

    async def _check_gemini_auth_fast(self) -> bool:
        """
        Fast authentication check that verifies required files exist.
        Used on every message to detect if configuration was deleted.
        """
        oauth_creds = os.path.expanduser("~/.gemini/oauth_creds.json")
        settings = os.path.expanduser("~/.gemini/settings.json")
        return (
            os.path.exists(oauth_creds) and os.path.isfile(oauth_creds) and
            os.path.exists(settings) and os.path.isfile(settings)
        )

    async def _check_gemini_auth(self) -> bool:
        """
        Authentication check that verifies Gemini CLI is properly configured.
        Used during startup polling to wait for initial configuration.

        Uses the fast file-based check only, avoiding `gemini --prompt` which
        triggers a full LLM inference call and delays subprocess startup.
        """
        return await self._check_gemini_auth_fast()

    async def _open_gemini_login_terminal(self) -> bool:
        """
        Attempt to open a terminal to log in with Gemini.

        Returns `True` if successful, `False` otherwise.
        """
        try:
            from jupyterlab_commands_toolkit.tools import execute_command
        except Exception:
            return False

        response = await execute_command("terminal:create-new")
        return response.get("success", False)
