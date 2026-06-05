import asyncio
import os
import shutil

from jupyter_ai_persona_manager import PersonaDefaults, PersonaRequirementsUnmet
from jupyterlab_chat.models import Message
from ..base_acp_persona import BaseAcpPersona

# Raise `PersonaRequirementsUnmet` if `agy` not installed
if shutil.which("agy") is None:
    raise PersonaRequirementsUnmet(
        "This persona requires `agy` CLI to be installed."
        " See https://antigravity.google/download#antigravity-cli for installation instructions."
    )

# Raise `PersonaRequirementsUnmet` if `agy-acp` not installed.
# `agy` has no native --acp mode; `agy-acp` is a stdio ACP adapter from
# https://github.com/openabdev/openab that bridges ACP JSON-RPC to `agy -p` invocations.
if shutil.which("agy-acp") is None:
    raise PersonaRequirementsUnmet(
        "This persona requires `agy-acp` to be installed."
        " Build it from source: `cargo build --release` in the `agy-acp` directory"
        " of https://github.com/openabdev/openab, then place the binary on your PATH."
    )


class AntigravityAcpPersona(BaseAcpPersona):
    _terminal_opened: bool

    def __init__(self, *args, **kwargs):
        # agy has no native --acp mode; use the openab agy-acp adapter instead.
        executable = ["agy-acp"]
        super().__init__(*args, executable=executable, **kwargs)
        self._terminal_opened = False

    @property
    def defaults(self) -> PersonaDefaults:
        avatar_path = str(os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "static", "antigravity.svg")
        ))

        return PersonaDefaults(
            name="Antigravity",
            description="Antigravity in Jupyter AI!",
            avatar_path=avatar_path,
            system_prompt="unused"
        )

    async def before_agent_subprocess(self) -> None:
        # The Antigravity ACP agent subprocess fails to start if the user is not signed
        # in. Therefore we must implement this method to wait until the user is
        # signed in. The ACP agent server does not start until this is complete.
        failed_auth_check = False
        while True:
            # If authenticated with Antigravity, return
            if await self._check_antigravity_auth():
                break

            # Reaching here := user is not signed in
            if not failed_auth_check:
                self.log.info("[Antigravity] User is not signed in.")
                failed_auth_check = True

            # Re-check every 2 seconds
            await asyncio.sleep(2)

        # Reaching this point := user is authenticated
        self.log.info("[Antigravity] User is signed in.")

    async def is_authed(self) -> bool:
        # Check if the before_subprocess task is done (subprocess has started)
        if not self._before_subprocess_future.done():
            return False

        # In Antigravity, configuration can change at runtime (e.g., if settings.json
        # is deleted), so we need to verify that Antigravity is still properly
        # configured before processing each message. Use a fast file check.
        return await self._check_antigravity_auth_fast()

    async def handle_no_auth(self, message: Message) -> None:
        await super().handle_no_auth(message)

        # Return canned reply with setup instructions
        self.send_message("You're not configured to use Antigravity yet. Please run the following command in a terminal to complete the setup:\n\n```\nagy\n```")

        # Open the terminal to help the user with setup
        if not self._terminal_opened:
            self._terminal_opened = await self._open_antigravity_login_terminal()
            if self._terminal_opened:
                self.send_message("I've opened a new terminal to help with that.")

    async def _check_antigravity_auth_fast(self) -> bool:
        """
        Fast authentication check that verifies required files exist.
        Used on every message to detect if configuration was deleted.

        The Antigravity CLI stores OAuth credentials at ~/.gemini/oauth_creds.json
        (shared with Gemini CLI) and its own settings at ~/.gemini/antigravity-cli/settings.json.
        """
        oauth_creds = os.path.expanduser("~/.gemini/oauth_creds.json")
        settings = os.path.expanduser("~/.gemini/antigravity-cli/settings.json")
        return (
            os.path.exists(oauth_creds) and os.path.isfile(oauth_creds) and
            os.path.exists(settings) and os.path.isfile(settings)
        )

    async def _check_antigravity_auth(self) -> bool:
        """
        Authentication check that verifies Antigravity CLI is properly configured.
        Used during startup polling to wait for initial configuration.

        Uses the fast file-based check only, avoiding `agy --prompt` which
        triggers a full LLM inference call and delays subprocess startup.
        """
        return await self._check_antigravity_auth_fast()

    async def _open_antigravity_login_terminal(self) -> bool:
        """
        Attempt to open a terminal to log in with Antigravity.

        Returns `True` if successful, `False` otherwise.
        """
        try:
            from jupyterlab_commands_toolkit.tools import execute_command
        except Exception:
            return False

        response = await execute_command("terminal:create-new")
        return response.get("success", False)
