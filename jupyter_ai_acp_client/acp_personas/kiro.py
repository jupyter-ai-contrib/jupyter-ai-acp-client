import asyncio
import os
import platform
import re
import shutil
import subprocess
from typing import ClassVar, Optional

from jupyter_ai_persona_manager import (
    ModelOption,
    PersonaDefaults,
    PersonaRequirementsUnmet,
)
from jupyterlab_chat.models import Message
from ..base_acp_persona import BaseAcpPersona
from ..default_acp_client import JaiAcpClient
from ..kiro_client import KiroAcpClient, KiroModels

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

    if current_version < required_version or current_version[0] >= 3:
        raise PersonaRequirementsUnmet(
            f"kiro-cli version {version_str} is installed, but version >=1.25.0,<3 is required."
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

    # kiro-cli's ACP surface is non-standard (legacy `models` field, deprecated
    # `session/set_model`, vendor usage/command notifications), so this persona
    # uses a Kiro-scoped client that handles all of it. See `kiro_client.py`.
    acp_client_class: ClassVar[type[JaiAcpClient]] = KiroAcpClient

    def __init__(self, *args, **kwargs):
        executable = ["kiro-cli", "acp"]
        super().__init__(*args, executable=executable, **kwargs)
        self._terminal_opened = False
        # The legacy `models` payload the client captures off the raw session
        # response (`None` until a session is created/loaded). Kiro advertises
        # models this way instead of through ACP v1 config options.
        self._kiro_models: Optional[KiroModels] = None

    def set_kiro_models(self, models: Optional[KiroModels]) -> None:
        """
        Store the legacy `models` payload the `KiroAcpClient` captured off the
        raw session response. Called during session create/load, before the
        base persona syncs its awareness config, so the models are present when
        `_build_awareness_config` runs.
        """
        self._kiro_models = models

    def _build_awareness_config(self):
        """
        Build the awareness config as the base does, then fill the model picker
        from Kiro's legacy `models` payload when no ACP config option advertised
        one (Kiro's normal case). A genuine `"model"`-category config option, if
        the agent ever sends one, still wins.
        """
        model, general_settings = super()._build_awareness_config()
        if not model.options and self._kiro_models:
            model.current = self._kiro_models.current_model_id
            model.options = [
                ModelOption(
                    id=option.model_id,
                    name=option.name or option.model_id,
                    description=option.description,
                )
                for option in (self._kiro_models.available_models or [])
                if option.model_id
            ]
        return model, general_settings

    async def update_model(self, model_id: str) -> None:
        """
        Switch the model. When models come from Kiro's legacy payload (no ACP
        model config option), apply the choice via the deprecated
        `session/set_model` request; otherwise defer to the standard
        config-option path. The legacy choice is kept agent-side on the session
        (resumed by ID), not persisted with the chat.
        """
        if self._model_config_option() is None and self._kiro_models:
            client = await self.get_client()
            session_id = await self.get_session_id()
            await client.set_session_model(model_id, session_id)
            self._kiro_models.current_model_id = model_id
            return
        await super().update_model(model_id)

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
    
    async def before_agent_subprocess(self) -> None:
        # The Kiro ACP agent subprocess fails to start if the user is not signed
        # in. Therefore we must implement this method to wait until the user is
        # signed in. The ACP agent server does not start until this is complete.
        failed_auth_check = False
        while True:
            # If authenticated with Kiro, return
            if await self._check_kiro_auth():
                break

            # Reaching here := user is not signed in
            if not failed_auth_check:
                self.log.info("[Kiro] User is not signed in.")
                failed_auth_check = True

            # Re-check every 2 seconds
            await asyncio.sleep(2)
        
        # Reaching this point := user is authenticated
        self.log.info("[Kiro] User is signed in.")
    
    async def is_authed(self) -> bool:
        # In Kiro, the user remains signed in even if they sign out while the
        # ACP agent server is running. Therefore we can just return the status
        # of the `before_agent_subprocess()` task to check if the user is
        # authenticated.
        return self._before_subprocess_future.done()
    
    async def handle_no_auth(self, message: Message) -> None:
        await super().handle_no_auth(message)

        # Determine which command to show
        use_device_flow = await self._should_use_device_flow()
        command = "kiro-cli login --use-device-flow" if use_device_flow else "kiro-cli login"
        
        # Return canned reply with appropriate command
        self.send_message(f"You're not signed in to Kiro yet. Please run the following command in a terminal to sign in:\n\n```\n{command}\n```")

        # Open the terminal to help the user login
        if not self._terminal_opened:
            self._terminal_opened = await self._open_kiro_login_terminal()
            if self._terminal_opened:
                self.send_message("I've opened a new terminal to help with that.")

    async def _check_kiro_auth(self) -> bool:
        """
        Helper method that checks if the client is authenticated with Kiro.
        """
        process = await asyncio.create_subprocess_exec(
            "kiro-cli", "whoami",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        await process.wait()
        return process.returncode == 0
    
    async def _should_use_device_flow(self) -> bool:
        """Check if device flow should be used on Linux."""
        try:
            if platform.system() != 'Linux':
                return False
            
            # Check SSH
            if any(var in os.environ for var in ['SSH_CLIENT', 'SSH_CONNECTION', 'SSH_TTY']):
                return True
            
            # Check WSL
            try:
                with open('/proc/sys/kernel/osrelease', 'r') as f:
                    if any(x in f.read().lower() for x in ['microsoft', 'wsl']):
                        return not shutil.which('wslview')
            except:
                pass
            
            # Check xdg-open
            return not shutil.which('xdg-open')
        except Exception as e:
            self.log.warning(f"[Kiro] Error detecting device flow requirement: {e}")
            return False
    
    async def _open_kiro_login_terminal(self) -> bool:
        """
        Attempt to open a terminal to log in with Kiro.

        Returns `True` if successful, `False` otherwise.
        """
        try:
            from jupyterlab_commands_toolkit.tools import execute_command
        except Exception:
            return False

        response = await execute_command("terminal:create-new")
        return response.get("success", False)
