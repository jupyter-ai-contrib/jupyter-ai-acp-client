import os
import re
import shutil
import subprocess
from asyncio.subprocess import Process
from pathlib import Path

from acp.exceptions import RequestError
from jupyter_ai_persona_manager import PersonaDefaults, PersonaRequirementsUnmet
from jupyterlab_chat.models import Message

from ..base_acp_persona import BaseAcpPersona

# Path to the bundled opencode.json that ships with this package.
# Configures permission: {edit: "ask", bash: "ask"} so OpenCode requests
# approval before file edits and shell commands.
_BUNDLED_CONFIG = os.path.join(os.path.dirname(__file__), "opencode.json")


def _has_user_config() -> bool:
    """Check if user has a global OpenCode config file."""
    config_dir = Path.home() / ".config" / "opencode"
    return (config_dir / "opencode.json").exists() or (config_dir / "opencode.jsonc").exists()


def _is_auth_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        keyword in message
        for keyword in (
            "api key",
            "api_key",
            "authentication",
            "authorized",
            "credential",
            "forbidden",
            "not configured",
        )
    )


def _check_opencode() -> None:
    """Raise PersonaRequirementsUnmet if opencode is missing or wrong version."""
    if shutil.which("opencode") is None:
        raise PersonaRequirementsUnmet(
            "This persona requires `opencode` to be installed."
            " See https://opencode.ai for installation instructions."
        )

    try:
        result = subprocess.run(
            ["opencode", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        raise PersonaRequirementsUnmet(
            "opencode --version command timed out."
            " Please ensure opencode is properly installed."
        )
    except FileNotFoundError:
        raise PersonaRequirementsUnmet(
            "opencode command not found."
            " Please ensure opencode is properly installed."
        )

    if result.returncode != 0:
        stderr = result.stderr.strip()
        error_msg = (
            f"opencode --version returned non-zero exit code {result.returncode}."
            " Please ensure opencode is properly installed."
        )
        if stderr:
            error_msg += f"\nStderr output: {stderr}"
        raise PersonaRequirementsUnmet(error_msg)

    version_match = re.search(r"(\d+\.\d+\.\d+)", result.stdout)
    if not version_match:
        raise PersonaRequirementsUnmet(
            "Could not extract version number from opencode --version output."
            f" Got: {result.stdout.strip()}"
        )

    version_str = version_match.group(1)
    current_version = tuple(int(x) for x in version_str.split("."))

    if current_version < (1, 0, 0) or current_version[0] >= 2:
        raise PersonaRequirementsUnmet(
            f"opencode version {version_str} is installed,"
            " but version >=1.0.0,<2 is required."
            " Please upgrade opencode. See https://opencode.ai for instructions."
        )


_check_opencode()


class OpenCodeAcpPersona(BaseAcpPersona):
    def __init__(self, *args, **kwargs):
        executable = ["opencode", "acp"]
        super().__init__(*args, executable=executable, **kwargs)

    @property
    def defaults(self) -> PersonaDefaults:
        avatar_path = str(
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "..", "static", "opencode.svg"
                )
            )
        )

        return PersonaDefaults(
            name="OpenCode",
            description="OpenCode as an ACP agent persona.",
            avatar_path=avatar_path,
            system_prompt="unused",
        )

    async def _init_agent_subprocess(self) -> Process:
        env: dict[str, str] | None = None

        # Only inject bundled config if the user hasn't configured OpenCode themselves.
        # Precedence: OPENCODE_CONFIG env var > ~/.config/opencode/opencode.{json,jsonc} > bundled
        if "OPENCODE_CONFIG" not in os.environ and not _has_user_config():
            env = os.environ.copy()
            env["OPENCODE_CONFIG"] = _BUNDLED_CONFIG

        return await super()._init_agent_subprocess(env=env)

    async def is_authed(self) -> bool:
        return True

    async def process_message(self, message: Message) -> None:
        try:
            await super().process_message(message)
        except RequestError as error:
            if not _is_auth_error(error):
                raise

            self.log.info(
                "[OpenCode] Authentication or configuration required: %s",
                error,
            )
            await self.handle_no_auth(message)

    async def handle_no_auth(self, message: Message) -> None:
        self.send_message(
            "OpenCode isn't configured yet."
            "\n\n- Run `opencode auth` in a terminal to configure your LLM provider."
            "\n\n- Or set your provider's API key as an environment variable"
            " (e.g. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) before starting JupyterLab."
            "\n\n- If you set the environment variable in a new shell,"
            " restart the JupyterLab server so this process can see it."
        )
