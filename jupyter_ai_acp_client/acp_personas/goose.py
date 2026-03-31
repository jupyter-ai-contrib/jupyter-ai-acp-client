import os
import re
import shutil
import subprocess

from acp.exceptions import RequestError
from jupyter_ai_persona_manager import PersonaDefaults, PersonaRequirementsUnmet
from jupyterlab_chat.models import Message

from ..base_acp_persona import BaseAcpPersona


def _is_setup_error(error: Exception) -> bool:
    """Check if error indicates Goose needs provider configuration.

    Source-verified against block/goose (server.rs):
    - Session creation errors: -32603 with data prefixed "Failed to set provider:"
      or "Failed to create session/agent:"
    - Framework errors: -32603 with data=None (sacp dispatch layer)
    - Goose never sends -32000, but we handle it for forward compatibility.
    - Prompt-time provider errors are streamed as text, not RequestError.
    """
    if not isinstance(error, RequestError):
        return False
    if error.code == -32000:
        return True
    if error.code != -32603:
        return False
    data = str(error.data or "").lower()
    if not data:
        return True  # framework error, likely during session init
    return "failed to set provider" in data or "failed to create" in data


def _check_goose():
    """Verify goose is installed and has ACP support (>= 1.8.0, < 2)."""
    if shutil.which("goose") is None:
        raise PersonaRequirementsUnmet(
            "This persona requires the Goose CLI."
            " See https://github.com/block/goose for installation instructions."
        )

    try:
        result = subprocess.run(
            ["goose", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            error_msg = (
                f"goose --version returned non-zero exit code {result.returncode}."
                " Please ensure goose is properly installed."
            )
            if stderr:
                error_msg += f"\nStderr output: {stderr}"
            raise PersonaRequirementsUnmet(error_msg)

        version_match = re.search(r"(\d+\.\d+\.\d+)", result.stdout)
        if not version_match:
            raise PersonaRequirementsUnmet(
                "Could not extract version number from goose --version output."
                f" Got: {result.stdout.strip()}"
            )

        version_str = version_match.group(1)
        version_parts = [int(x) for x in version_str.split(".")]
        current_version = tuple(version_parts)
        required_version = (1, 8, 0)

        if current_version < required_version or current_version[0] >= 2:
            raise PersonaRequirementsUnmet(
                f"Goose version {version_str} is installed,"
                " but version >=1.8.0,<2 is required."
                " See https://github.com/block/goose for instructions."
            )

    except subprocess.TimeoutExpired:
        raise PersonaRequirementsUnmet(
            "goose --version command timed out."
            " Please ensure goose is properly installed."
        )
    except FileNotFoundError:
        raise PersonaRequirementsUnmet(
            "goose command not found."
            " Please ensure goose is properly installed."
        )


_check_goose()


class GooseAcpPersona(BaseAcpPersona):
    def __init__(self, *args, **kwargs):
        executable = ["goose", "acp"]
        super().__init__(*args, executable=executable, **kwargs)

    @property
    def defaults(self) -> PersonaDefaults:
        avatar_path = str(
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "..", "static", "goose.svg"
                )
            )
        )

        return PersonaDefaults(
            name="Goose",
            description="Block's Goose as an ACP agent persona.",
            avatar_path=avatar_path,
            system_prompt="unused",
        )

    async def process_message(self, message: Message) -> None:
        try:
            await super().process_message(message)
        except RequestError as error:
            if not _is_setup_error(error):
                raise

            self.log.info(
                "[Goose] Setup error (code=%s): %s (data=%s)",
                error.code,
                str(error),
                error.data,
            )
            await self.handle_no_auth(message)

    async def handle_no_auth(self, message: Message) -> None:
        self.send_message(
            "Goose isn't configured yet."
            "\n\n- Run `goose configure` in a terminal to set up a provider."
            "\n\nRestart the JupyterLab server after configuration."
        )
