import os
import re
import shutil
import subprocess
from asyncio.subprocess import Process
from pathlib import Path
from typing import Optional

from acp.exceptions import RequestError
from jupyter_ai_persona_manager import PersonaDefaults, PersonaRequirementsUnmet
from jupyterlab_chat.models import Message

from ..base_acp_persona import BaseAcpPersona


def _is_setup_error(error: Exception) -> bool:
    """Check if error indicates Goose needs provider configuration.

    Goose wraps all session-init errors as -32603 (InternalError) with
    descriptive data prefixes: "Failed to set provider:", "Failed to create
    session/agent:", or "Authentication error:". Framework-level errors
    from the sacp layer arrive as -32603 with data=None. Goose never sends
    -32000 today, but we handle it for forward compatibility. Prompt-time
    provider errors are streamed as text, not RequestError.
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
    return (
        "failed to set provider" in data
        or "failed to create" in data
        or "authentication" in data
    )


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


def _get_user_config_path() -> Path:
    """Return the Goose config path for the current platform."""
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "Block" / "goose" / "config" / "config.yaml"
    config_home = os.environ.get("XDG_CONFIG_HOME")
    if config_home:
        return Path(config_home) / "goose" / "config.yaml"
    return Path.home() / ".config" / "goose" / "config.yaml"


def _parse_goose_mode(config_text: str) -> Optional[str]:
    """Extract an explicit GOOSE_MODE value from Goose config text."""
    pattern = re.compile(
        r'^\s*GOOSE_MODE\s*:\s*(?:"([^"]+)"|\'([^\']+)\'|([^#\s][^#]*?))\s*(?:#.*)?$'
    )
    for line in config_text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        value = next((group for group in match.groups() if group is not None), "").strip()
        return value or None
    return None


def _get_explicit_user_mode() -> Optional[str]:
    """Return the explicit GOOSE_MODE from config.yaml, if present."""
    config_path = _get_user_config_path()
    if not config_path.exists():
        return None
    try:
        return _parse_goose_mode(config_path.read_text(encoding="utf-8"))
    except OSError:
        return None


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

    @staticmethod
    def _build_subprocess_env() -> Optional[dict[str, str]]:
        """Return env overrides for Goose ACP process, or None if unchanged.

        Goose defaults to autonomous mode, which bypasses permission prompts.
        For ACP integration we default to `approve` mode so Goose routes tool
        permission requests through ACP and the Jupyter permission UI can render.
        Precedence mirrors OpenCode persona style:
        explicit env var > explicit config setting > ACP-safe default.
        """
        if "GOOSE_MODE" in os.environ:
            return None
        if _get_explicit_user_mode() is not None:
            return None
        env = os.environ.copy()
        env["GOOSE_MODE"] = "approve"
        return env

    async def _init_agent_subprocess(self) -> Process:
        env = self._build_subprocess_env()
        if env is None:
            mode = os.environ.get("GOOSE_MODE") or _get_explicit_user_mode()
            if mode is not None:
                self.log.info("[Goose] Respecting explicit GOOSE_MODE=%s.", mode)
        else:
            self.log.info("[Goose] Defaulting GOOSE_MODE=approve for ACP permission flow.")
        return await super()._init_agent_subprocess(env=env)

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
