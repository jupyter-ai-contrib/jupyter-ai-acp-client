import os
import re
import shutil
import subprocess
from asyncio.subprocess import Process
from pathlib import Path
from typing import NamedTuple

from acp.exceptions import RequestError
from jupyter_ai_persona_manager import PersonaDefaults, PersonaRequirementsUnmet
from jupyterlab_chat.models import Message
import yaml

from ..base_acp_persona import BaseAcpPersona

_DEFAULT_GOOSE_MODE = "approve"


def _is_setup_error(error: Exception) -> bool:
    """Return whether an error indicates Goose still needs setup."""
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


def _check_goose() -> None:
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
        current_version = tuple(int(x) for x in version_str.split("."))
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
    goose_path_root = os.environ.get("GOOSE_PATH_ROOT")
    if goose_path_root:
        return Path(goose_path_root) / "config" / "config.yaml"
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "Block" / "goose" / "config" / "config.yaml"
    config_home = os.environ.get("XDG_CONFIG_HOME")
    if config_home:
        return Path(config_home) / "goose" / "config.yaml"
    return Path.home() / ".config" / "goose" / "config.yaml"


def _parse_goose_config(config_text: str) -> dict[str, object] | None:
    """Parse Goose config YAML into a top-level mapping."""
    try:
        config = yaml.safe_load(config_text)
    except yaml.YAMLError:
        return None
    return config if isinstance(config, dict) else None


def _get_config_value(key: str) -> str | None:
    """Return a scalar value from Goose config, if present."""
    config_path = _get_user_config_path()
    if not config_path.exists():
        return None
    try:
        config = _parse_goose_config(config_path.read_text(encoding="utf-8"))
    except OSError:
        return None
    if config is None:
        return None
    value = config.get(key)
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _get_config_mode() -> str | None:
    """Return the explicit GOOSE_MODE from Goose config, if present."""
    return _get_config_value("GOOSE_MODE")


def _get_explicit_provider() -> str | None:
    """Return the explicit Goose provider from env or config."""
    return os.environ.get("GOOSE_PROVIDER") or _get_config_value("GOOSE_PROVIDER")


def _get_explicit_mode() -> str | None:
    """Return the explicit Goose mode from env or config."""
    return os.environ.get("GOOSE_MODE") or _get_config_mode()


class _GooseModeDecision(NamedTuple):
    mode: str
    env: dict[str, str] | None
    explicit: bool


def _resolve_mode_decision() -> _GooseModeDecision:
    """Resolve the mode Goose ACP should run with."""
    explicit_mode = _get_explicit_mode()
    if explicit_mode is not None:
        return _GooseModeDecision(
            mode=explicit_mode,
            env=None,
            explicit=True,
        )

    env = os.environ.copy()
    env["GOOSE_MODE"] = _DEFAULT_GOOSE_MODE
    return _GooseModeDecision(
        mode=_DEFAULT_GOOSE_MODE,
        env=env,
        explicit=False,
    )


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

    async def _init_agent_subprocess(self) -> Process:
        decision = _resolve_mode_decision()
        if decision.explicit:
            self.log.info("[Goose] Respecting explicit GOOSE_MODE=%s.", decision.mode)
        else:
            self.log.info(
                "[Goose] Defaulting GOOSE_MODE=%s for ACP permission flow.",
                decision.mode,
            )
        return await super()._init_agent_subprocess(env=decision.env)

    async def is_authed(self) -> bool:
        return _get_explicit_provider() is not None

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
