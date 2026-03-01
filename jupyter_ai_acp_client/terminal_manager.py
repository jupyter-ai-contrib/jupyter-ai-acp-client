"""Terminal manager for ACP client terminal operations."""

import asyncio
import logging
import os
import shlex
import signal as signal_module
import uuid
from asyncio.subprocess import Process
from dataclasses import dataclass, field
from typing import Any

from acp import RequestError
from acp.schema import (
    CreateTerminalResponse,
    EnvVariable,
    KillTerminalCommandResponse,
    ReleaseTerminalResponse,
    TerminalExitStatus,
    TerminalOutputResponse,
    WaitForTerminalExitResponse,
)

log = logging.getLogger(__name__)

# Default cap when the agent does not specify an output byte limit.
# 10 MiB keeps memory bounded while being generous for most commands.
DEFAULT_OUTPUT_BYTE_LIMIT: int = 10 * 1024 * 1024

# Hard ceiling on concurrent terminals per manager instance.
MAX_TERMINALS: int = 50

# Environment variable names that must never be overridden by agents.
# These enable library-injection attacks, code injection, or alter
# security-sensitive loader/interpreter behaviour.
# Sources: elttam.com/blog/env/, JupyterHub Issue #1654, Kibana CVE-2019-7609.
_DENIED_ENV_VARS: frozenset[str] = frozenset({
    # Native library injection
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "LD_AUDIT",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
    "DYLD_FRAMEWORK_PATH",
    # Python code injection
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONHOME",
    # Node.js code injection
    "NODE_OPTIONS",
    # Shell startup injection
    "BASH_ENV",
    "ENV",
})


def _log_output_task_exception(task: asyncio.Task) -> None:
    """Done callback for output reader tasks — logs unhandled exceptions."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.error("Terminal output reader failed", exc_info=exc)


@dataclass
class TerminalInfo:
    """Tracks state for a single terminal instance."""

    process: Process
    session_id: str
    output_buffer: bytearray = field(default_factory=bytearray)
    output_byte_limit: int | None = None
    truncated: bool = False
    exit_code: int | None = None
    exit_signal: str | None = None
    _output_task: asyncio.Task | None = field(default=None, repr=False)


class TerminalManager:
    """
    Manages terminal lifecycle for ACP client.

    Handles creation, output capture, waiting, killing, and releasing
    of terminal processes according to the ACP terminal protocol.
    """

    def __init__(self, event_loop: asyncio.AbstractEventLoop):
        """
        Initialize the terminal manager.

        :param event_loop: The asyncio event loop for creating background tasks.
        """
        self._event_loop = event_loop
        self._terminals: dict[str, TerminalInfo] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_terminal(self, terminal_id: str, session_id: str) -> TerminalInfo:
        """
        Validate terminal exists and belongs to the session.

        :raises RequestError: If terminal not found or belongs to different session.
        """
        info = self._terminals.get(terminal_id)
        if info is None:
            raise RequestError.resource_not_found(terminal_id)
        if info.session_id != session_id:
            raise RequestError.invalid_request(
                {"terminal_id": "terminal belongs to different session"}
            )
        return info

    @staticmethod
    def _set_exit_status(info: TerminalInfo, returncode: int | None) -> None:
        """
        Populate *exit_code* and *exit_signal* from a raw ``returncode``.

        On POSIX, a negative return code ``-N`` means the process was
        terminated by signal *N*.  The ACP schema constrains ``exit_code``
        to ``>= 0`` (Pydantic ``ge=0``), so we translate negative values
        into a human-readable signal name on ``exit_signal`` and leave
        ``exit_code`` as ``None``.
        """
        if returncode is None:
            info.exit_code = None
            info.exit_signal = None
        elif returncode < 0:
            info.exit_code = None
            try:
                info.exit_signal = signal_module.Signals(abs(returncode)).name
            except ValueError:
                info.exit_signal = f"SIG{abs(returncode)}"
        else:
            info.exit_code = returncode
            info.exit_signal = None

    @staticmethod
    def _trim_front_at_char_boundary(buf: bytearray, limit: int) -> None:
        """
        Remove the oldest bytes so that *buf* is at most *limit* bytes,
        preserving valid UTF-8 by skipping forward past any orphaned
        continuation bytes at the new start.

        Operates **in-place** on *buf*.
        """
        excess = len(buf) - limit
        if excess <= 0:
            return

        # Trim the front
        del buf[:excess]

        # The first byte(s) might now be UTF-8 continuation bytes
        # (10xxxxxx = 0x80..0xBF).  Skip forward to the next start byte.
        skip = 0
        while skip < len(buf) and (buf[skip] & 0xC0) == 0x80:
            skip += 1

        if skip:
            del buf[:skip]

    async def _read_terminal_output(self, terminal_id: str) -> None:
        """
        Background task to continuously read terminal output.

        Reads from stdout and respects output_byte_limit by keeping the
        **most recent** output (tail retention) and trimming the front,
        as required by the ACP protocol.
        """
        info = self._terminals.get(terminal_id)
        if info is None or info.process.stdout is None:
            return

        try:
            while True:
                chunk = await info.process.stdout.read(4096)
                if not chunk:
                    break

                info.output_buffer.extend(chunk)

                # Enforce byte limit via tail retention (trim the front).
                if (
                    info.output_byte_limit is not None
                    and len(info.output_buffer) > info.output_byte_limit
                ):
                    info.truncated = True
                    self._trim_front_at_char_boundary(
                        info.output_buffer, info.output_byte_limit
                    )

            # Process has finished, capture exit status
            exit_code = await info.process.wait()
            self._set_exit_status(info, exit_code)

        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception(
                "Unexpected error reading output for terminal %s", terminal_id
            )
            # Best-effort: capture exit status even on reader failure.
            try:
                exit_code = await info.process.wait()
                self._set_exit_status(info, exit_code)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def create_terminal(
        self,
        command: str,
        session_id: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: list[EnvVariable] | None = None,
        output_byte_limit: int | None = None,
        **kwargs: Any,
    ) -> CreateTerminalResponse:
        """
        Create a new terminal and start executing a command.

        Returns immediately with a terminal_id; the command runs in the background.
        """
        # Enforce terminal count limit
        if len(self._terminals) >= MAX_TERMINALS:
            raise RequestError.invalid_request(
                {
                    "terminal_id": (
                        f"terminal limit reached ({MAX_TERMINALS}); "
                        "release existing terminals first"
                    )
                }
            )

        # Validate command
        if not command or not command.strip():
            raise RequestError.invalid_params({"command": "command cannot be empty"})

        # Validate cwd if provided
        if cwd is not None:
            if not os.path.isabs(cwd):
                raise RequestError.invalid_params(
                    {"cwd": "cwd must be an absolute path"}
                )
            if not os.path.isdir(cwd):
                raise RequestError.invalid_params(
                    {"cwd": "cwd directory does not exist"}
                )

        # Build environment dict
        env_dict = None
        if env:
            env_dict = os.environ.copy()
            for e in env:
                upper_name = e.name.upper()
                if upper_name in _DENIED_ENV_VARS or e.name in _DENIED_ENV_VARS:
                    raise RequestError.invalid_params(
                        {"env": f"setting {e.name!r} is not allowed"}
                    )
                env_dict[e.name] = e.value

        # Apply default output byte limit when the agent omits it.
        # output_byte_limit=0 means "retain nothing" and is honoured as-is.
        effective_byte_limit = (
            output_byte_limit
            if output_byte_limit is not None
            else DEFAULT_OUTPUT_BYTE_LIMIT
        )

        # Build command arguments.
        # ACP spec defines `command` as the executable and `args` as argv[1:], but
        # agents commonly send shell-style strings (e.g. 'ls -la'). When explicit
        # args are provided, use them as-is. Otherwise split the command string so
        # 'ls -la' becomes ['ls', '-la'] rather than ['ls -la'] (which fails with
        # FileNotFoundError since no executable is literally named 'ls -la').
        if args:
            cmd_args = [command] + args
        else:
            try:
                cmd_args = shlex.split(command)
            except ValueError as e:
                raise RequestError.invalid_params(
                    {"command": f"could not parse command: {e}"}
                )

        if not cmd_args:
            raise RequestError.invalid_params({"command": "command cannot be empty"})

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                cwd=cwd,
                env=env_dict,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,  # Merge stderr into stdout
                start_new_session=True,  # New process group for clean kill
            )
        except FileNotFoundError:
            raise RequestError.invalid_params(
                {"command": f"command not found: {command}"}
            )
        except PermissionError:
            raise RequestError.invalid_params(
                {"command": f"permission denied: {command}"}
            )
        except OSError as e:
            raise RequestError.internal_error({"command": command, "error": str(e)})

        terminal_id = str(uuid.uuid4())
        info = TerminalInfo(
            process=process,
            session_id=session_id,
            output_byte_limit=effective_byte_limit,
        )
        self._terminals[terminal_id] = info

        # Start background task to read output. Register a done callback so that
        # any unexpected exception (e.g. OSError from a broken pipe) is logged
        # immediately rather than silently discarded.
        info._output_task = self._event_loop.create_task(
            self._read_terminal_output(terminal_id)
        )
        info._output_task.add_done_callback(_log_output_task_exception)

        return CreateTerminalResponse(terminal_id=terminal_id)

    async def terminal_output(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> TerminalOutputResponse:
        """
        Retrieve current terminal output without blocking.

        Returns the captured output so far, truncation status, and exit status
        if the command has finished.
        """
        info = self._validate_terminal(terminal_id, session_id)

        output = info.output_buffer.decode("utf-8", errors="replace")

        # Build exit_status if process has finished.
        # Use process.returncode directly to avoid a race where the
        # background reader has not yet populated info.exit_code.
        exit_status = None
        if info.process.returncode is not None:
            self._set_exit_status(info, info.process.returncode)
            exit_status = TerminalExitStatus(
                exit_code=info.exit_code,
                signal=info.exit_signal,
            )

        return TerminalOutputResponse(
            output=output,
            truncated=info.truncated,
            exit_status=exit_status,
        )

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> WaitForTerminalExitResponse:
        """
        Block until the terminal command completes.

        Returns the exit code and/or signal that terminated the process.
        """
        info = self._validate_terminal(terminal_id, session_id)

        # Wait for the process to complete
        exit_code = await info.process.wait()
        self._set_exit_status(info, exit_code)

        return WaitForTerminalExitResponse(
            exit_code=info.exit_code,
            signal=info.exit_signal,
        )

    async def kill_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> KillTerminalCommandResponse | None:
        """
        Terminate a running command without releasing resources.

        The terminal remains valid for subsequent terminal_output and
        wait_for_terminal_exit calls. The agent must still call
        release_terminal afterward.
        """
        info = self._validate_terminal(terminal_id, session_id)

        if info.process.returncode is None:
            # Kill the entire process group so child processes are cleaned up.
            try:
                os.killpg(os.getpgid(info.process.pid), signal_module.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                # Process already exited or is inaccessible — fall back to
                # direct kill which is a no-op if already dead.
                info.process.kill()
            exit_code = await info.process.wait()
            self._set_exit_status(info, exit_code)

        return KillTerminalCommandResponse()

    async def release_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> ReleaseTerminalResponse | None:
        """
        Kill any running command and deallocate all resources.

        After release, the terminal_id becomes invalid.
        """
        info = self._validate_terminal(terminal_id, session_id)

        # Kill process if still running
        if info.process.returncode is None:
            try:
                os.killpg(os.getpgid(info.process.pid), signal_module.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                info.process.kill()
            await info.process.wait()

        # Cancel the output reading task if it's still running
        if info._output_task is not None and not info._output_task.done():
            info._output_task.cancel()
            try:
                await info._output_task
            except asyncio.CancelledError:
                pass

        # Remove from tracking
        del self._terminals[terminal_id]

        return ReleaseTerminalResponse()

    async def cleanup_session(self, session_id: str) -> None:
        """
        Clean up all terminals associated with a session.

        Should be called when a session ends.
        """
        terminal_ids = [
            tid
            for tid, info in self._terminals.items()
            if info.session_id == session_id
        ]
        for terminal_id in terminal_ids:
            try:
                await self.release_terminal(session_id, terminal_id)
            except Exception:
                log.warning(
                    "Failed to release terminal %s for session %s",
                    terminal_id,
                    session_id,
                    exc_info=True,
                )
