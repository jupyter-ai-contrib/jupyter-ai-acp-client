"""Terminal manager for ACP client terminal operations."""

import asyncio
import os
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

    async def _read_terminal_output(self, terminal_id: str) -> None:
        """
        Background task to continuously read terminal output.

        Reads from stdout and respects output_byte_limit with character
        boundary truncation as required by the ACP protocol.
        """
        info = self._terminals.get(terminal_id)
        if info is None or info.process.stdout is None:
            return

        try:
            while True:
                chunk = await info.process.stdout.read(4096)
                if not chunk:
                    break

                # Check if we need to truncate
                if info.output_byte_limit is not None:
                    current_len = len(info.output_buffer)
                    if current_len >= info.output_byte_limit:
                        # Already at limit, mark as truncated but don't add more
                        info.truncated = True
                        continue

                    remaining = info.output_byte_limit - current_len
                    if len(chunk) > remaining:
                        # Need to truncate - ensure we truncate at character boundary
                        chunk = self._truncate_at_char_boundary(chunk, remaining)
                        info.truncated = True

                info.output_buffer.extend(chunk)

            # Process has finished, capture exit status
            exit_code = await info.process.wait()
            info.exit_code = exit_code

        except asyncio.CancelledError:
            pass

    def _truncate_at_char_boundary(self, data: bytes, max_bytes: int) -> bytes:
        """
        Truncate bytes at a valid UTF-8 character boundary.

        The ACP protocol requires truncation at character boundaries to
        maintain valid string output.
        """
        if max_bytes <= 0:
            return b""

        truncated = data[:max_bytes]

        # Walk back from the end to find a valid UTF-8 boundary
        # UTF-8 continuation bytes start with 10xxxxxx (0x80-0xBF)
        while truncated and (truncated[-1] & 0xC0) == 0x80:
            truncated = truncated[:-1]

        # If we're at a lead byte of a multi-byte sequence that got cut off,
        # remove it too
        if truncated:
            last_byte = truncated[-1]
            # Check if it's a multi-byte lead byte (11xxxxxx)
            if last_byte >= 0xC0:
                # Count expected continuation bytes
                if last_byte >= 0xF0:
                    expected_len = 4
                elif last_byte >= 0xE0:
                    expected_len = 3
                elif last_byte >= 0xC0:
                    expected_len = 2
                else:
                    expected_len = 1

                # Incomplete sequence, remove the lead byte
                truncated = truncated[:-1]

        return truncated

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
                env_dict[e.name] = e.value

        # Build command arguments
        cmd_args = [command] + (args or [])

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                cwd=cwd,
                env=env_dict,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,  # Merge stderr into stdout
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
            output_byte_limit=output_byte_limit,
        )
        self._terminals[terminal_id] = info

        # Start background task to read output
        info._output_task = self._event_loop.create_task(
            self._read_terminal_output(terminal_id)
        )

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

        # Build exit_status if process has finished
        exit_status = None
        if info.process.returncode is not None:
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
        info.exit_code = exit_code

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
            # Process is still running, kill it
            info.process.kill()
            exit_code = await info.process.wait()
            info.exit_code = exit_code
            info.exit_signal = "SIGKILL"

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
                pass  # Best effort cleanup
