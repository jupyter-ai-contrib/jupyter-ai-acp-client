"""Windows-compatible subprocess wrapper that provides asyncio StreamReader/StreamWriter.

On Windows with Tornado's SelectorEventLoop, asyncio.create_subprocess_exec() raises
NotImplementedError because SelectorEventLoop doesn't support subprocess operations.
This module provides a fallback that uses subprocess.Popen and bridges its pipes to
asyncio.StreamReader/StreamWriter via background threads.
"""

import asyncio
import subprocess
import sys
import threading
from typing import Any, Optional


class _WriteTransport(asyncio.Transport):
    """Minimal asyncio.Transport wrapping a synchronous pipe for StreamWriter."""

    def __init__(self, pipe, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self._pipe = pipe
        self._loop = loop
        self._closing = False

    def write(self, data: bytes) -> None:
        if self._closing:
            return
        try:
            self._pipe.write(data)
            self._pipe.flush()
        except OSError:
            pass

    def can_write_eof(self) -> bool:
        return False

    def is_closing(self) -> bool:
        return self._closing

    def close(self) -> None:
        self._closing = True
        try:
            self._pipe.close()
        except OSError:
            pass

    def get_extra_info(self, name: str, default=None):
        return default


class WindowsProcess:
    """A subprocess.Popen wrapper that mimics asyncio.subprocess.Process.

    Provides .stdin (asyncio.StreamWriter) and .stdout (asyncio.StreamReader)
    compatible with ACP's ClientSideConnection isinstance checks.
    """

    def __init__(self, popen: subprocess.Popen, loop: asyncio.AbstractEventLoop):
        self._popen = popen
        self._loop = loop
        self._reader = asyncio.StreamReader(limit=50 * 1024 * 1024)
        self._reader_thread: Optional[threading.Thread] = None

        # Build StreamWriter around stdin pipe
        transport = _WriteTransport(popen.stdin, loop)
        protocol = asyncio.StreamReaderProtocol(asyncio.StreamReader())
        self._writer = asyncio.StreamWriter(transport, protocol, None, loop)

        # Start background thread to feed stdout data into the StreamReader
        self._reader_thread = threading.Thread(
            target=self._read_stdout, daemon=True, name="acp-win32-stdout-reader"
        )
        self._reader_thread.start()

    def _read_stdout(self) -> None:
        """Background thread: reads from Popen.stdout and feeds asyncio.StreamReader."""
        try:
            while True:
                data = self._popen.stdout.read(65536)
                if not data:
                    break
                self._loop.call_soon_threadsafe(self._reader.feed_data, data)
        except (OSError, ValueError):
            pass
        finally:
            self._loop.call_soon_threadsafe(self._reader.feed_eof)

    @property
    def stdin(self) -> asyncio.StreamWriter:
        return self._writer

    @property
    def stdout(self) -> asyncio.StreamReader:
        return self._reader

    @property
    def pid(self) -> int:
        return self._popen.pid

    @property
    def returncode(self) -> Optional[int]:
        return self._popen.returncode

    async def wait(self) -> int:
        """Wait for the process to exit."""
        return await self._loop.run_in_executor(None, self._popen.wait)

    def terminate(self) -> None:
        self._popen.terminate()

    def kill(self) -> None:
        self._popen.kill()


async def create_subprocess_windows(
    *args: str,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
    limit: int = 50 * 1024 * 1024,
    env: Optional[dict[str, str]] = None,
    **kwargs: Any,
) -> WindowsProcess:
    """Create a subprocess on Windows using Popen, returning a WindowsProcess
    that provides asyncio.StreamReader/StreamWriter for ACP compatibility."""
    loop = asyncio.get_event_loop()
    popen_kwargs: dict[str, Any] = dict(
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr if stderr is not None else sys.stderr,
        bufsize=0,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if env is not None:
        popen_kwargs["env"] = env
    popen = subprocess.Popen(list(args), **popen_kwargs)
    return WindowsProcess(popen, loop)
