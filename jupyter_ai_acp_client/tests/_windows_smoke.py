"""
End-to-end Windows smoke test, run from CI rather than under pytest.

The unit tests exercise the subprocess bridge against `sys.executable`. This
script exercises it against a real npm-installed ACP adapter, on the same kind
of event loop `jupyter_server` gives the extension, and drives it with the real
ACP SDK. It therefore covers both Windows defects at once:

  1. `SelectorEventLoop` cannot spawn subprocesses      -> NotImplementedError
  2. `CreateProcessW` only resolves `.exe`, so a bare
     command name never finds an npm `.cmd` shim        -> FileNotFoundError

Passing an unresolved bare name here is deliberate: it is exactly what the
persona classes do, and it is what defect 2 broke.

Exits non-zero on failure.
"""

import asyncio
import subprocess
import sys

from acp import PROTOCOL_VERSION, connect_to_agent
from acp.schema import ClientCapabilities, FileSystemCapabilities

from jupyter_ai_acp_client._win32_subprocess import create_subprocess, terminate_process

AGENT = "claude-agent-acp"


def _descendant_node_pids() -> set[int]:
    """PIDs of running `node` processes belonging to an ACP adapter."""
    out = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name='node.exe'\""
            " | Where-Object { $_.CommandLine -like '*acp*' }"
            " | ForEach-Object { $_.ProcessId }",
        ],
        capture_output=True,
        text=True,
    ).stdout
    return {int(line) for line in out.split() if line.strip().isdigit()}


class _NullClient:
    """Minimal ACP client: `initialize` needs no callbacks to come back."""

    async def request_permission(self, *args, **kwargs):
        raise NotImplementedError

    async def session_update(self, *args, **kwargs):
        return None

    async def write_text_file(self, *args, **kwargs):
        raise NotImplementedError

    async def read_text_file(self, *args, **kwargs):
        raise NotImplementedError

    async def create_terminal(self, *args, **kwargs):
        raise NotImplementedError

    async def terminal_output(self, *args, **kwargs):
        raise NotImplementedError

    async def wait_for_terminal_exit(self, *args, **kwargs):
        raise NotImplementedError

    async def kill_terminal(self, *args, **kwargs):
        raise NotImplementedError

    async def release_terminal(self, *args, **kwargs):
        raise NotImplementedError

    async def ext_method(self, *args, **kwargs):
        raise NotImplementedError

    async def ext_notification(self, *args, **kwargs):
        return None


async def main() -> None:
    loop = asyncio.get_running_loop()
    print("event loop:", type(loop).__name__)
    assert not hasattr(loop, "_proactor"), (
        "expected a SelectorEventLoop; this test is meaningless on a Proactor loop"
    )

    before = _descendant_node_pids()

    proc = await create_subprocess(
        AGENT,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=sys.stderr,
        limit=50 * 1024 * 1024,
    )
    print("spawned %r as pid %s via %s" % (AGENT, proc.pid, type(proc).__name__))

    try:
        conn = connect_to_agent(_NullClient(), proc.stdin, proc.stdout)
        response = await asyncio.wait_for(
            conn.initialize(
                protocol_version=PROTOCOL_VERSION,
                client_capabilities=ClientCapabilities(
                    fs=FileSystemCapabilities(
                        read_text_file=False, write_text_file=False
                    )
                ),
            ),
            timeout=120,
        )
        print("ACP initialize OK; protocol version", response.protocol_version)

        spawned = _descendant_node_pids() - before
        assert spawned, "expected the `.cmd` shim to have started a node agent"
        print("agent node pid(s):", sorted(spawned))
    finally:
        # Close the connection first so the SDK's reader/sender tasks stop
        # cleanly, then terminate. The adapter runs as `cmd.exe` -> `node`, so
        # this is the real test of tree termination: killing the shim alone
        # would leave node orphaned.
        try:
            await asyncio.wait_for(conn.close(), timeout=10)
        except Exception:
            pass
        await terminate_process(proc, timeout=15)

    for _ in range(50):
        survivors = _descendant_node_pids() & spawned
        if not survivors:
            break
        await asyncio.sleep(0.2)
    assert not survivors, "orphaned agent process(es) after shutdown: %s" % sorted(
        survivors
    )
    print("process tree terminated cleanly; no orphaned agents")


if __name__ == "__main__":
    if sys.platform != "win32":
        print("skipped: Windows only")
        raise SystemExit(0)

    # Reproduce what jupyter_server does to the event loop on Windows.
    loop = asyncio.SelectorEventLoop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
    print("Windows ACP smoke test passed.")
