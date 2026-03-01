"""Comprehensive tests for TerminalManager.

Covers:
  - Signal-aware exit status translation (_set_exit_status)
  - Front-trim at UTF-8 char boundary (_trim_front_at_char_boundary)
  - Tail-retention truncation in _read_terminal_output
  - Terminal count limit (MAX_TERMINALS)
  - Denied environment variables (_DENIED_ENV_VARS)
  - Default output byte limit (DEFAULT_OUTPUT_BYTE_LIMIT)
  - Process group kill (os.killpg with fallback)
  - Cleanup session logging
  - Race-safe terminal_output exit status
"""

import asyncio
import signal as signal_module
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from acp import RequestError

from jupyter_ai_acp_client.terminal_manager import (
    DEFAULT_OUTPUT_BYTE_LIMIT,
    MAX_TERMINALS,
    TerminalInfo,
    TerminalManager,
    _DENIED_ENV_VARS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager() -> TerminalManager:
    """Create a TerminalManager with the running event loop."""
    loop = asyncio.get_running_loop()
    return TerminalManager(loop)


def _make_info(
    returncode: int | None = None,
    exit_code: int | None = None,
    exit_signal: str | None = None,
    output: bytes = b"",
    byte_limit: int | None = None,
) -> TerminalInfo:
    """Build a TerminalInfo with a mocked process."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.pid = 12345
    proc.stdout = None
    proc.wait = AsyncMock(return_value=returncode if returncode is not None else 0)
    proc.kill = MagicMock()

    info = TerminalInfo(
        process=proc,
        session_id="sess-1",
        output_byte_limit=byte_limit,
    )
    info.output_buffer = bytearray(output)
    info.exit_code = exit_code
    info.exit_signal = exit_signal
    return info


SESSION = "sess-1"


# ===================================================================
# _set_exit_status
# ===================================================================

class TestSetExitStatus:
    """Tests for signal-aware exit status translation."""

    async def test_positive_exit_code(self):
        info = _make_info()
        TerminalManager._set_exit_status(info, 0)
        assert info.exit_code == 0
        assert info.exit_signal is None

    async def test_nonzero_positive_exit_code(self):
        info = _make_info()
        TerminalManager._set_exit_status(info, 1)
        assert info.exit_code == 1
        assert info.exit_signal is None

    async def test_negative_sigkill(self):
        info = _make_info()
        TerminalManager._set_exit_status(info, -9)
        assert info.exit_code is None
        assert info.exit_signal == "SIGKILL"

    async def test_negative_sigterm(self):
        info = _make_info()
        TerminalManager._set_exit_status(info, -15)
        assert info.exit_code is None
        assert info.exit_signal == "SIGTERM"

    async def test_negative_sigint(self):
        info = _make_info()
        TerminalManager._set_exit_status(info, -2)
        assert info.exit_code is None
        assert info.exit_signal == "SIGINT"

    async def test_negative_unknown_signal(self):
        info = _make_info()
        TerminalManager._set_exit_status(info, -255)
        assert info.exit_code is None
        assert info.exit_signal == "SIG255"

    async def test_none_returncode(self):
        info = _make_info()
        TerminalManager._set_exit_status(info, None)
        assert info.exit_code is None
        assert info.exit_signal is None

    async def test_idempotent(self):
        info = _make_info()
        TerminalManager._set_exit_status(info, -9)
        TerminalManager._set_exit_status(info, -9)
        assert info.exit_code is None
        assert info.exit_signal == "SIGKILL"


# ===================================================================
# _trim_front_at_char_boundary
# ===================================================================

class TestTrimFrontAtCharBoundary:
    """Tests for UTF-8 safe front-trimming."""

    async def test_no_trim_needed(self):
        buf = bytearray(b"hello")
        TerminalManager._trim_front_at_char_boundary(buf, 10)
        assert buf == bytearray(b"hello")

    async def test_exact_limit(self):
        buf = bytearray(b"hello")
        TerminalManager._trim_front_at_char_boundary(buf, 5)
        assert buf == bytearray(b"hello")

    async def test_trim_ascii(self):
        buf = bytearray(b"abcdef")
        TerminalManager._trim_front_at_char_boundary(buf, 3)
        assert buf == bytearray(b"def")

    async def test_trim_preserves_complete_utf8_2byte(self):
        # "Aé" = b'A\xc3\xa9', trim to 2 bytes → removes "A", cut lands
        # on \xa9 (continuation), skip forward past it → empty or next char
        buf = bytearray("Aé".encode("utf-8"))  # b'A\xc3\xa9'
        TerminalManager._trim_front_at_char_boundary(buf, 2)
        # After trimming front by 1 byte: b'\xc3\xa9' (2 bytes) — that's valid "é"
        assert buf == "é".encode("utf-8")

    async def test_trim_skips_orphaned_continuation(self):
        # b'\xc3\xa9XY' (4 bytes), limit=2 → excess=2, del [:2] → b'XY'
        buf = bytearray(b"\xc3\xa9XY")
        TerminalManager._trim_front_at_char_boundary(buf, 2)
        assert buf == bytearray(b"XY")

    async def test_trim_cuts_into_multibyte(self):
        # "€" = b'\xe2\x82\xac' (3 bytes)
        # "A€B" = b'A\xe2\x82\xacB' (5 bytes), limit=3 → excess=2
        # del [:2] → b'\x82\xacB', \x82 is continuation → skip 2 cont. bytes → b'B'
        buf = bytearray("A€B".encode("utf-8"))
        TerminalManager._trim_front_at_char_boundary(buf, 3)
        assert buf == bytearray(b"B")

    async def test_trim_4byte_char(self):
        # "𝄞" = b'\xf0\x9d\x84\x9e' (4 bytes)
        # "A𝄞B" (6 bytes), limit=4 → excess=2
        # del [:2] → b'\x84\x9eB', both cont. → skip 2 → b'B'
        buf = bytearray("A𝄞B".encode("utf-8"))
        TerminalManager._trim_front_at_char_boundary(buf, 4)
        assert buf == bytearray(b"B")

    async def test_trim_to_zero(self):
        buf = bytearray(b"abc")
        TerminalManager._trim_front_at_char_boundary(buf, 0)
        assert buf == bytearray(b"")

    async def test_empty_buffer(self):
        buf = bytearray(b"")
        TerminalManager._trim_front_at_char_boundary(buf, 0)
        assert buf == bytearray(b"")

    async def test_all_continuation_bytes_after_trim(self):
        # Buffer is only continuation bytes after front-trim → all skipped
        buf = bytearray(b"\x80\x80\x80")
        TerminalManager._trim_front_at_char_boundary(buf, 2)
        # excess=1, del [:1] → b'\x80\x80', both continuation → skip all → empty
        assert buf == bytearray(b"")


# ===================================================================
# create_terminal
# ===================================================================

class TestCreateTerminal:
    """Tests for create_terminal validation and limits."""

    async def test_empty_command_rejected(self):
        mgr = _make_manager()
        with pytest.raises(RequestError):
            await mgr.create_terminal(command="", session_id=SESSION)

    async def test_whitespace_command_rejected(self):
        mgr = _make_manager()
        with pytest.raises(RequestError):
            await mgr.create_terminal(command="   ", session_id=SESSION)

    async def test_relative_cwd_rejected(self):
        mgr = _make_manager()
        with pytest.raises(RequestError):
            await mgr.create_terminal(command="echo hi", session_id=SESSION, cwd="relative/path")

    async def test_nonexistent_cwd_rejected(self):
        mgr = _make_manager()
        with pytest.raises(RequestError):
            await mgr.create_terminal(
                command="echo hi", session_id=SESSION, cwd="/nonexistent_dir_xyz_123"
            )

    async def test_terminal_count_limit(self):
        mgr = _make_manager()
        # Fill up the terminal dict with dummy entries
        for i in range(MAX_TERMINALS):
            mgr._terminals[f"term-{i}"] = _make_info()

        with pytest.raises(RequestError):
            await mgr.create_terminal(command="echo hi", session_id=SESSION)

    async def test_denied_env_var_ld_preload(self):
        mgr = _make_manager()
        env = [MagicMock(name="LD_PRELOAD", value="/evil.so")]
        env[0].name = "LD_PRELOAD"
        env[0].value = "/evil.so"
        with pytest.raises(RequestError):
            await mgr.create_terminal(command="echo hi", session_id=SESSION, env=env)

    async def test_denied_env_var_case_insensitive(self):
        mgr = _make_manager()
        env = [MagicMock()]
        env[0].name = "ld_preload"
        env[0].value = "/evil.so"
        with pytest.raises(RequestError):
            await mgr.create_terminal(command="echo hi", session_id=SESSION, env=env)

    async def test_allowed_env_var_passes(self):
        mgr = _make_manager()
        env = [MagicMock()]
        env[0].name = "MY_VAR"
        env[0].value = "hello"
        # Should not raise for allowed env vars (but will fail on actual
        # subprocess exec — we just verify env validation passes)
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = MagicMock()
            mock_proc.stdout = AsyncMock()
            mock_proc.stdout.read = AsyncMock(return_value=b"")
            mock_proc.returncode = 0
            mock_proc.wait = AsyncMock(return_value=0)
            mock_exec.return_value = mock_proc
            resp = await mgr.create_terminal(command="echo hi", session_id=SESSION, env=env)
            assert resp.terminal_id is not None

    async def test_default_byte_limit_applied(self):
        mgr = _make_manager()
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = MagicMock()
            mock_proc.stdout = AsyncMock()
            mock_proc.stdout.read = AsyncMock(return_value=b"")
            mock_proc.returncode = 0
            mock_proc.wait = AsyncMock(return_value=0)
            mock_exec.return_value = mock_proc
            resp = await mgr.create_terminal(command="echo hi", session_id=SESSION)
            info = mgr._terminals[resp.terminal_id]
            assert info.output_byte_limit == DEFAULT_OUTPUT_BYTE_LIMIT

    async def test_explicit_byte_limit_zero_honoured(self):
        mgr = _make_manager()
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = MagicMock()
            mock_proc.stdout = AsyncMock()
            mock_proc.stdout.read = AsyncMock(return_value=b"")
            mock_proc.returncode = 0
            mock_proc.wait = AsyncMock(return_value=0)
            mock_exec.return_value = mock_proc
            resp = await mgr.create_terminal(
                command="echo hi", session_id=SESSION, output_byte_limit=0
            )
            info = mgr._terminals[resp.terminal_id]
            assert info.output_byte_limit == 0

    async def test_start_new_session_passed(self):
        mgr = _make_manager()
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = MagicMock()
            mock_proc.stdout = AsyncMock()
            mock_proc.stdout.read = AsyncMock(return_value=b"")
            mock_proc.returncode = 0
            mock_proc.wait = AsyncMock(return_value=0)
            mock_exec.return_value = mock_proc
            await mgr.create_terminal(command="echo hi", session_id=SESSION)
            _, kwargs = mock_exec.call_args
            assert kwargs.get("start_new_session") is True


# ===================================================================
# terminal_output
# ===================================================================

class TestTerminalOutput:
    """Tests for terminal_output including race-safe exit status."""

    async def test_returns_output(self):
        mgr = _make_manager()
        info = _make_info(output=b"hello world")
        mgr._terminals["t1"] = info
        resp = await mgr.terminal_output(session_id=SESSION, terminal_id="t1")
        assert resp.output == "hello world"

    async def test_exit_status_none_while_running(self):
        mgr = _make_manager()
        info = _make_info(returncode=None)
        mgr._terminals["t1"] = info
        resp = await mgr.terminal_output(session_id=SESSION, terminal_id="t1")
        assert resp.exit_status is None

    async def test_exit_status_from_returncode(self):
        mgr = _make_manager()
        info = _make_info(returncode=0)
        mgr._terminals["t1"] = info
        resp = await mgr.terminal_output(session_id=SESSION, terminal_id="t1")
        assert resp.exit_status is not None
        assert resp.exit_status.exit_code == 0
        assert resp.exit_status.signal is None

    async def test_exit_status_signal_from_returncode(self):
        """Race-safe: uses process.returncode, not info.exit_code."""
        mgr = _make_manager()
        info = _make_info(returncode=-9)
        # Simulate race: background reader hasn't set info.exit_code yet
        info.exit_code = None
        info.exit_signal = None
        mgr._terminals["t1"] = info
        resp = await mgr.terminal_output(session_id=SESSION, terminal_id="t1")
        assert resp.exit_status is not None
        assert resp.exit_status.exit_code is None
        assert resp.exit_status.signal == "SIGKILL"

    async def test_invalid_terminal_raises(self):
        mgr = _make_manager()
        with pytest.raises(RequestError):
            await mgr.terminal_output(session_id=SESSION, terminal_id="nonexistent")

    async def test_wrong_session_raises(self):
        mgr = _make_manager()
        info = _make_info()
        info.session_id = "other-session"
        mgr._terminals["t1"] = info
        with pytest.raises(RequestError):
            await mgr.terminal_output(session_id=SESSION, terminal_id="t1")


# ===================================================================
# wait_for_terminal_exit
# ===================================================================

class TestWaitForTerminalExit:
    """Tests for wait_for_terminal_exit with signal-aware handling."""

    async def test_normal_exit(self):
        mgr = _make_manager()
        info = _make_info(returncode=0)
        info.process.wait = AsyncMock(return_value=0)
        mgr._terminals["t1"] = info
        resp = await mgr.wait_for_terminal_exit(session_id=SESSION, terminal_id="t1")
        assert resp.exit_code == 0
        assert resp.signal is None

    async def test_signal_exit(self):
        mgr = _make_manager()
        info = _make_info()
        info.process.wait = AsyncMock(return_value=-15)
        mgr._terminals["t1"] = info
        resp = await mgr.wait_for_terminal_exit(session_id=SESSION, terminal_id="t1")
        assert resp.exit_code is None
        assert resp.signal == "SIGTERM"


# ===================================================================
# kill_terminal
# ===================================================================

class TestKillTerminal:
    """Tests for kill_terminal with process group kill and fallback."""

    async def test_kill_running_process(self):
        mgr = _make_manager()
        info = _make_info(returncode=None)
        info.process.wait = AsyncMock(return_value=-9)
        info.process.returncode = None
        mgr._terminals["t1"] = info

        with patch("os.killpg") as mock_killpg, patch("os.getpgid", return_value=12345):
            await mgr.kill_terminal(session_id=SESSION, terminal_id="t1")
            mock_killpg.assert_called_once_with(12345, signal_module.SIGKILL)

        assert info.exit_code is None
        assert info.exit_signal == "SIGKILL"

    async def test_kill_falls_back_on_lookup_error(self):
        mgr = _make_manager()
        info = _make_info(returncode=None)
        info.process.wait = AsyncMock(return_value=-9)
        info.process.returncode = None
        mgr._terminals["t1"] = info

        with patch("os.killpg", side_effect=ProcessLookupError), \
             patch("os.getpgid", return_value=12345):
            await mgr.kill_terminal(session_id=SESSION, terminal_id="t1")
            info.process.kill.assert_called_once()

    async def test_kill_already_exited(self):
        mgr = _make_manager()
        info = _make_info(returncode=0)
        mgr._terminals["t1"] = info
        resp = await mgr.kill_terminal(session_id=SESSION, terminal_id="t1")
        assert resp is not None
        # Should not have called kill since process already exited
        info.process.kill.assert_not_called()

    async def test_kill_uses_set_exit_status(self):
        """After kill, exit_signal should reflect actual signal, not hard-coded."""
        mgr = _make_manager()
        info = _make_info(returncode=None)
        # Simulate process was already dying from SIGTERM before our SIGKILL
        info.process.wait = AsyncMock(return_value=-15)
        info.process.returncode = None
        mgr._terminals["t1"] = info

        with patch("os.killpg"), patch("os.getpgid", return_value=12345):
            await mgr.kill_terminal(session_id=SESSION, terminal_id="t1")

        assert info.exit_signal == "SIGTERM"


# ===================================================================
# release_terminal
# ===================================================================

class TestReleaseTerminal:
    """Tests for release_terminal resource cleanup."""

    async def test_release_removes_terminal(self):
        mgr = _make_manager()
        info = _make_info(returncode=0)
        info._output_task = MagicMock()
        info._output_task.done.return_value = True
        mgr._terminals["t1"] = info
        await mgr.release_terminal(session_id=SESSION, terminal_id="t1")
        assert "t1" not in mgr._terminals

    async def test_release_kills_running_process(self):
        mgr = _make_manager()
        info = _make_info(returncode=None)
        info.process.returncode = None
        info.process.wait = AsyncMock(return_value=-9)
        info._output_task = MagicMock()
        info._output_task.done.return_value = True
        mgr._terminals["t1"] = info

        with patch("os.killpg") as mock_killpg, patch("os.getpgid", return_value=12345):
            await mgr.release_terminal(session_id=SESSION, terminal_id="t1")
            mock_killpg.assert_called_once()

        assert "t1" not in mgr._terminals

    async def test_release_cancels_output_task(self):
        mgr = _make_manager()
        info = _make_info(returncode=0)

        # Create a real asyncio task that we can cancel.
        async def _hang_forever():
            await asyncio.sleep(3600)

        real_task = asyncio.get_running_loop().create_task(_hang_forever())
        info._output_task = real_task
        mgr._terminals["t1"] = info

        await mgr.release_terminal(session_id=SESSION, terminal_id="t1")
        assert real_task.cancelled()


# ===================================================================
# cleanup_session
# ===================================================================

class TestCleanupSession:
    """Tests for cleanup_session logging on failure."""

    async def test_cleans_all_session_terminals(self):
        mgr = _make_manager()
        for i in range(3):
            info = _make_info(returncode=0)
            info.session_id = SESSION
            info._output_task = MagicMock()
            info._output_task.done.return_value = True
            mgr._terminals[f"t{i}"] = info

        # Add a terminal from another session
        other = _make_info(returncode=0)
        other.session_id = "other-sess"
        other._output_task = MagicMock()
        other._output_task.done.return_value = True
        mgr._terminals["other"] = other

        await mgr.cleanup_session(SESSION)
        assert "other" in mgr._terminals
        assert all(f"t{i}" not in mgr._terminals for i in range(3))

    async def test_logs_warning_on_failure(self):
        mgr = _make_manager()
        info = _make_info(returncode=0)
        info.session_id = SESSION
        mgr._terminals["t1"] = info

        with patch.object(
            mgr, "release_terminal", new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ), patch("jupyter_ai_acp_client.terminal_manager.log") as mock_log:
            await mgr.cleanup_session(SESSION)
            mock_log.warning.assert_called_once()
            # Should include terminal_id and session_id
            args = mock_log.warning.call_args
            assert "t1" in str(args)
            assert SESSION in str(args)


# ===================================================================
# _read_terminal_output (tail retention integration)
# ===================================================================

class TestReadTerminalOutput:
    """Integration-style tests for the background output reader."""

    async def test_tail_retention(self):
        mgr = _make_manager()
        info = _make_info(byte_limit=10)
        mgr._terminals["t1"] = info

        # Simulate stdout that produces chunks
        chunks = [b"AAAAAAAAAA", b"BBBBBBBBBB", b""]  # 10 + 10 bytes, limit 10
        async def fake_read(n):
            return chunks.pop(0) if chunks else b""
        info.process.stdout = MagicMock()
        info.process.stdout.read = fake_read
        info.process.wait = AsyncMock(return_value=0)

        await mgr._read_terminal_output("t1")

        # Should keep the tail (most recent 10 bytes)
        assert info.output_buffer == bytearray(b"BBBBBBBBBB")
        assert info.truncated is True
        assert info.exit_code == 0

    async def test_no_truncation_under_limit(self):
        mgr = _make_manager()
        info = _make_info(byte_limit=100)
        mgr._terminals["t1"] = info

        chunks = [b"hello", b""]
        async def fake_read(n):
            return chunks.pop(0) if chunks else b""
        info.process.stdout = MagicMock()
        info.process.stdout.read = fake_read
        info.process.wait = AsyncMock(return_value=0)

        await mgr._read_terminal_output("t1")

        assert info.output_buffer == bytearray(b"hello")
        assert info.truncated is False

    async def test_utf8_preserved_during_trim(self):
        mgr = _make_manager()
        # "café" = b'caf\xc3\xa9' (5 bytes), limit=4
        info = _make_info(byte_limit=4)
        mgr._terminals["t1"] = info

        chunks = [b"caf\xc3\xa9", b""]
        async def fake_read(n):
            return chunks.pop(0) if chunks else b""
        info.process.stdout = MagicMock()
        info.process.stdout.read = fake_read
        info.process.wait = AsyncMock(return_value=0)

        await mgr._read_terminal_output("t1")

        # excess=1, del [:1] → b'af\xc3\xa9' (4 bytes)
        # 'a' is ASCII start byte → no further skip
        assert info.output_buffer == bytearray(b"af\xc3\xa9")
        assert info.truncated is True

    async def test_exit_status_captured_on_reader_exception(self):
        mgr = _make_manager()
        info = _make_info(byte_limit=100)
        mgr._terminals["t1"] = info

        call_count = 0
        async def failing_read(n):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return b"partial"
            raise OSError("broken pipe")

        info.process.stdout = MagicMock()
        info.process.stdout.read = failing_read
        info.process.wait = AsyncMock(return_value=1)

        await mgr._read_terminal_output("t1")

        # Even with reader failure, exit status should be captured
        assert info.exit_code == 1


# ===================================================================
# Constants
# ===================================================================

class TestConstants:
    """Verify module-level constants are sensible."""

    async def test_default_byte_limit_is_10mb(self):
        assert DEFAULT_OUTPUT_BYTE_LIMIT == 10 * 1024 * 1024

    async def test_max_terminals_is_50(self):
        assert MAX_TERMINALS == 50

    async def test_denied_env_vars_include_ld_preload(self):
        assert "LD_PRELOAD" in _DENIED_ENV_VARS

    async def test_denied_env_vars_include_dyld(self):
        assert "DYLD_INSERT_LIBRARIES" in _DENIED_ENV_VARS
