import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from jupyter_ai_acp_client.base_acp_persona import BaseAcpPersona


pytestmark = pytest.mark.asyncio


def completed(value):
    future = asyncio.get_running_loop().create_future()
    future.set_result(value)
    return future


@pytest.fixture
def persona_type():
    class Persona:
        _MAX_RESPAWNS = BaseAcpPersona._MAX_RESPAWNS
        _RESPAWN_WINDOW = BaseAcpPersona._RESPAWN_WINDOW
        _respawn_timestamps = []
        _stopping_subprocess = False
        get_agent_subprocess = BaseAcpPersona.get_agent_subprocess
        get_client = BaseAcpPersona.get_client
        get_session_response = BaseAcpPersona.get_session_response
        get_session_id = BaseAcpPersona.get_session_id
        _shutdown = BaseAcpPersona._shutdown

        def __init__(self):
            self.event_loop = asyncio.get_running_loop()
            self.id = "test"
            self.log = MagicMock()
            self.before_agent_subprocess = AsyncMock()
            self._init_agent_subprocess = AsyncMock(
                return_value=SimpleNamespace(returncode=None)
            )
            self._init_client = AsyncMock(return_value=MagicMock())
            self._get_existing_sessions = MagicMock(return_value={self.id: "old"})
            self._session_client_future = self.__class__._client_future
            self._client_session_future = completed("old response")

            async def create_session():
                self._get_existing_sessions.return_value = {self.id: "new"}
                return "new response"

            self._init_client_session = AsyncMock(side_effect=create_session)

    return Persona


@pytest_asyncio.fixture
async def persona(persona_type):
    persona_type._subprocess_future = completed(SimpleNamespace(returncode=1, pid=123))
    client = MagicMock()
    client.end_session = AsyncMock()
    client.list_sessions.return_value = []
    client.get_connection = AsyncMock(return_value=SimpleNamespace(close=AsyncMock()))
    persona_type._client_future = completed(client)
    persona_type._before_subprocess_future = completed(None)
    instance = persona_type()
    yield instance
    for future in (
        persona_type._subprocess_future,
        persona_type._client_future,
        persona_type._before_subprocess_future,
        instance._client_session_future,
    ):
        if future is not None and not future.done():
            future.cancel()
            await asyncio.gather(future, return_exceptions=True)


async def test_live_process_is_reused(persona):
    process = SimpleNamespace(returncode=None)
    persona.__class__._subprocess_future = completed(process)

    assert await persona.get_agent_subprocess() is process
    persona._init_agent_subprocess.assert_not_called()


async def test_dead_process_respawns_once_for_concurrent_callers(persona):
    processes = await asyncio.gather(
        persona.get_agent_subprocess(), persona.get_agent_subprocess()
    )

    assert processes[0] is processes[1]
    persona._init_agent_subprocess.assert_awaited_once()
    await persona.get_client()
    persona._init_client.assert_awaited_once()


async def test_shutdown_does_not_respawn(persona, monkeypatch):
    monkeypatch.setattr(
        "jupyter_ai_acp_client.base_acp_persona.os.getpgid",
        MagicMock(side_effect=ProcessLookupError),
    )

    await persona._shutdown()

    persona.before_agent_subprocess.assert_not_called()
    persona._init_agent_subprocess.assert_not_called()
    persona._init_client_session.assert_not_called()
    assert persona.__class__._subprocess_future is None
    assert persona.__class__._client_future is None


async def test_other_instance_cannot_respawn_during_teardown(persona):
    persona.__class__._stopping_subprocess = True

    with pytest.raises(RuntimeError, match="shutting down"):
        await persona.get_client()

    persona._init_agent_subprocess.assert_not_called()


async def test_crash_loop_is_bounded(persona):
    persona._init_agent_subprocess.return_value = SimpleNamespace(returncode=1)

    for _ in range(2):
        with pytest.raises(RuntimeError, match="repeatedly crashed"):
            await persona.get_agent_subprocess()

    assert persona._init_agent_subprocess.await_count == persona._MAX_RESPAWNS
    assert persona.log.error.call_count == 2


async def test_crash_limit_expires(persona, monkeypatch):
    persona.__class__._respawn_timestamps = [100.0] * persona._MAX_RESPAWNS
    monkeypatch.setattr(
        "jupyter_ai_acp_client.base_acp_persona.time.monotonic", lambda: 160.0
    )

    assert (await persona.get_agent_subprocess()).returncode is None
    assert persona.__class__._respawn_timestamps == [160.0]


async def test_crash_limit_is_per_persona_class(persona, persona_type):
    class OtherPersona(persona_type):
        pass

    other = OtherPersona()
    other.__class__._respawn_timestamps = [float("inf")] * other._MAX_RESPAWNS

    assert (await persona.get_agent_subprocess()).returncode is None
    assert len(persona.__class__._respawn_timestamps) == 1
    assert len(other.__class__._respawn_timestamps) == other._MAX_RESPAWNS


async def test_each_instance_recovers_its_session(persona, persona_type):
    other = persona_type()

    assert await persona.get_session_id() == "new"
    assert await other.get_session_response() == "new response"
    assert await other.get_session_id() == "new"
    persona._init_client_session.assert_awaited_once()
    other._init_client_session.assert_awaited_once()
    persona._init_agent_subprocess.assert_awaited_once()
    other._init_agent_subprocess.assert_not_called()


async def test_concurrent_session_callers_share_recovery(persona):
    assert await asyncio.gather(
        persona.get_session_id(), persona.get_session_id()
    ) == ["new", "new"]
    persona._init_client_session.assert_awaited_once()


async def test_unrelated_session_failure_is_not_retried(persona):
    persona.__class__._subprocess_future = completed(SimpleNamespace(returncode=None))
    persona._client_session_future = asyncio.get_running_loop().create_future()
    persona._client_session_future.set_exception(ValueError("invalid metadata"))

    with pytest.raises(ValueError, match="invalid metadata"):
        await persona.get_session_id()

    persona._init_client_session.assert_not_called()


async def test_missing_session_metadata_is_not_retried(persona):
    persona.__class__._subprocess_future = completed(SimpleNamespace(returncode=None))
    persona._get_existing_sessions.return_value = {}

    with pytest.raises(AssertionError):
        await persona.get_session_id()

    persona._init_client_session.assert_not_called()


async def test_initial_session_tracks_recovered_client(persona):
    client = persona._init_client.return_value
    client.get_agent_capabilities = AsyncMock(
        return_value=SimpleNamespace(load_session=False)
    )
    persona._create_session = AsyncMock(return_value="initial response")
    persona._client_session_future = asyncio.create_task(
        BaseAcpPersona._init_client_session(persona)
    )

    assert await persona._client_session_future == "initial response"
    assert await persona.get_session_response() == "initial response"
    persona._create_session.assert_awaited_once_with(client)
    persona._init_client_session.assert_not_called()


async def test_pending_stale_session_is_cancelled(persona):
    started = asyncio.Event()

    async def old_session():
        started.set()
        await asyncio.Event().wait()

    old_future = asyncio.create_task(old_session())
    persona._client_session_future = old_future
    await started.wait()

    assert await persona.get_session_id() == "new"
    assert old_future.cancelled()
    persona._init_client_session.assert_awaited_once()
