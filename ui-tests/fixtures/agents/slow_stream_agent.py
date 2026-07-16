"""
A fake ACP agent that streams a long reply slowly — one chunk every ~200ms for
up to ~60s — so a test has a comfortable window to interrupt it mid-stream.

It counts up ("1 2 3 ...") one number per chunk. Between chunks it checks a
cancel flag set by ACP's `session/cancel` notification (delivered to `cancel()`),
so when the client cancels, the loop stops promptly and no further chunks land —
letting a test assert the message stops growing and the persona stops writing.

Not part of the shipped package; see AGENTS.md. Method signatures match the
installed agent-client-protocol.
"""

import asyncio
from typing import Any

from acp import (
    Agent,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    run_agent,
    text_block,
    update_agent_message,
)
from acp.interfaces import Client

# Stream cadence: total ~= CHUNKS * DELAY seconds (300 * 0.2s = 60s).
CHUNKS = 300
DELAY_S = 0.2


class SlowStreamAgent(Agent):
    _conn: Client

    def __init__(self) -> None:
        # Session IDs for which a cancel has been requested.
        self._cancelled: set[str] = set()

    def on_connect(self, conn: Client) -> None:
        self._conn = conn

    async def initialize(
        self, protocol_version: int, **kwargs: Any
    ) -> InitializeResponse:
        return InitializeResponse(protocol_version=protocol_version)

    async def new_session(
        self, cwd: str, mcp_servers: Any = None, **kwargs: Any
    ) -> NewSessionResponse:
        return NewSessionResponse(session_id="slow-stream-session")

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        # ACP delivers session/cancel here; the prompt loop checks this flag.
        self._cancelled.add(session_id)

    async def prompt(
        self, prompt: list, session_id: str, **kwargs: Any
    ) -> PromptResponse:
        self._cancelled.discard(session_id)
        for i in range(1, CHUNKS + 1):
            if session_id in self._cancelled:
                return PromptResponse(stop_reason="cancelled")
            await self._conn.session_update(
                session_id=session_id,
                update=update_agent_message(text_block(f"{i} ")),
            )
            await asyncio.sleep(DELAY_S)
        return PromptResponse(stop_reason="end_turn")


def main() -> None:
    asyncio.run(run_agent(SlowStreamAgent()))


if __name__ == "__main__":
    main()
