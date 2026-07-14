"""
A minimal fake ACP agent for E2E tests: it replies "hello" to every prompt.

An ACP agent is just an executable that speaks the Agent Client Protocol over
stdio. `BaseAcpPersona` spawns whatever `executable` it is given and talks ACP to
it, so a fake agent is simply a Python script implementing `acp.Agent` and served
with `acp.run_agent` — no network, no real CLI, fully deterministic. A fixture
persona (see ../personas) points its `executable` at this file.

Based on the SDK's `examples/echo_agent.py`, trimmed to a fixed reply. The method
signatures here match the installed `agent-client-protocol` (see the pin in
jupyter-ai-acp-client's pyproject), which differs from the SDK's `main` examples —
keep them in sync with the pinned SDK, not the repo's latest.
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

REPLY = "hello"


class HelloAgent(Agent):
    _conn: Client

    def on_connect(self, conn: Client) -> None:
        self._conn = conn

    async def initialize(
        self, protocol_version: int, **kwargs: Any
    ) -> InitializeResponse:
        return InitializeResponse(protocol_version=protocol_version)

    async def new_session(
        self, cwd: str, mcp_servers: Any = None, **kwargs: Any
    ) -> NewSessionResponse:
        # A fixed session id is fine: one agent process serves one test chat.
        return NewSessionResponse(session_id="hello-session")

    async def prompt(
        self, prompt: list, session_id: str, **kwargs: Any
    ) -> PromptResponse:
        # Stream a single agent message chunk, then end the turn. The client's
        # session_update handler appends this text to the chat message.
        await self._conn.session_update(
            session_id=session_id,
            update=update_agent_message(text_block(REPLY)),
        )
        return PromptResponse(stop_reason="end_turn")


def main() -> None:
    asyncio.run(run_agent(HelloAgent()))


if __name__ == "__main__":
    main()
