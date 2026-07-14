"""
A minimal fake ACP agent for E2E tests: it replies with a fixed string (default
"hello") to every prompt.

An ACP agent is just an executable that speaks the Agent Client Protocol over
stdio. `BaseAcpPersona` spawns whatever `executable` it is given and talks ACP to
it, so a fake agent is simply a Python script implementing `acp.Agent` and served
with `acp.run_agent` — no network, no real CLI, fully deterministic. A fixture
persona (see ../personas) points its `executable` at this file.

The reply text is set by an optional `--reply` CLI flag, so one agent script can
back several personas that each answer distinctly — useful for asserting that a
message routed to the *right* persona (see persona-control.spec.ts). Passing the
reply on the `executable` is the same pattern usage_agent.py uses for `--mode`.

Based on the SDK's `examples/echo_agent.py`, trimmed to a fixed reply. The method
signatures here match the installed `agent-client-protocol` (see the pin in
jupyter-ai-acp-client's pyproject), which differs from the SDK's `main` examples —
keep them in sync with the pinned SDK, not the repo's latest.
"""

import argparse
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

DEFAULT_REPLY = "hello"


class HelloAgent(Agent):
    _conn: Client

    def __init__(self, reply: str = DEFAULT_REPLY) -> None:
        self._reply = reply

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
            update=update_agent_message(text_block(self._reply)),
        )
        return PromptResponse(stop_reason="end_turn")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fake ACP agent with a fixed reply.")
    parser.add_argument(
        "--reply",
        default=DEFAULT_REPLY,
        help="The text the agent replies with to every prompt.",
    )
    args = parser.parse_args()
    asyncio.run(run_agent(HelloAgent(args.reply)))


if __name__ == "__main__":
    main()
