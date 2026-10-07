"""
A fake ACP agent that advertises a fixed set of slash commands, then echoes back
whatever prompt it receives.

ACP agents advertise slash commands with an `available_commands_update` session
notification (there is no field for them on the session response). This agent
sends that notification shortly after the session is created — from a background
task, so the client has finished registering the session by the time the update
arrives — announcing a deterministic command list the client surfaces as chat
input completions when the user types `/`.

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
from acp.schema import AvailableCommand, AvailableCommandsUpdate

# The commands the agent advertises, fixed so the test can assert them exactly.
COMMANDS = [
    ("compact", "Compact the conversation context"),
    ("clear", "Clear the conversation history"),
    ("help", "Show available commands"),
]


class SlashCommandsAgent(Agent):
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
        session_id = "slash-commands-session"
        # Announce the commands after new_session returns, so the client has
        # mapped this session to its persona before the notification lands.
        asyncio.create_task(self._announce_commands(session_id))
        return NewSessionResponse(session_id=session_id)

    async def _announce_commands(self, session_id: str) -> None:
        await asyncio.sleep(0.1)
        await self._conn.session_update(
            session_id=session_id,
            update=AvailableCommandsUpdate(
                sessionUpdate="available_commands_update",
                availableCommands=[
                    AvailableCommand(name=name, description=description)
                    for name, description in COMMANDS
                ],
            ),
        )

    async def prompt(
        self, prompt: list, session_id: str, **kwargs: Any
    ) -> PromptResponse:
        await self._conn.session_update(
            session_id=session_id,
            update=update_agent_message(text_block("ok")),
        )
        return PromptResponse(stop_reason="end_turn")


def main() -> None:
    asyncio.run(run_agent(SlashCommandsAgent()))


if __name__ == "__main__":
    main()
