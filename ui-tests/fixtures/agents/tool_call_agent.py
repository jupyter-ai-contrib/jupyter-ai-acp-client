"""
A fake ACP agent that requests permission for one file edit on every prompt,
then replies "allowed" or "rejected" from the user's decision.
"""

import asyncio
import os
import uuid
from typing import Any

from acp import (
    Agent,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    run_agent,
    start_tool_call,
    text_block,
    tool_diff_content,
    update_agent_message,
    update_tool_call,
)
from acp.interfaces import Client
from acp.schema import PermissionOption, ToolCallLocation, ToolCallUpdate

TOOL_CALL_ID = "edit-1"
TOOL_CALL_TITLE = "Edit target file"
TARGET_FILE = "tool-call-target.txt"
ALLOW_OPTION_ID = "allow"
REJECT_OPTION_ID = "reject"


class ToolCallAgent(Agent):
    _conn: Client

    def __init__(self) -> None:
        self._cwds: dict[str, str] = {}

    def on_connect(self, conn: Client) -> None:
        self._conn = conn

    async def initialize(
        self, protocol_version: int, **kwargs: Any
    ) -> InitializeResponse:
        return InitializeResponse(protocol_version=protocol_version)

    async def new_session(
        self, cwd: str, mcp_servers: Any = None, **kwargs: Any
    ) -> NewSessionResponse:
        # One agent process serves all chats of the persona, and permission
        # requests are keyed by session, so each chat needs its own session ID.
        session_id = uuid.uuid4().hex
        # The frontend compares paths with the resolved server root.
        self._cwds[session_id] = os.path.realpath(cwd)
        return NewSessionResponse(session_id=session_id)

    async def prompt(
        self, prompt: list, session_id: str, **kwargs: Any
    ) -> PromptResponse:
        path = os.path.join(self._cwds[session_id], TARGET_FILE)
        content = [tool_diff_content(path, "new line\n", "old line\n")]
        await self._conn.session_update(
            session_id=session_id,
            update=start_tool_call(
                TOOL_CALL_ID,
                TOOL_CALL_TITLE,
                kind="edit",
                status="pending",
                content=content,
                locations=[ToolCallLocation(path=path)],
            ),
        )

        response = await self._conn.request_permission(
            session_id=session_id,
            tool_call=ToolCallUpdate(
                tool_call_id=TOOL_CALL_ID,
                title=TOOL_CALL_TITLE,
                kind="edit",
                content=content,
            ),
            options=[
                PermissionOption(
                    option_id=ALLOW_OPTION_ID, name="Allow", kind="allow_once"
                ),
                PermissionOption(
                    option_id=REJECT_OPTION_ID, name="Reject", kind="reject_once"
                ),
            ],
        )
        allowed = getattr(response.outcome, "option_id", None) == ALLOW_OPTION_ID

        await self._conn.session_update(
            session_id=session_id,
            update=update_tool_call(
                TOOL_CALL_ID, status="completed" if allowed else "failed"
            ),
        )
        await self._conn.session_update(
            session_id=session_id,
            update=update_agent_message(
                text_block("allowed" if allowed else "rejected")
            ),
        )
        return PromptResponse(stop_reason="end_turn")


def main() -> None:
    asyncio.run(run_agent(ToolCallAgent()))


if __name__ == "__main__":
    main()
