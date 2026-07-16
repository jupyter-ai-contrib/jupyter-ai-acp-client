"""
A fake ACP agent for exercising the ways an agent can report usage.

ACP v1 exposes usage through two distinct channels:

- **session/usage** — a `usage_update` session update carrying the context
  window fill (`used`/`size`) and an optional cumulative `cost`. This is the
  standard mechanism: https://agentclientprotocol.com/rfds/session-usage
- **end-turn token usage** — a `usage` object on the prompt response carrying
  cumulative session token counts (input/output/total, …). This is currently
  **experimental**: https://agentclientprotocol.com/rfds/end-turn-token-usage

Some agents use neither: kiro-cli (v2 engine, 2.12.1) streams a bare context
percentage in a `_kiro.dev/metadata` extension notification, which the client
records through its `ext_notification` hook.

The client surfaces these differently in the toolbar's usage chip (context fill
renders a ring + percent; session tokens render a token count and a breakdown
popover), so we need agents that report one, the other, both, or the vendor
extension.

Rather than near-identical agents, this one takes a `--mode` selecting what it
reports on each turn:

    usage_agent.py --mode session    # session/usage only (context [+ cost])
    usage_agent.py --mode response   # response.usage only (session tokens)
    usage_agent.py --mode both       # both channels
    usage_agent.py --mode kiro       # _kiro.dev/metadata percentage only

A fixture persona wraps each mode (see ../personas/*-usage_persona.py). Method
signatures match the installed agent-client-protocol (see the pin in
jupyter-ai-acp-client's pyproject), not the SDK's `main` examples.
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
from acp.schema import Cost, Usage, UsageUpdate

# Fixed numbers per mode, chosen so each renders a distinct, easily-asserted UI:
#   session  -> 1200 / 4000 = 30% context fill, plus a cost
#   both     -> 2000 / 8000 = 25% context fill, plus a cost
#   kiro     -> a bare 42% context fill, no token counts, no cost
# Token totals are cumulative session counts, reported on the prompt response.
CONTEXT = {
    "session": UsageUpdate(
        session_update="usage_update",
        used=1200,
        size=4000,
        cost=Cost(amount=0.42, currency="USD"),
    ),
    "both": UsageUpdate(
        session_update="usage_update",
        used=2000,
        size=8000,
        cost=Cost(amount=1.50, currency="USD"),
    ),
}
TOKENS = Usage(input_tokens=1000, output_tokens=500, total_tokens=1500)
KIRO_PERCENT = 42.0

REPLY = "usage reported"


class UsageAgent(Agent):
    """Reports usage via session/usage, response.usage, both, or the kiro
    vendor extension, per `mode`."""

    _conn: Client

    def __init__(self, mode: str) -> None:
        self._mode = mode

    def on_connect(self, conn: Client) -> None:
        self._conn = conn

    async def initialize(
        self, protocol_version: int, **kwargs: Any
    ) -> InitializeResponse:
        return InitializeResponse(protocol_version=protocol_version)

    async def new_session(
        self, cwd: str, mcp_servers: Any = None, **kwargs: Any
    ) -> NewSessionResponse:
        return NewSessionResponse(session_id=f"{self._mode}-usage-session")

    async def prompt(
        self, prompt: list, session_id: str, **kwargs: Any
    ) -> PromptResponse:
        # A short reply so the chat renders a message; usage rides alongside it.
        await self._conn.session_update(
            session_id=session_id,
            update=update_agent_message(text_block(REPLY)),
        )

        # session/usage channel: a usage_update carrying context fill (+ cost).
        if self._mode in CONTEXT:
            await self._conn.session_update(
                session_id=session_id,
                update=CONTEXT[self._mode],
            )

        # vendor extension channel: kiro-cli streams a bare percentage in a
        # `_kiro.dev/metadata` notification (the SDK adds the `_` prefix).
        if self._mode == "kiro":
            await self._conn.ext_notification(
                "kiro.dev/metadata",
                {"sessionId": session_id, "contextUsagePercentage": KIRO_PERCENT},
            )

        # end-turn token usage channel: cumulative session tokens on the response.
        usage = TOKENS if self._mode in ("response", "both") else None
        return PromptResponse(stop_reason="end_turn", usage=usage)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fake ACP usage-reporting agent.")
    parser.add_argument(
        "--mode",
        choices=["session", "response", "both", "kiro"],
        required=True,
        help="Which usage channel(s) to report on each turn.",
    )
    args = parser.parse_args()
    asyncio.run(run_agent(UsageAgent(args.mode)))


if __name__ == "__main__":
    main()
