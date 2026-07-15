"""
A fake ACP agent that advertises a session *mode* through one or both of ACP's
two channels, selected by a `--channel` CLI flag:

- `set_mode`      — the dedicated `session/set_mode` state (`modes` on the
                    session response), changed via `session/set_mode`.
- `config_option` — a config option with category `"mode"`, changed via
                    `session/set_config_option`.
- `both`          — advertises the mode through *both* channels at once, to
                    check the client de-duplicates it into a single control.

On every prompt it replies with its current mode as YAML, so a test can switch
the mode control, send a message, and assert the reply reflects the change —
verifying the mode round-trips through whichever channel the agent used.

Not part of the shipped package; see AGENTS.md and ../personas/*mode*_persona.py.
Method signatures match the installed agent-client-protocol.
"""

import argparse
import asyncio
from typing import Any

import yaml
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
from acp.schema import (
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    SessionMode,
    SessionModeState,
    SetSessionConfigOptionResponse,
    SetSessionModeResponse,
)

# The mode ids/labels the agent offers, and its default. Fixed so a test can
# assert the reply text deterministically.
MODE_VALUES = ["ask", "code", "architect"]
MODE_DEFAULT = "ask"
# The id used for the mode when advertised as a config option.
MODE_CONFIG_ID = "mode"


def _mode_config_option(current: str) -> SessionConfigOptionSelect:
    return SessionConfigOptionSelect(
        id=MODE_CONFIG_ID,
        name="Mode",
        type="select",
        category="mode",
        currentValue=current,
        options=[SessionConfigSelectOption(value=v, name=v.title()) for v in MODE_VALUES],
    )


def _mode_state(current: str) -> SessionModeState:
    return SessionModeState(
        currentModeId=current,
        availableModes=[SessionMode(id=v, name=v.title()) for v in MODE_VALUES],
    )


class ModeAgent(Agent):
    _conn: Client

    def __init__(self, channel: str) -> None:
        # "set_mode", "config_option", or "both".
        self._channel = channel
        self._mode = MODE_DEFAULT

    def on_connect(self, conn: Client) -> None:
        self._conn = conn

    async def initialize(
        self, protocol_version: int, **kwargs: Any
    ) -> InitializeResponse:
        return InitializeResponse(protocol_version=protocol_version)

    async def new_session(
        self, cwd: str, mcp_servers: Any = None, **kwargs: Any
    ) -> NewSessionResponse:
        modes = None
        config_options = None
        if self._channel in ("set_mode", "both"):
            modes = _mode_state(self._mode)
        if self._channel in ("config_option", "both"):
            config_options = [_mode_config_option(self._mode)]
        return NewSessionResponse(
            session_id="mode-session",
            modes=modes,
            config_options=config_options,
        )

    async def set_session_mode(
        self, session_id: str, mode_id: str, **kwargs: Any
    ) -> SetSessionModeResponse | None:
        self._mode = mode_id
        return SetSessionModeResponse()

    async def set_config_option(
        self, config_id: str, session_id: str, value: Any, **kwargs: Any
    ) -> SetSessionConfigOptionResponse | None:
        if config_id == MODE_CONFIG_ID:
            self._mode = value
        # Echo back the advertised option so the client stays in sync.
        return SetSessionConfigOptionResponse(
            configOptions=[_mode_config_option(self._mode)]
        )

    async def prompt(
        self, prompt: list, session_id: str, **kwargs: Any
    ) -> PromptResponse:
        body = yaml.safe_dump({"mode": self._mode}, default_flow_style=False)
        await self._conn.session_update(
            session_id=session_id,
            update=update_agent_message(text_block(f"```\n{body}```")),
        )
        return PromptResponse(stop_reason="end_turn")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--channel",
        choices=["set_mode", "config_option", "both"],
        default="set_mode",
    )
    args = parser.parse_args()
    asyncio.run(run_agent(ModeAgent(args.channel)))


if __name__ == "__main__":
    main()
