"""
A fake ACP agent that echoes its current session config options as YAML.

It advertises two select config options (`model` and `effort_level`) with sane
defaults, honors `set_config_option` to change them, and on every prompt replies
with the current config as a YAML dict inside a Markdown code block, e.g.:

    ```
    config_options:
      model: 'claude-haiku-45'
      effort_level: 'medium'
    ```

This lets an E2E test change a control in the toolbar, send a message, and assert
the reply reflects the change — verifying the full set-config round trip.

See ../personas/echo_persona.py for the wrapping persona, and AGENTS.md for how
fake agents work. Method signatures match the installed agent-client-protocol.
"""

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
    SetSessionConfigOptionResponse,
)

# Config option id -> (default value, [allowed values]). The defaults match the
# example in the issue.
CONFIG_DEFAULTS = {
    "model": ("claude-haiku-45", ["claude-haiku-45", "claude-opus-48", "claude-fable-5"]),
    "effort_level": ("medium", ["low", "medium", "high"]),
}


def _option(config_id: str, current: str, values: list[str]) -> SessionConfigOptionSelect:
    return SessionConfigOptionSelect(
        id=config_id,
        name=config_id.replace("_", " ").title(),
        type="select",
        currentValue=current,
        options=[
            SessionConfigSelectOption(value=v, name=v) for v in values
        ],
    )


class EchoConfigAgent(Agent):
    _conn: Client

    def __init__(self) -> None:
        # Current value of each config option, seeded with the defaults.
        self._config: dict[str, str] = {
            cid: default for cid, (default, _values) in CONFIG_DEFAULTS.items()
        }

    def on_connect(self, conn: Client) -> None:
        self._conn = conn

    async def initialize(
        self, protocol_version: int, **kwargs: Any
    ) -> InitializeResponse:
        return InitializeResponse(protocol_version=protocol_version)

    def _options(self) -> list[SessionConfigOptionSelect]:
        return [
            _option(cid, self._config[cid], values)
            for cid, (_default, values) in CONFIG_DEFAULTS.items()
        ]

    async def new_session(
        self, cwd: str, mcp_servers: Any = None, **kwargs: Any
    ) -> NewSessionResponse:
        return NewSessionResponse(
            session_id="echo-config-session",
            config_options=self._options(),
        )

    async def set_config_option(
        self, config_id: str, session_id: str, value: Any, **kwargs: Any
    ) -> SetSessionConfigOptionResponse | None:
        if config_id in self._config:
            self._config[config_id] = value
        # Echo back the full, updated option set (the client refetches after a
        # change, and this keeps the advertised state in sync).
        return SetSessionConfigOptionResponse(configOptions=self._options())

    async def prompt(
        self, prompt: list, session_id: str, **kwargs: Any
    ) -> PromptResponse:
        body = yaml.safe_dump(
            {"config_options": dict(self._config)},
            default_flow_style=False,
            sort_keys=False,
        )
        reply = f"```\n{body}```"
        await self._conn.session_update(
            session_id=session_id,
            update=update_agent_message(text_block(reply)),
        )
        return PromptResponse(stop_reason="end_turn")


def main() -> None:
    asyncio.run(run_agent(EchoConfigAgent()))


if __name__ == "__main__":
    main()
