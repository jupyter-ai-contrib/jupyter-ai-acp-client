"""
A fake ACP agent that advertises *two* config options in the same category, to
exercise the client's tie-breaking rule.

ACP says that when several config options share a category, the client resolves
the tie by array order — the earliest wins the prominent slot (the Model picker,
the Mode selector), and later same-category options are shown as ordinary
settings (https://agentclientprotocol.com/protocol/v1/session-config-options
#option-categories).

This agent advertises, in order: two `category: "model"` options
(`model` then `model_alt`) and two `category: "mode"` options (`mode` then
`mode_alt`). So the client should surface `model`/`mode` as the prominent
Model/Mode controls and `model_alt`/`mode_alt` as general settings. Each option
has distinct choices so the rendered controls are unambiguous. On every prompt
it echoes its current config as YAML.

Not part of the shipped package; see AGENTS.md. Method signatures match the
installed agent-client-protocol.
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

# id -> (category, display name, default value, [allowed values]). Order matters:
# the first option of each category wins the prominent slot.
CONFIG_SPECS = [
    ("model", "model", "Model", "opus", ["opus", "haiku"]),
    ("model_alt", "model", "Backup Model", "gpt", ["gpt", "gemini"]),
    ("mode", "mode", "Mode", "ask", ["ask", "code"]),
    ("mode_alt", "mode", "Backup Mode", "fast", ["fast", "slow"]),
]


def _option(spec) -> SessionConfigOptionSelect:
    config_id, category, name, current, values = spec
    return SessionConfigOptionSelect(
        id=config_id,
        name=name,
        type="select",
        category=category,
        currentValue=current,
        options=[SessionConfigSelectOption(value=v, name=v.title()) for v in values],
    )


class DuplicateGroupsAgent(Agent):
    _conn: Client

    def __init__(self) -> None:
        self._config = {spec[0]: spec[3] for spec in CONFIG_SPECS}

    def on_connect(self, conn: Client) -> None:
        self._conn = conn

    async def initialize(
        self, protocol_version: int, **kwargs: Any
    ) -> InitializeResponse:
        return InitializeResponse(protocol_version=protocol_version)

    def _options(self) -> list[SessionConfigOptionSelect]:
        return [
            _option((cid, cat, name, self._config[cid], values))
            for (cid, cat, name, _default, values) in CONFIG_SPECS
        ]

    async def new_session(
        self, cwd: str, mcp_servers: Any = None, **kwargs: Any
    ) -> NewSessionResponse:
        return NewSessionResponse(
            session_id="duplicate-groups-session",
            config_options=self._options(),
        )

    async def set_config_option(
        self, config_id: str, session_id: str, value: Any, **kwargs: Any
    ) -> SetSessionConfigOptionResponse | None:
        if config_id in self._config:
            self._config[config_id] = value
        return SetSessionConfigOptionResponse(configOptions=self._options())

    async def prompt(
        self, prompt: list, session_id: str, **kwargs: Any
    ) -> PromptResponse:
        body = yaml.safe_dump(
            {"config": dict(self._config)}, default_flow_style=False, sort_keys=False
        )
        await self._conn.session_update(
            session_id=session_id,
            update=update_agent_message(text_block(f"```\n{body}```")),
        )
        return PromptResponse(stop_reason="end_turn")


def main() -> None:
    asyncio.run(run_agent(DuplicateGroupsAgent()))


if __name__ == "__main__":
    main()
