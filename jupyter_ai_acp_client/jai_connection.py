"""
`ClientSideConnection` subclass exposing the SDK connection's raw-message
surfaces as public API.

Some agents still serve JSON-RPC surfaces that predate ACP v1 and are gone
from the typed SDK models: kiro-cli attaches a `models` field to its
`session/new` response and accepts a `session/set_model` request. The typed
response models discard unknown fields, so those surfaces are only reachable
at the raw message layer. The SDK `Connection` (held by `ClientSideConnection`
as the protected `_conn`) already provides a raw-message observer and a
generic request sender; this subclass re-surfaces the two so `JaiAcpClient`
can read and speak the legacy surfaces without touching the typed flow.
"""

from typing import Any, Callable

from acp.core import ClientSideConnection


class JaiClientSideConnection(ClientSideConnection):
    """A `ClientSideConnection` with raw-message access."""

    def add_raw_observer(self, observer: Callable[[Any], None]) -> None:
        """
        Register a callback invoked with a `StreamEvent` for every JSON-RPC
        message on this connection: incoming messages before they are
        dispatched, outgoing messages after they are sent. The event carries
        `.direction` and the parsed `.message` dict.
        """
        self._conn.add_observer(observer)

    async def send_raw_request(self, method: str, params: dict) -> Any:
        """
        Send a JSON-RPC request outside the typed schema and return the raw
        result. Raises `acp.exceptions.RequestError` if the agent answers with
        an error.
        """
        return await self._conn.send_request(method, params)
