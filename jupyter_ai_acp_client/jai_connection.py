"""
`ClientSideConnection` subclass that also reads legacy fields off the raw
session responses.

Some agents still serve JSON-RPC surfaces that predate ACP v1 and are gone
from the typed SDK models: kiro-cli attaches a `models` field to its
`session/new` response and accepts a `session/set_model` request. The typed
response models discard unknown fields, so those surfaces are only reachable
at the raw message layer.

`new_session` and `load_session` here mirror the SDK's thin typed wrappers
(send serialized params over the inner `Connection`, validate the response)
with one addition: the raw response dict is checked for the legacy `models`
field before validation, and matches are reported through the callback set
with `set_legacy_models_listener`. The capture is scoped to these two calls on
purpose: the SDK's raw-message observer hook (`Connection.add_observer`)
deep-copies every JSON-RPC message on the connection while any observer is
registered, a per-message cost this client avoids (its sessions move
multi-megabyte embedded resources).

`send_raw_request` exposes the inner `Connection`'s generic sender for
requests outside the typed schema (`session/set_model`).

The SDK marks `ClientSideConnection` `@final`; that is not enforced at
runtime, and this subclass overrides it deliberately, extending the same
protected `_conn` surface the SDK wrappers themselves use.
"""

from typing import Any, Callable, Optional

from acp.client.connection import ClientSideConnection
from acp.meta import AGENT_METHODS
from acp.schema import (
    LoadSessionRequest,
    LoadSessionResponse,
    NewSessionRequest,
    NewSessionResponse,
)
from acp.utils import serialize_params, validate_model, validate_model_from_dict

LegacyModelsListener = Callable[[str, dict], None]


class JaiClientSideConnection(ClientSideConnection):
    """A `ClientSideConnection` that captures legacy session fields."""

    _legacy_models_listener: Optional[LegacyModelsListener] = None

    def set_legacy_models_listener(self, listener: LegacyModelsListener) -> None:
        """
        Set the callback invoked as `listener(session_id, models)` when a
        session response carries the legacy `models` field.
        """
        self._legacy_models_listener = listener

    def _capture_legacy_models(
        self, response: Any, fallback_session_id: Optional[str] = None
    ) -> None:
        """
        Report the legacy `models` field of a raw session response, if present.
        `session/new` responses carry their own `sessionId`; `session/load`
        responses do not, so the caller passes the requested session ID as the
        fallback key.
        """
        if self._legacy_models_listener is None or not isinstance(response, dict):
            return
        session_id = response.get("sessionId") or fallback_session_id
        models = response.get("models")
        if isinstance(session_id, str) and isinstance(models, dict):
            self._legacy_models_listener(session_id, models)

    async def new_session(
        self,
        cwd: str,
        additional_directories: Optional[list] = None,
        mcp_servers: Optional[list] = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        response = await self._conn.send_request(
            AGENT_METHODS["session_new"],
            serialize_params(
                NewSessionRequest(
                    cwd=cwd,
                    additional_directories=additional_directories,
                    mcp_servers=mcp_servers or [],
                    field_meta=kwargs or None,
                )
            ),
        )
        self._capture_legacy_models(response)
        return validate_model(response, NewSessionResponse)

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: Optional[list] = None,
        additional_directories: Optional[list] = None,
        **kwargs: Any,
    ) -> LoadSessionResponse:
        response = await self._conn.send_request(
            AGENT_METHODS["session_load"],
            serialize_params(
                LoadSessionRequest(
                    cwd=cwd,
                    additional_directories=additional_directories,
                    mcp_servers=mcp_servers or [],
                    session_id=session_id,
                    field_meta=kwargs or None,
                )
            ),
        )
        self._capture_legacy_models(response, fallback_session_id=session_id)
        return validate_model_from_dict(response, LoadSessionResponse)

    async def send_raw_request(self, method: str, params: dict) -> Any:
        """
        Send a JSON-RPC request outside the typed schema and return the raw
        result. Raises `acp.exceptions.RequestError` if the agent answers with
        an error.
        """
        return await self._conn.send_request(method, params)
