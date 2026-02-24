"""
Permission request management for the ACP tool call approval flow.

This module provides the PermissionManager class that tracks pending permission
requests using asyncio Futures, keyed by (session_id, tool_call_id).
"""

import asyncio
from typing import Optional


PERMISSION_OPTIONS = [
    {"option_id": "allow_once", "title": "Allow Once", "description": ""},
    {"option_id": "allow_always", "title": "Allow Always", "description": ""},
    {"option_id": "reject_once", "title": "Reject Once", "description": ""},
]


class PermissionManager:
    """
    Manages pending permission requests using asyncio Futures.

    Each request is keyed by (session_id, tool_call_id) and holds a Future
    that resolves with the selected option_id when the user clicks a button.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._pending: dict[tuple[str, str], asyncio.Future[str]] = {}
        self._loop = loop

    def create_request(
        self,
        session_id: str,
        tool_call_id: str,
    ) -> asyncio.Future[str]:
        """
        Create a pending permission request.

        Returns an asyncio.Future that resolves with the selected option_id
        when resolve() is called.
        """
        key = (session_id, tool_call_id)
        future: asyncio.Future[str] = self._loop.create_future()
        self._pending[key] = future
        return future

    def resolve(self, session_id: str, tool_call_id: str, option_id: str) -> bool:
        """
        Resolve a pending permission request with the user's selected option_id.

        Returns True if the request was found and resolved, False if the
        key is unknown or the Future is already done.
        """
        key = (session_id, tool_call_id)
        future = self._pending.get(key)
        if future is None or future.done():
            return False
        future.set_result(option_id)
        return True

    def cleanup(self, session_id: str, tool_call_id: str) -> None:
        """Remove a pending permission request."""
        self._pending.pop((session_id, tool_call_id), None)

    def has_pending(self, session_id: str) -> bool:
        """Check if any pending permission exists for a session."""
        return any(sid == session_id for sid, _ in self._pending)
