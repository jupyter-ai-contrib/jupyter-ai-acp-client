"""
Permission request management for the ACP tool call approval flow.

This module provides the PermissionManager class that tracks pending permission
requests using asyncio Futures, keyed by (session_id, tool_call_id).
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PendingRequest:
    """A pending permission request with its Future and agent-provided options."""
    future: asyncio.Future[str]
    options: list[dict] = field(default_factory=list)


class PermissionManager:
    """
    Manages pending permission requests using asyncio Futures.

    Each request is keyed by (session_id, tool_call_id) and holds a Future
    that resolves with the selected option_id when the user clicks a button.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._pending: dict[tuple[str, str], PendingRequest] = {}
        self._loop = loop

    def create_request(
        self,
        session_id: str,
        tool_call_id: str,
        options: list[dict] | None = None,
    ) -> asyncio.Future[str]:
        """
        Create a pending permission request.
        """
        key = (session_id, tool_call_id)
        future: asyncio.Future[str] = self._loop.create_future()
        self._pending[key] = PendingRequest(future=future, options=options or [])
        return future

    def resolve(self, session_id: str, tool_call_id: str, option_id: str) -> bool:
        """
        Resolve a pending permission request with the user's selected option_id.

        Returns True if the request was found and resolved, False if the
        key is unknown or the Future is already done.
        """
        key = (session_id, tool_call_id)
        req = self._pending.get(key)
        if req is None or req.future.done():
            return False
        req.future.set_result(option_id)
        return True

    def cleanup(self, session_id: str, tool_call_id: str) -> None:
        """Remove a pending permission request."""
        self._pending.pop((session_id, tool_call_id), None)

    def has_pending(self, session_id: str) -> bool:
        """Check if any pending permission exists for a session."""
        return any(sid == session_id for sid, _ in self._pending)

    def reject_all_pending(self, session_id: str) -> int:
        """
        Auto-reject all pending permission requests for a session.
        """
        rejected = 0
        keys_to_remove = [key for key in self._pending if key[0] == session_id]
        for key in keys_to_remove:
            req = self._pending.pop(key, None)
            if req is not None and not req.future.done():
                reject_id = self._find_reject_option_id(req.options)
                req.future.set_result(reject_id)
                rejected += 1
        return rejected

    @staticmethod
    def _find_reject_option_id(options: list[dict]) -> str:
        """
        Find the reject option_id from the options list.
        """
        for opt in options:
            desc = opt.get("description", "")
            if "reject" in desc:
                return opt["option_id"]
        return "reject_once"
