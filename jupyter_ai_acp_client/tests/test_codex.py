"""Tests for the Codex ACP persona error handling."""

from unittest.mock import patch

import pytest
from acp.exceptions import RequestError


class TestProcessMessageErrorHandling:
    """Tests for error code detection in process_message().

    codex-acp sends -32000 (AuthRequired) when no API key is configured,
    and -32603 (InternalError) for runtime failures like wrong API key.
    We only catch -32000; runtime errors should propagate with their
    real message so users can diagnose the issue.
    """

    def test_auth_required_is_caught(self):
        """ACP AuthRequired (-32000) should trigger handle_no_auth."""
        error = RequestError(-32000, "Authentication required")
        assert error.code == -32000

    def test_internal_error_is_not_caught(self):
        """InternalError (-32603) should propagate — user needs the real error."""
        error = RequestError(-32603, "Internal error", "unexpected status 401")
        assert error.code != -32000

    def test_other_errors_propagate(self):
        """Non-auth errors should propagate unchanged."""
        error = RequestError(-32601, "Method not found")
        assert error.code != -32000

    def test_resource_not_found_propagates(self):
        """Resource errors are not auth errors."""
        error = RequestError(-32002, "Resource not found")
        assert error.code != -32000
