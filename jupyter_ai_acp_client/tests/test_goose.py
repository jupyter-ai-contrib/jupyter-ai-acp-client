"""Tests for the Goose ACP persona helper functions."""

from unittest.mock import MagicMock, patch

import pytest
from acp.exceptions import RequestError
from jupyter_ai_persona_manager import PersonaRequirementsUnmet

# goose.py has a module-level guard that raises PersonaRequirementsUnmet
# when the goose CLI is not installed. Mock the guard so we can import
# helpers in CI without the CLI.
_mock_run = MagicMock()
_mock_run.returncode = 0
_mock_run.stdout = "goose 1.28.0"
_mock_run.stderr = ""

with patch("shutil.which", return_value="/usr/bin/goose"), \
     patch("subprocess.run", return_value=_mock_run):
    from jupyter_ai_acp_client.acp_personas.goose import (
        _check_goose,
        _is_setup_error,
    )


class TestIsSetupError:
    """Tests for _is_setup_error() — checks error.data for provider config issues."""

    def test_provider_not_configured(self):
        error = RequestError(-32603, "Internal error", "Failed to set provider: No default provider")
        assert _is_setup_error(error) is True

    def test_no_provider(self):
        error = RequestError(-32603, "Internal error", "No provider configured")
        assert _is_setup_error(error) is True

    def test_unrelated_internal_error(self):
        error = RequestError(-32603, "Internal error", "Session not found")
        assert _is_setup_error(error) is False

    def test_auth_error_code_not_setup(self):
        """ACP auth errors (-32000) are not setup errors."""
        error = RequestError(-32000, "Authentication required")
        assert _is_setup_error(error) is False

    def test_plain_exception(self):
        assert _is_setup_error(Exception("provider error")) is False

    def test_no_data(self):
        error = RequestError(-32603, "Internal error")
        assert _is_setup_error(error) is False

    def test_case_insensitive(self):
        error = RequestError(-32603, "Internal error", "FAILED TO SET PROVIDER: ...")
        assert _is_setup_error(error) is True


def _mock_result(stdout="goose 1.28.0", returncode=0, stderr=""):
    m = MagicMock()
    m.stdout = stdout
    m.returncode = returncode
    m.stderr = stderr
    return m


class TestCheckGoose:
    """Tests for _check_goose() version guard."""

    def test_not_installed(self):
        with patch("shutil.which", return_value=None):
            with pytest.raises(PersonaRequirementsUnmet, match="requires the Goose CLI"):
                _check_goose()

    def test_valid_version(self):
        with patch("shutil.which", return_value="/usr/bin/goose"), \
             patch("subprocess.run", return_value=_mock_result("goose 1.28.0")):
            _check_goose()  # should not raise

    def test_old_version(self):
        with patch("shutil.which", return_value="/usr/bin/goose"), \
             patch("subprocess.run", return_value=_mock_result("goose 1.7.0")):
            with pytest.raises(PersonaRequirementsUnmet, match=">=1.8.0"):
                _check_goose()

    def test_exact_minimum_version(self):
        with patch("shutil.which", return_value="/usr/bin/goose"), \
             patch("subprocess.run", return_value=_mock_result("goose 1.8.0")):
            _check_goose()  # should not raise

    def test_major_version_2_rejected(self):
        with patch("shutil.which", return_value="/usr/bin/goose"), \
             patch("subprocess.run", return_value=_mock_result("goose 2.0.0")):
            with pytest.raises(PersonaRequirementsUnmet, match=">=1.8.0,<2"):
                _check_goose()
