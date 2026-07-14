"""Server configuration for integration tests.

!! Never use this configuration in production because it
opens the server to the world and provide access to JupyterLab
JavaScript objects through the global window variable.
"""
import os
from pathlib import Path

from jupyterlab.galata import configure_jupyter_server

configure_jupyter_server(c)

# The HTTP port (--ServerApp.port) and MCP port (--MCPExtensionApp.mcp_port) are
# passed on the `jlpm start` command line (see playwright.config.js) — CLI args
# win over the defaults set above, so nothing to do here for ports.

# Uncomment to set server log level to debug level
# c.ServerApp.log_level = "DEBUG"

# Disable the real vendored ACP personas so only the test fixtures load. (See
# jupyter_ai_acp_client/acp_personas/__init__.py.) This keeps the persona list
# deterministic regardless of which agent CLIs happen to be installed on the
# machine running the tests.
os.environ["JUPYTER_AI_ACP_CLIENT_E2E_TESTING_CI_ONLY"] = "1"

# Each test suite installs the fake personas it needs at runtime, into its own
# working directory (see tests/persona-fixtures.ts). The PersonaManager loads the
# nearest `.jupyter/personas/` walking up from a chat's directory, so a suite's
# chats see only its personas. The fixture persona files locate their fake agent
# scripts via this env var.
os.environ["JAI_TEST_AGENTS_DIR"] = str(
    Path(__file__).parent.resolve() / "fixtures" / "agents"
)
