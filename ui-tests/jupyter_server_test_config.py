"""Server configuration for integration tests.

!! Never use this configuration in production because it
opens the server to the world and provide access to JupyterLab
JavaScript objects through the global window variable.
"""
import json
import os
import shutil
from pathlib import Path

from jupyterlab.galata import configure_jupyter_server

configure_jupyter_server(c)

# The HTTP port (--ServerApp.port) and MCP port (--MCPExtensionApp.mcp_port) are
# passed on the `jlpm start` command line per suite (see playwright.config.js) —
# CLI args win over the defaults set above, so nothing to do here for ports.

# Uncomment to set server log level to debug level
# c.ServerApp.log_level = "DEBUG"

# Disable the real vendored ACP personas so only the test fixtures below load.
# (See jupyter_ai_acp_client/acp_personas/__init__.py.) This keeps the persona
# list deterministic — just the fixtures the suite requested — regardless of
# which agent CLIs happen to be installed on the machine running the tests.
os.environ["JUPYTER_AI_ACP_CLIENT_E2E_TESTING_CI_ONLY"] = "1"

# --- Test persona fixtures ---------------------------------------------------
#
# The PersonaManager loads personas from the nearest `.jupyter/personas/` walking
# up from a chat's own directory. We exploit that to give each test file its own
# persona set from one shared server: each gets its own working directory with
# `<dir>/.jupyter/personas/` populated with the personas it declared. A spec
# creates its chats under its own directory, so it sees only those personas.
#
# `JAI_TEST_LAYOUT` (set in playwright.config.js, the source of truth) is a JSON
# object mapping directory name -> list of fixture persona names. Each fixture
# persona spawns a fake agent from `fixtures/agents/`, whose path we export via
# `JAI_TEST_AGENTS_DIR`.

_UI_TESTS_DIR = Path(__file__).parent.resolve()
_FIXTURES = _UI_TESTS_DIR / "fixtures"
_PERSONAS_SRC = _FIXTURES / "personas"
_AGENTS_SRC = _FIXTURES / "agents"

# Server root: galata runs the server from a fresh mkdtemp by default.
_ROOT = Path(c.ServerApp.root_dir).resolve() if c.ServerApp.root_dir else _UI_TESTS_DIR

# Let fixture persona files locate the fake agent scripts.
os.environ["JAI_TEST_AGENTS_DIR"] = str(_AGENTS_SRC)

_layout = json.loads(os.environ.get("JAI_TEST_LAYOUT", "{}"))
for dir_name, personas in _layout.items():
    dest_dir = _ROOT / dir_name / ".jupyter" / "personas"
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in personas:
        src = _PERSONAS_SRC / f"{name}_persona.py"
        if not src.exists():
            raise FileNotFoundError(
                f"Test persona '{name}' (for '{dir_name}') has no fixture at {src}"
            )
        shutil.copy(src, dest_dir / src.name)
