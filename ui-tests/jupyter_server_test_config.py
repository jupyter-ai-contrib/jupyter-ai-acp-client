"""Server configuration for integration tests.

!! Never use this configuration in production because it
opens the server to the world and provide access to JupyterLab
JavaScript objects through the global window variable.
"""
import os
import shutil
from pathlib import Path

from jupyterlab.galata import configure_jupyter_server

configure_jupyter_server(c)

# Uncomment to set server log level to debug level
# c.ServerApp.log_level = "DEBUG"

# --- Test persona fixtures ---------------------------------------------------
#
# The PersonaManager auto-loads persona classes from `<root>/.jupyter/personas/`.
# We use that to install fake ACP personas per test suite without shipping
# anything in the package: a suite sets `JAI_TEST_PERSONAS` to a comma-separated
# list of fixture names before starting the server, and we copy the matching
# `fixtures/personas/<name>_persona.py` into `.jupyter/personas/`. Each fixture
# persona spawns a fake agent from `fixtures/agents/`, whose path we export via
# `JAI_TEST_AGENTS_DIR` so the copied file can find it.

_UI_TESTS_DIR = Path(__file__).parent.resolve()
_FIXTURES = _UI_TESTS_DIR / "fixtures"
_PERSONAS_SRC = _FIXTURES / "personas"
_AGENTS_SRC = _FIXTURES / "agents"

# Server root: galata runs the server from the ui-tests dir by default.
_ROOT = Path(c.ServerApp.root_dir).resolve() if c.ServerApp.root_dir else _UI_TESTS_DIR
_PERSONAS_DEST = _ROOT / ".jupyter" / "personas"

# Let fixture persona files locate the fake agent scripts.
os.environ["JAI_TEST_AGENTS_DIR"] = str(_AGENTS_SRC)

# Start from a clean personas dir each launch so a prior suite's personas don't
# leak into this one.
if _PERSONAS_DEST.exists():
    shutil.rmtree(_PERSONAS_DEST)

_requested = [
    name.strip()
    for name in os.environ.get("JAI_TEST_PERSONAS", "").split(",")
    if name.strip()
]
if _requested:
    _PERSONAS_DEST.mkdir(parents=True, exist_ok=True)
    for name in _requested:
        src = _PERSONAS_SRC / f"{name}_persona.py"
        if not src.exists():
            raise FileNotFoundError(
                f"Requested test persona '{name}' has no fixture at {src}"
            )
        shutil.copy(src, _PERSONAS_DEST / src.name)
