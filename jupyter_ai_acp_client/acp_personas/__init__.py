"""Vendored ACP personas shipped by this package.

During E2E testing we want only the test fixture personas to load, not the real
vendored agents (they'd need their CLIs installed, add startup noise, and can
interfere with the deterministic fixtures). Setting
`JUPYTER_AI_ACP_CLIENT_E2E_TESTING_CI_ONLY=1` disables the whole vendored set:
importing any persona in this package raises `PersonaRequirementsUnmet`, which
the PersonaManager treats as "skip this persona" (the same path used when a real
agent's CLI is missing). The env var is deliberately verbose to avoid collisions.
"""

import os

from jupyter_ai_persona_manager import PersonaRequirementsUnmet

if os.environ.get("JUPYTER_AI_ACP_CLIENT_E2E_TESTING_CI_ONLY") == "1":
    raise PersonaRequirementsUnmet(
        "Vendored ACP personas are disabled during E2E testing "
        "(JUPYTER_AI_ACP_CLIENT_E2E_TESTING_CI_ONLY=1)."
    )
