# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
"""E2E test matrix for jupyter_ai_acp_client.

Runs the Playwright/galata ui-tests suite under three transports so the ACP
client's chat integration is exercised end-to-end in every supported mode:

    - default   -- RTC-free, WebSocket ``WsChatModel``
    - jcollab   -- RTC via ``jupyter_collaboration``
    - jsd       -- RTC via ``jupyter_server_documents``

Each session builds an isolated venv (uv) and installs the extension -- the
prebuilt wheel via ``E2E_WHEEL`` in CI, or from source locally -- plus the
transport's packages, then runs the suite. Playwright launches JupyterLab from
this venv, so the server under test runs the selected transport.

The JS deps and browser binaries are expected to be present in ``ui-tests``
already (installed once at the CI job level, where ``playwright install-deps``
can use sudo); the ``jlpm install`` / ``playwright install`` calls below are
idempotent no-ops on a hit and make local runs self-contained.

Usage::

    nox -l                          # list sessions
    nox -s e2e                      # all three transports
    nox -s "e2e(env='jcollab')"     # one transport
"""
import os

import nox

# Prefer uv for fast, isolated env creation; fall back to virtualenv.
nox.options.default_venv_backend = "uv|virtualenv"

# env name -> extra packages that provide the transport.
#
# jupyter_collaboration is pinned <5: 5.0.0 shows a "document is taking some
# time to load" dialog that never clears for .chat documents under slow-load
# timing, hanging the ui-tests. Same pin as .github/workflows/build.yml. The
# floors mirror the validated jupyter-ai-router RTC matrix.
_ENVS = {
    "default": [],
    "jcollab": ["jupyter_collaboration>=4,<5"],
    "jsd": ["jupyter_server_documents"],
}


@nox.session(python="3.10")
@nox.parametrize("env", list(_ENVS))
def e2e(session: nox.Session, env: str) -> None:
    """Run the ui-tests suite against one transport."""
    # The prebuilt wheel from the CI ``build`` job; from source for local runs.
    target = os.environ.get("E2E_WHEEL") or "."
    session.install("jupyterlab>=4.0.0,<5", target, *_ENVS[env])
    with session.chdir("ui-tests"):
        session.run("jlpm", "install", external=True)
        session.run("jlpm", "playwright", "install", "chromium", external=True)
        session.run("jlpm", "playwright", "test", external=True)
