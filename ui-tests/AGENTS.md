# E2E tests for jupyter-ai-acp-client

Notes for an agent writing or fixing E2E (Galata) tests here. For generic
Playwright/Galata setup and run commands, see [README.md](./README.md); this file
covers what's specific to testing ACP agent behavior.

## The core idea: a fake ACP agent is just an executable

An ACP persona (`BaseAcpPersona`) is constructed with an `executable: list[str]`.
It spawns that command and speaks the Agent Client Protocol to it over stdio;
that's the _entire_ contract. So to test the UI against a deterministic agent we
don't need a real CLI or network — we write a small Python script that implements
`acp.Agent`, serve it with `acp.run_agent`, and point a persona's `executable` at
it.

- Fake agents live in [`fixtures/agents/`](./fixtures/agents/). `hello_agent.py`
  replies `"hello"` to every prompt (~40 lines); `echo_config_agent.py`
  advertises config options and echoes its current config as YAML. Base new ones
  on these (and on the SDK's `examples/echo_agent.py` upstream).
- Fixture personas live in [`fixtures/personas/`](./fixtures/personas/), named
  `<name>_persona.py`. Each subclasses `BaseAcpPersona`, sets `defaults`, and
  points `executable` at a fake agent script.

**Match the installed SDK, not the SDK repo.** `acp.Agent` method signatures
differ between the pinned `agent-client-protocol` version (see the main
`pyproject.toml`) and the SDK's `main` branch examples. Write against the
installed version — e.g. `prompt(self, prompt, session_id, ...)` in 0.9.0. Check
with `python -c "import acp, inspect; print(inspect.signature(acp.Agent.prompt))"`.

## Persona loading, isolation, and how a suite picks its personas

The PersonaManager loads persona classes from the nearest `.jupyter/personas/`
found by walking **up** from a chat's own directory. We use that for isolation:
one shared test server, but each suite works in its own directory with its own
personas.

A suite **declares its personas in the spec itself** and installs them in
`beforeAll`, then creates its chats under the same directory:

```ts
import { installPersonas, openChat } from './persona-fixtures';

const TEST_DIR = 'replies'; // this suite's working directory

test.describe('…', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, ['hello']); // its personas
  });

  test('…', async ({ page }) => {
    const chat = await openChat(page, TEST_DIR); // chat lives under TEST_DIR
    // → the PersonaManager loads only <TEST_DIR>/.jupyter/personas/
  });
});
```

`installPersonas` (in `tests/persona-fixtures.ts`) reads each fixture persona's
source from `fixtures/personas/` and uploads it to
`<TEST_DIR>/.jupyter/personas/` via Galata's contents API. `openChat` creates the
chat under `<TEST_DIR>/`. So the persona set is owned by the test file that uses
it — no central registry, and two suites can request overlapping or disjoint
persona sets without interfering.

Notes:

- **No entry points.** The real vendored personas are `jupyter_ai.personas`
  entry points and would load in every install; the fixtures deliberately are
  not. `jupyter_server_test_config.py` sets
  `JUPYTER_AI_ACP_CLIENT_E2E_TESTING_CI_ONLY=1`, which disables the vendored set
  (gated in `jupyter_ai_acp_client/acp_personas/__init__.py`) so only the
  fixtures load — deterministic regardless of which agent CLIs are installed.
- **`beforeAll`, not per-test setup**, and it uses the worker-scoped `request`
  fixture (not `page`) — installing files needs no browser, and this keeps UI
  mode working.
- **A fixture persona's class must be defined in its file** (the loader keeps
  only classes whose `__module__` matches the file stem); it may import
  `BaseAcpPersona` etc. The filename must contain `persona`.
- The fake agent path comes from `JAI_TEST_AGENTS_DIR` (exported by the server
  config), so a fixture persona finds its agent wherever it's installed.

## One shared server (ports)

`playwright.config.js` runs a single `webServer` on a random-but-reload-stable
port (pinned into `JAI_TEST_PORT` so every worker's config reload agrees — a
fresh `Math.random()` per reload would desync server vs. client). Its MCP port is
offset from the HTTP port so it doesn't collide with the default (3001) or a dev
server. Both ports are passed as `jlpm start` CLI args, which win over galata's
config defaults. `reuseExistingServer` is `false` so a run never silently reuses
a dev server that lacks the E2E config — free the port before running locally.

The two `projects` in the config just group tests by file in the output; they
share the one server.

## Linting (CI gate)

The repo's `lint:check` runs in CI and **covers `ui-tests/`** — eslint ignores
this directory, but **prettier does not** (its glob is repo-wide). An unformatted
spec or fixture will fail the build. Before pushing, run the root `jlpm lint`
(prettier `--write` + stylelint + eslint) from the repo root:

```bash
cd ..            # repo root
jlpm lint        # fixes formatting; jlpm lint:check to verify without writing
```

## Watching tests run

Use **headed mode**:

```bash
jlpm playwright test replies.spec.ts --headed   # real browser, auto-runs
jlpm playwright test replies.spec.ts --debug    # step through with Inspector
```

Playwright's `--ui` mode renders the page from reconstructed trace snapshots,
which for a Galata/JupyterLab app can paint **blank** even though the test runs
and the action list goes green. `jlpm playwright test --ui --headed` forces a
real browser window if you want the UI explorer.

## Writing a spec

`replies.spec.ts` is the minimal template; `ui-controls.spec.ts` shows changing
toolbar controls and asserting the effect. Load-bearing selectors:

- **Persona picker:** the toolbar button is
  `.jp-jupyter-ai-acp-client-personaControls-persona-btn`. It appears once the
  PersonaManager registers personas, so wait for it with a generous timeout
  (personas + ACP session init take seconds). Click it and pick by name via
  `getByRole('menuitem', { name })`.
- **Session controls** (model/mode/config): the controls row renders each control
  **twice** — a real visible copy and an `aria-hidden`, `inert` measurement copy
  used to size the row. A naive `.control-btn` locator matches the hidden copy
  first (resolves but computes `hidden`, so `toBeVisible` fails). Target the
  visible ones with the direct-child combinator
  `.…-controls > .…-control-btn` (see `ui-controls.spec.ts`).
- **Send:** type into `.jp-chat-input-container` `getByRole('combobox')`, click
  `.jp-chat-send-button`.
- **Assert the reply:** rendered messages are `.jp-chat-rendered-message`. The
  first is the human's; the agent's reply follows.

## Gotchas

- **`beforeAll` runs once per worker**, and the personas it installs persist for
  the suite. If you split a suite across workers (sharding), each worker's
  `beforeAll` reinstalls — which is fine (idempotent upload).
- **Backend changes need a server restart, frontend changes a rebuild.** A
  `.py`-only change to a fake agent or persona is picked up when a chat next
  initializes; changes to the extension's `src/` need `jlpm build` first.
