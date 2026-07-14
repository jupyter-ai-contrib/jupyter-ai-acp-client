# E2E tests for jupyter-ai-acp-client

Notes for an agent writing or fixing E2E (Galata) tests here. For generic
Playwright/Galata setup and run commands, see [README.md](./README.md); this file
covers what's specific to testing ACP agent behavior.

## The core idea: a fake ACP agent is just an executable

An ACP persona (`BaseAcpPersona`) is constructed with an `executable: list[str]`.
It spawns that command and speaks the Agent Client Protocol to it over stdio;
that's the *entire* contract. So to test the UI against a deterministic agent we
don't need a real CLI or network — we write a small Python script that implements
`acp.Agent`, serve it with `acp.run_agent`, and point a persona's `executable` at
it.

- Fake agents live in [`fixtures/agents/`](./fixtures/agents/). `hello_agent.py`
  is the reference: it replies `"hello"` to every prompt in ~40 lines. Base new
  ones on it (and on the SDK's `examples/echo_agent.py` upstream).
- Fixture personas live in [`fixtures/personas/`](./fixtures/personas/), named
  `<name>_persona.py`. Each subclasses `BaseAcpPersona`, sets `defaults`, and
  points `executable` at a fake agent script.

**Match the installed SDK, not the SDK repo.** `acp.Agent` method signatures
differ between the pinned `agent-client-protocol` version (see the main
`pyproject.toml`) and the SDK's `main` branch examples. Write against the
installed version — e.g. `prompt(self, prompt, session_id, ...)` in 0.9.0. Check
with `python -c "import acp, inspect; print(inspect.signature(acp.Agent.prompt))"`.

## How fixture personas get loaded (no entry points)

The test personas are **not** registered as `jupyter_ai.personas` entry points —
that would attempt to load them in every install, including end users'. Instead
we use the PersonaManager's local-persona loader: it auto-loads any
`*persona*.py` under `<server-root>/.jupyter/personas/`.

`jupyter_server_test_config.py` wires this up at server start:

1. Reads `JAI_TEST_PERSONAS` (comma-separated fixture names, e.g. `hello,slow`).
2. Wipes `<root>/.jupyter/personas/` (so a prior run's personas don't leak) and
   copies `fixtures/personas/<name>_persona.py` for each requested name.
3. Exports `JAI_TEST_AGENTS_DIR` so the copied persona file can locate its fake
   agent script regardless of where it was copied.

Constraints from the loader worth knowing:

- The persona **class must be defined in** the fixture file itself — the loader
  only keeps classes whose `__module__` equals the file's stem. It *may* import
  `BaseAcpPersona`, `PersonaDefaults`, etc. (they're on `sys.path`), but the
  class declaration has to be local.
- The filename must contain `persona` and not start with `_` or `.`.

## Selecting personas per suite

`playwright.config.js` passes `JAI_TEST_PERSONAS` (default `hello`) into the
webServer env. Playwright runs **one** webServer per invocation, so you cannot
restart it mid-run from inside a `describe`. Two ways to give a suite a different
persona set:

- **Whole run:** `JAI_TEST_PERSONAS=foo,bar jlpm playwright test`. Fine when all
  suites in the run want the same set.
- **Per suite (recommended when sets differ):** define a Playwright *project* per
  persona set, each with its own `webServer` (distinct port + `JAI_TEST_PERSONAS`),
  and select the project with `test.describe.configure({ ... })` / project
  `testMatch`. This is the idiomatic Playwright answer to "different server config
  per group of tests" and avoids fighting the single-webServer model.

Keep a fixture persona's behavior obvious from its name, and add a fake agent per
distinct behavior you want to assert (no usage vs. usage-with-limit, different
tool-call kinds, slow init, error paths, etc.) rather than branching one agent on
env.

## Writing a spec

`hello-agent.spec.ts` is the template. The flow and the load-bearing selectors:

- **Open a chat:** upload an empty `*.chat` file via
  `page.filebrowser.contents.uploadContent('{}', 'text', name)`, then execute the
  `jupyterlab-chat:open` command and wait for the tab.
- **Persona picker:** the toolbar button is
  `.jp-jupyter-ai-acp-client-personaControls-persona-btn`. It only appears once
  the PersonaManager has registered personas, so wait for it with a generous
  timeout (personas + ACP session init can take seconds). Click it and pick the
  persona by name via `getByRole('menuitem', { name })`.
- **Send:** type into `.jp-chat-input-container` `getByRole('combobox')`, click
  `.jp-chat-send-button`.
- **Assert the reply:** rendered messages are `.jp-chat-rendered-message`; filter
  by `hasText`. The first rendered message is the human's; the agent's reply
  follows.

Note the picker's default selection comes from the `jupyter_ai_default_persona`
PageConfig value (the persona-manager default, Jupyternaut). That persona is
usually not installed in the test env, so the picker reconciles to the sole
available persona — still, select explicitly in the test to exercise the picker.

## Gotchas

- **Other installed personas are noise.** If your dev machine has real agent CLIs
  (Goose, Claude, etc.) installed, those personas also load and may log init
  errors in the server output. They don't affect a test that selects a
  `TestACP-*` persona by name, but they make the log noisy. CI won't have them.
- **`__file__` resolves** in a fixture persona (loaded via
  `spec_from_file_location`), but prefer `JAI_TEST_AGENTS_DIR` for the agent path
  so the file works wherever it's copied.
- **Backend changes need a server restart, frontend changes a rebuild.** A
  `.py`-only change to a fake agent or persona is picked up on the next server
  start; changes to the extension's `src/` need `jlpm build` first.
