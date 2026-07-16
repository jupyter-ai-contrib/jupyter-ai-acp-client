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
  advertises config options and echoes its current config as YAML;
  `usage_agent.py` reports usage via ACP's two channels, selected by a `--mode`
  CLI flag (see "Testing usage" below). Base new ones on these (and on the SDK's
  `examples/echo_agent.py` upstream). An agent can take config as CLI flags on
  its `executable` — that's how one script backs several personas.
- Fixture personas live in [`fixtures/personas/`](./fixtures/personas/), named
  `<name>_persona.py`. Each subclasses `BaseAcpPersona`, sets `defaults`, and
  points `executable` at a fake agent script.

**Match the installed SDK, not the SDK repo.** `acp.Agent` method signatures
differ between the pinned `agent-client-protocol` version (see the main
`pyproject.toml`, currently 0.11) and the SDK's `main` branch examples, and
across pinned versions too — e.g. `prompt` is `(self, prompt, session_id, ...)`
in 0.9 but `(self, session_id, prompt, ...)` in 0.11, and `session/set_model`
was removed entirely in 0.11 (models are `category: "model"` config options
now). Write against the installed version and pass args by keyword. Check with
`python -c "import acp, inspect; print(inspect.signature(acp.Agent.prompt))"`.

## Persona loading, isolation, and how a suite picks its personas

The PersonaManager loads persona classes from the nearest `.jupyter/personas/`
found by walking **up** from a chat's own directory. We use that for isolation:
one shared test server, but each suite works in its own directory with its own
personas.

A suite **declares its personas in the spec itself** and installs them in
`beforeAll`; a `TestHelpers` instance (from `tests/test-helpers.ts`) drives the
chat under the suite's directory:

```ts
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

const TEST_DIR = 'replies'; // this suite's working directory
const PERSONAS = [FixturePersona.Hello]; // its personas

test.describe('…', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('…', async ({ page }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat(); // chat lives under TEST_DIR
    await helpers.selectPersona(FixturePersona.Hello);
    const reply = await helpers.sendMessage('hi');
    // → routed to only this suite's persona
  });
});
```

The available fixtures are the `FixturePersona` enum in `test-helpers.ts`, whose
`FIXTURE_PERSONAS` table is the single source of truth for each persona's display
name — so specs never hardcode persona names. `installPersonas` reads each
fixture's source from `fixtures/personas/<value>_persona.py` and uploads it to
`<TEST_DIR>/.jupyter/personas/` via Galata's contents API; `TestHelpers` methods
(`openChat`, `selectPersona`, `setControl`, `sendMessage`, …) take `{ dir, page }`
once at construction so specs stay minimal. The persona set is owned by the test
file that uses it — two suites can request overlapping or disjoint sets without
interfering.

Notes:

- **No entry points.** The real vendored personas are `jupyter_ai.personas`
  entry points and would load in every install; the fixtures deliberately are
  not. `jupyter_server_test_config.py` sets
  `JUPYTER_AI_ACP_CLIENT_E2E_TESTING_ONLY=1`, which disables the vendored set
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

Every `*.spec.ts` under `tests/` runs against that one server; the config
defines no `projects`, so adding a spec file needs no config change.

## Linting (CI gate)

The repo's `lint:check` runs in CI and **covers `ui-tests/`** — eslint ignores
this directory, but **prettier does not** (its glob is repo-wide). An unformatted
spec or fixture will fail the build. Before pushing, run the root `jlpm lint`
(prettier `--write` + stylelint + eslint) from the repo root:

```bash
cd ..            # repo root
jlpm lint        # fixes formatting; jlpm lint:check to verify without writing
```

## Running tests interactively

```bash
cd ui-tests && jlpm && jlpm playwright test --ui
```

Other useful modes:

```bash
jlpm playwright test replies.spec.ts --headed   # real browser, auto-runs
jlpm playwright test replies.spec.ts --debug    # step through with Inspector
```

Note: `--ui` mode renders the page from reconstructed trace snapshots, which for
a Galata/JupyterLab app can paint **blank** even though the test runs and the
action list goes green. `jlpm playwright test --ui --headed` forces a real
browser window per run.

## Writing a spec

`replies.spec.ts` is the minimal template; `ui-controls.spec.ts` shows changing
toolbar controls and asserting the effect. Load-bearing selectors:

- **Persona picker:** the toolbar button is
  `.jp-jai-personaControls-persona-btn` (the toolbar is rendered by
  jupyter-ai-persona-manager, hence its `jp-jai-` prefix). It appears once the
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

## Testing usage (the ACP usage channels)

ACP v1 reports usage through two distinct channels, which the toolbar's usage
chip renders differently; kiro-cli uses a vendor extension instead:

- **`session/usage`** ([standard](https://agentclientprotocol.com/rfds/session-usage))
  — a `usage_update` session update carrying the context window fill
  (`used`/`size`) and an optional cumulative `cost`. The chip shows a **ring
  gauge + percent**; the popover shows a Context section and the cost.
- **`response.usage`** ([experimental](https://agentclientprotocol.com/rfds/end-turn-token-usage))
  — a `usage` object on the `PromptResponse` carrying cumulative session token
  counts (input/output/total, …). The chip shows a **token total with no ring**;
  the popover lists the token breakdown.
- **`_kiro.dev/metadata`** (kiro vendor extension) — a notification carrying a
  bare `contextUsagePercentage`. The chip shows a **ring gauge + percent** with
  no token counts; the popover shows a percent-only Context section.

An agent may report any combination. `usage_agent.py` does all four via a
`--mode {session,response,both,kiro}` flag, and four fixture personas
(`session-usage`, `response-usage`, `both-usage`, `kiro-usage`) wrap it with the
respective flag. `session-usage.spec.ts` loads all four and asserts each
rendering — ring presence, the chip's percent/token text, and the popover's
sections. `TestHelpers`
exposes `waitForUsage`, `hasUsageRing`, `usageChipText`, and `openUsageCard` for
this. The usage popover is a MUI portal at the page root, so `openUsageCard`
returns a **page-scoped** locator, not a chat-scoped one.

Numbers are fixed in the agent so the expected UI text is deterministic (e.g.
`1200/4000` → `"30%"`, `total_tokens: 1500` → `"1.5k"`); the client formats
token counts compactly (`formatTokens`), so assert the compact form.

**These multi-persona suites (`session-usage.spec.ts` and
`persona-control.spec.ts`) rely on metadata routing.** They load several
personas and route by the picker's `to_persona` metadata, which needs the
PersonaManager change from jupyter-ai-persona-manager PR #59 (shipped in
0.1.0b1). Older published versions (≤0.0.12) routed only by `@`-mention and
auto-replied only when a chat had exactly one persona, so a multi-persona suite
reached nobody and no reply rendered. They ran unconditionally once acp-client's
`jupyter_ai_persona_manager` floor was bumped to `>=0.1.0b1`; keep that floor at
or above the release including #59 for these suites to pass on CI.
(`replies`/`ui-controls` are unaffected: one persona each, covered even by the
old single-persona auto-reply.)

## Config option categories: model, mode, and tie-breaking

Session config options carry an optional
[`category`](https://agentclientprotocol.com/protocol/v1/session-config-options#option-categories)
that tells the client where to place the control. The adapter buckets by it:
`"model"` → the prominent Model picker, `"mode"` → the Mode selector,
`"model_config"` → settings beside the model, everything else → a general
setting. (The dedicated `session/set_model` API was removed from the protocol, so
a model is just a `"model"`-category config option now; a mode may come either
through the dedicated `session/set_mode` state **or** a `"mode"` config option.)

- **Modes — `mode_agent.py`** advertises a mode through one or both channels via
  `--channel {set_mode,config_option,both}`; three fixtures (`set-mode`,
  `config-mode`, `both-mode`) wrap it. `session-modes.spec.ts` asserts each
  channel round-trips and — crucially — that a mode advertised through _both_
  renders as a **single** control (the client prefers the config option and
  de-dupes). It echoes its current mode as YAML so switching the control and
  sending a message proves the round trip.
- **Tie-breaking — `duplicate_groups_agent.py`** advertises two options in the
  same category (two `model`, two `mode`). ACP resolves such ties by array order,
  so the earliest wins the prominent slot and the runner-up shows as a plain
  setting under its own label. `duplicate-config-option-groups.spec.ts` asserts
  both the placement and that each still round-trips.

`TestHelpers#controlCount(title)` counts visible controls with a given `title`
(the control's label) — used to assert the no-duplication and tie-break layouts.

## Testing slash commands

An ACP agent advertises slash commands with an `available_commands_update` session
notification (there's no field for them on the session response). The client reads
that list from the persona's awareness slot and offers it as chat-input
completions when the user types `/`. `slash_commands_agent.py` announces a fixed
list (`/compact`, `/clear`, `/help`) shortly after `new_session`;
`slash-commands.spec.ts` asserts they appear and filter by the typed prefix.

`TestHelpers#slashCompletions(prefix, waitFor)` types `prefix` into the input and
returns the command names in the autocomplete popup (a page-scoped MUI portal).
The command list arrives over awareness **asynchronously** after the ACP session
initializes, and the provider only re-queries on a keystroke — so the helper
polls, clearing and retyping `prefix` each round, until every name in `waitFor` is
present. Always pass the agent's commands as `waitFor`, or the read can race the
update and see only built-ins (e.g. `/refresh-personas`).

## Gotchas

- **`beforeAll` runs once per worker**, and the personas it installs persist for
  the suite. If you split a suite across workers (sharding), each worker's
  `beforeAll` reinstalls — which is fine (idempotent upload).
- **Backend changes need a server restart, frontend changes a rebuild.** A
  `.py`-only change to a fake agent or persona is picked up when a chat next
  initializes; changes to the extension's `src/` need `jlpm build` first.
