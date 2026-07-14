/**
 * Configuration for Playwright using default from @jupyterlab/galata.
 *
 * Each spec's fake personas are loaded by the test server from
 * JAI_TEST_PERSONAS (see jupyter_server_test_config.py). To keep suites
 * isolated — so one suite's personas don't slow down or interfere with
 * another's — each spec runs under its own Playwright project with its own
 * webServer (a distinct port + JAI_TEST_PERSONAS). Playwright starts each
 * project's webServer on demand and tears it down at the end.
 */
const baseConfig = require('@jupyterlab/galata/lib/playwright-config');

// Base port for the per-suite servers. Randomized so a run doesn't collide with
// a dev server (or another run) already holding a fixed port; each suite takes a
// distinct offset from it. Playwright re-`require`s this config in each worker,
// so the port is computed once and pinned into the environment — otherwise a
// fresh random value per reload would desync the server's port from the port
// the test workers connect to.
if (!process.env.JAI_TEST_BASE_PORT) {
  process.env.JAI_TEST_BASE_PORT = String(
    8989 + Math.floor(Math.random() * 900)
  );
}
const BASE_PORT = Number(process.env.JAI_TEST_BASE_PORT);

// One project per persona set: (project name, spec, port offset, personas).
const SUITES = [
  { name: 'hello', spec: 'hello-agent.spec.ts', personas: 'hello' },
  { name: 'echo', spec: 'echo-config.spec.ts', personas: 'echo' }
].map((s, i) => ({ ...s, port: BASE_PORT + i }));

module.exports = {
  ...baseConfig,
  projects: SUITES.map(s => ({
    name: s.name,
    testMatch: `**/${s.spec}`,
    use: { ...(baseConfig.use || {}), baseURL: `http://localhost:${s.port}` }
  })),
  webServer: SUITES.map(s => ({
    // Each suite's server gets its own HTTP port and MCP port (the MCP extension
    // otherwise defaults to a fixed 3001 and two servers would collide). CLI args
    // win over galata's config defaults.
    command: `jlpm start --ServerApp.port=${s.port} --MCPExtensionApp.mcp_port=${s.port + 100}`,
    url: `http://localhost:${s.port}/lab`,
    timeout: 120 * 1000,
    // Never reuse an already-running server: the tests need one started with the
    // fixture personas below, and reusing an unrelated dev server would silently
    // run them with no personas. Free the ports before running locally.
    reuseExistingServer: false,
    env: {
      ...process.env,
      JAI_TEST_PERSONAS: s.personas
    }
  }))
};
