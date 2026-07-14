/**
 * Configuration for Playwright using default from @jupyterlab/galata.
 *
 * A single test server serves every suite. Each spec declares and installs the
 * fake personas it needs into its own working directory (see
 * tests/persona-fixtures.ts and AGENTS.md); the PersonaManager loads the nearest
 * `.jupyter/personas/` walking up from a chat's directory, so suites stay
 * isolated on one server.
 */
const baseConfig = require('@jupyterlab/galata/lib/playwright-config');

// Random port so a run doesn't collide with a dev server (or another run) on a
// fixed port. Playwright re-`require`s this config in each worker, so compute it
// once and pin it into the environment — a fresh random value per reload would
// desync the server's port from the port the test workers connect to.
if (!process.env.JAI_TEST_PORT) {
  process.env.JAI_TEST_PORT = String(8989 + Math.floor(Math.random() * 900));
}
const PORT = Number(process.env.JAI_TEST_PORT);

module.exports = {
  ...baseConfig,
  use: { ...(baseConfig.use || {}), baseURL: `http://localhost:${PORT}` },
  projects: [
    { name: 'replies', testMatch: '**/replies.spec.ts' },
    { name: 'ui-controls', testMatch: '**/ui-controls.spec.ts' }
  ],
  webServer: {
    // MCP port offset from the HTTP port so it doesn't collide with a default
    // (3001) or a dev server. CLI args win over galata's config defaults.
    command: `jlpm start --ServerApp.port=${PORT} --MCPExtensionApp.mcp_port=${PORT + 100}`,
    url: `http://localhost:${PORT}/lab`,
    timeout: 120 * 1000,
    // Never reuse an already-running server: reusing an unrelated dev server
    // would leave the E2E persona-disabling config unapplied. Free the port
    // before running locally.
    reuseExistingServer: false
  }
};
