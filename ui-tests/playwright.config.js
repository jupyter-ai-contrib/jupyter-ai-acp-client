/**
 * Configuration for Playwright using default from @jupyterlab/galata.
 *
 * A single test server serves every suite. Each test file works in its own
 * directory with its own persona set: SUITES below is the source of truth,
 * mapping a directory name to the fixture personas installed under
 * `<dir>/.jupyter/personas/` (see jupyter_server_test_config.py, which does the
 * copying at startup from JAI_TEST_LAYOUT). A spec creates its chats under its
 * own directory, so the PersonaManager loads only that directory's personas.
 */
const baseConfig = require('@jupyterlab/galata/lib/playwright-config');

// Source of truth: test-file directory -> personas available to it.
const SUITES = [
  { name: 'replies', personas: ['hello'] },
  { name: 'ui-controls', personas: ['echo'] }
];

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
    // Never reuse an already-running server: the tests need one started with the
    // persona layout below, and reusing an unrelated dev server would silently
    // run them with no personas. Free the port before running locally.
    reuseExistingServer: false,
    env: {
      ...process.env,
      // dir -> personas, consumed by jupyter_server_test_config.py.
      JAI_TEST_LAYOUT: JSON.stringify(
        Object.fromEntries(SUITES.map(s => [s.name, s.personas]))
      )
    }
  }
};
