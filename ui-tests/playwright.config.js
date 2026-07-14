/**
 * Configuration for Playwright using default from @jupyterlab/galata
 */
const baseConfig = require('@jupyterlab/galata/lib/playwright-config');

module.exports = {
  ...baseConfig,
  webServer: {
    command: 'jlpm start',
    url: 'http://localhost:8888/lab',
    timeout: 120 * 1000,
    reuseExistingServer: !process.env.CI,
    // Which fixture personas the server loads (see jupyter_server_test_config.py).
    // A suite needing a different set should run under its own Playwright project
    // with its own webServer env — see AGENTS.md.
    env: {
      ...process.env,
      JAI_TEST_PERSONAS: process.env.JAI_TEST_PERSONAS || 'hello'
    }
  }
};
