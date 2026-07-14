/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fake personas installed into it.
const TEST_DIR = 'ui-controls';
const PERSONAS = [FixturePersona.EchoConfig];

/** Assert the echoed YAML reports the expected model and effort_level. */
function expectConfig(reply: string, model: string, effort: string): void {
  expect(reply).toContain(`model: ${model}`);
  expect(reply).toContain(`effort_level: ${effort}`);
}

/**
 * Verifies the toolbar's model/settings controls drive the persona's session
 * config. The "Echo Config Agent" fixture advertises two select config options —
 * `model` (default claude-haiku-45) and `effort_level` (default medium) — and
 * replies to every message with its current config as YAML, so a control change
 * shows up in the next reply.
 */
test.describe('ui-controls', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('reports default config before any change', async ({ page }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.EchoConfig);
    await helpers.waitForControls();

    const reply = await helpers.sendMessage('show config');
    expectConfig(reply, 'claude-haiku-45', 'medium');
  });

  test('changing just the model is reflected in the reply', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.EchoConfig);
    await helpers.waitForControls();
    await helpers.setControl('Model', 'claude-opus-48');

    const reply = await helpers.sendMessage('show config');
    expectConfig(reply, 'claude-opus-48', 'medium');
  });

  test('changing just a non-model setting is reflected in the reply', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.EchoConfig);
    await helpers.waitForControls();
    await helpers.setControl('Effort Level', 'high');

    const reply = await helpers.sendMessage('show config');
    expectConfig(reply, 'claude-haiku-45', 'high');
  });

  test('changing both model and setting is reflected in the reply', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.EchoConfig);
    await helpers.waitForControls();
    await helpers.setControl('Model', 'claude-fable-5');
    await helpers.setControl('Effort Level', 'low');

    const reply = await helpers.sendMessage('show config');
    expectConfig(reply, 'claude-fable-5', 'low');
  });
});
