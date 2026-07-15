/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fake personas installed into it.
const TEST_DIR = 'session-modes';
const PERSONAS = [
  FixturePersona.SetMode,
  FixturePersona.ConfigMode,
  FixturePersona.BothMode
];

// The mode control's toolbar title (the awareness setting's name; see the mode
// agent) and a mode to switch to.
const MODE_CONTROL = 'Mode';
const NEW_MODE = 'Code';

/**
 * Verifies the toolbar's mode control for both of ACP v1's mode channels.
 *
 * ACP v1 lets an agent expose a session mode two ways: the dedicated
 * `session/set_mode` state (`modes` on the session response, changed via
 * `session/set_mode`), or a config option with category `"mode"` (changed via
 * `session/set_config_option`). The client should prefer config options and
 * respect any `category: "mode"` option — so when an agent advertises the mode
 * through *both* channels it must appear as a single control, not two, and
 * changing it must round-trip through whichever channel the agent implements.
 *
 * Each fixture persona wraps the same fake agent (mode_agent.py) with a
 * different `--channel`, so this one suite covers set_mode-only,
 * config-option-only, and both. The agent replies with its current mode as
 * YAML, so switching the control and sending a message proves the round trip.
 */
test.describe('session-modes', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('mode via session/set_mode round-trips', async ({ page }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.SetMode);
    await helpers.waitForControls();

    // Default mode is "ask"; switch to "code" and confirm the reply reflects it.
    await helpers.setControl(MODE_CONTROL, NEW_MODE);
    const reply = await helpers.sendMessage('what mode are you in?');
    expect(reply).toContain('mode: code');
  });

  test('mode via a category=mode config option round-trips', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.ConfigMode);
    await helpers.waitForControls();

    await helpers.setControl(MODE_CONTROL, NEW_MODE);
    const reply = await helpers.sendMessage('what mode are you in?');
    expect(reply).toContain('mode: code');
  });

  test('mode advertised through both channels renders exactly one control', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.BothMode);
    await helpers.waitForControls();

    // Not duplicated: the client folds both channels into a single control.
    expect(await helpers.controlCount(MODE_CONTROL)).toBe(1);

    // And it still round-trips (the client prefers the config-option channel).
    await helpers.setControl(MODE_CONTROL, NEW_MODE);
    const reply = await helpers.sendMessage('what mode are you in?');
    expect(reply).toContain('mode: code');
  });
});
