/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fake persona installed into it.
const TEST_DIR = 'duplicate-config-option-groups';
const PERSONAS = [FixturePersona.DuplicateGroups];

/**
 * Verifies the client's tie-breaking when several config options share a
 * category.
 *
 * ACP says that when multiple options carry the same category, the client
 * resolves the tie by array order — the earliest wins the prominent slot, and
 * later same-category options render as ordinary settings
 * (https://agentclientprotocol.com/protocol/v1/session-config-options
 * #option-categories). The fixture agent advertises, in order: two
 * `category: "model"` options (`model`, then `model_alt` labelled "Backup
 * Model") and two `category: "mode"` options (`mode`, then `mode_alt` labelled
 * "Backup Mode").
 *
 * So the toolbar should surface the first of each as the prominent Model / Mode
 * controls, and the runners-up as plain settings under their own labels — every
 * option stays reachable, none is dropped or duplicated. Changing the prominent
 * Model control round-trips through the winning option.
 */
test.describe('duplicate-config-option-groups', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('first option of each category wins the prominent control', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.DuplicateGroups);
    await helpers.waitForControls();

    // The first model/mode options are the prominent Model/Mode controls; the
    // second of each is a plain setting under its own label. Each appears once.
    expect(await helpers.controlCount('Model')).toBe(1);
    expect(await helpers.controlCount('Mode')).toBe(1);
    expect(await helpers.controlCount('Backup Model')).toBe(1);
    expect(await helpers.controlCount('Backup Mode')).toBe(1);
  });

  test('the winning model control round-trips', async ({ page }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.DuplicateGroups);
    await helpers.waitForControls();

    // The prominent Model control offers the first model option's choices.
    await helpers.setControl('Model', 'Haiku');
    const reply = await helpers.sendMessage('show config');
    // Only the prominent model changed; the backup model keeps its default.
    expect(reply).toContain('model: haiku');
    expect(reply).toContain('model_alt: gpt');
  });

  test('a runner-up option round-trips as a plain setting', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.DuplicateGroups);
    await helpers.waitForControls();

    await helpers.setControl('Backup Mode', 'Slow');
    const reply = await helpers.sendMessage('show config');
    expect(reply).toContain('mode_alt: slow');
    // The prominent mode is untouched.
    expect(reply).toContain('mode: ask');
  });
});
