/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fake personas installed into it.
const TEST_DIR = 'picker';
const PERSONAS = [FixturePersona.Hello];

/**
 * Verifies the picker never overrides an explicit user choice: after picking
 * "No one", awareness updates (personas rebroadcast usage, writing state, and
 * commands routinely) must not snap the selection back to a persona.
 *
 * On CI the chat has exactly the one fixture persona, which is the case where
 * the sole-persona convenience used to override the choice. Locally, other
 * installed persona packages (e.g. jupyternaut) may add personas beyond the
 * fixtures; the contract must hold there too.
 */
test.describe('picker', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('keeps an explicit "No one" selection', async ({ page }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.Hello);

    await helpers.selectNoOne();

    // Sending a message triggers awareness traffic; the explicit "No one"
    // must survive it.
    await helpers.sendWithoutReply('anyone there?');
    await page.waitForTimeout(2500);
    await expect(helpers.personaPicker).toContainText('No one');
  });
});
