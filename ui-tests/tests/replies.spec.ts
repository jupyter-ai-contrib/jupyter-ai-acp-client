/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fake personas installed into it.
const TEST_DIR = 'replies';
const PERSONAS = [FixturePersona.Hello];

/**
 * Verifies an ACP persona's reply reaches the chat: the Hello fixture always
 * replies "hello".
 */
test.describe('replies', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('shows the agent reply in the chat', async ({ page }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.Hello);

    const reply = await helpers.sendMessage('hi there');
    expect(reply).toContain('hello');
  });
});
