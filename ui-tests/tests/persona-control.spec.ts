/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fake personas installed into it.
const TEST_DIR = 'persona-control';
const PERSONAS = [
  FixturePersona.Hello,
  FixturePersona.Hello2,
  FixturePersona.Hello3
];

// Each fixture persona replies with a distinct word, so a reply unambiguously
// identifies which persona handled the message.
const REPLY: Record<string, string> = {
  [FixturePersona.Hello]: 'hello',
  [FixturePersona.Hello2]: 'bonjour',
  [FixturePersona.Hello3]: 'hola'
};

/**
 * Verifies the persona picker routes each message to the *selected* persona.
 *
 * With more than one persona in a chat, none is auto-selected, so the message's
 * `to_persona` metadata comes solely from the user's picker choice. This asserts
 * that after switching the picker, the message reaches exactly the chosen
 * persona (and only it) — switching back and forth to catch any stale routing.
 */
test.describe('persona-control', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('routes each message to the selected persona', async ({ page }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();

    // Switch to each persona in turn (and back to the first) and confirm the
    // reply is that persona's distinctive word, never another's.
    const order = [
      FixturePersona.Hello,
      FixturePersona.Hello2,
      FixturePersona.Hello3,
      FixturePersona.Hello
    ];
    for (const persona of order) {
      await helpers.selectPersona(persona);
      const reply = await helpers.sendMessage('who are you?');
      expect(reply).toContain(REPLY[persona]);
    }
  });
});
