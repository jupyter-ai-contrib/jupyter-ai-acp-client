/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fake persona installed into it.
const TEST_DIR = 'slash-commands';
const PERSONAS = [FixturePersona.SlashCommands];

/**
 * Verifies that a persona's slash commands, advertised by the ACP agent over
 * the awareness channel, surface as chat-input completions.
 *
 * The fake agent sends an `available_commands_update` announcing `/compact`,
 * `/clear`, and `/help` (see slash_commands_agent.py). The client reads that
 * list from the selected persona's awareness slot — no REST — and offers it in
 * the input's autocomplete popup when the user types `/`, filtered by what
 * they've typed so far.
 */
test.describe('slash-commands', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('offers the agent-advertised commands when typing "/"', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    // Selecting the persona stamps `to_persona` on the input metadata, which is
    // how the slash provider knows whose command list to read.
    await helpers.selectPersona(FixturePersona.SlashCommands);

    const names = await helpers.slashCompletions('/', [
      '/compact',
      '/clear',
      '/help'
    ]);
    expect(names).toEqual(
      expect.arrayContaining(['/compact', '/clear', '/help'])
    );
  });

  test('filters completions by the typed prefix', async ({ page }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.SlashCommands);

    // "/c" matches /compact and /clear, but not /help.
    const names = await helpers.slashCompletions('/c', ['/compact', '/clear']);
    expect(names).toEqual(expect.arrayContaining(['/compact', '/clear']));
    expect(names).not.toContain('/help');
  });
});
