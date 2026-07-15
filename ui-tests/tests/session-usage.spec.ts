/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fake personas installed into it.
const TEST_DIR = 'session-usage';
const PERSONAS = [
  FixturePersona.SessionUsage,
  FixturePersona.ResponseUsage,
  FixturePersona.BothUsage
];

/**
 * Verifies the toolbar's usage chip renders correctly for each of ACP's two
 * usage channels. An agent can report:
 *
 *   - `session/usage` (standard) — context window fill (+ cost). The chip shows
 *     a ring gauge and a percent.
 *   - `response.usage` (experimental) — cumulative session token counts. The
 *     chip shows a token total with no ring; the popover lists the breakdown.
 *   - both.
 *
 * Each fixture persona wraps the same fake agent in a different `--mode`, so a
 * single suite can assert all three renderings. See fixtures/agents/usage_agent.py.
 */
test.describe('session-usage', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('session/usage only renders a context ring and percent', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.SessionUsage);

    await helpers.sendMessage('report usage');
    await helpers.waitForUsage();

    // 1200 / 4000 = 30% context fill -> ring + "30%".
    expect(await helpers.hasUsageRing()).toBe(true);
    expect(await helpers.usageChipText()).toBe('30%');

    // The popover shows the context section and the reported cost, but no
    // session-token breakdown (that comes from the response.usage channel).
    const card = await helpers.openUsageCard();
    await expect(card).toContainText('Context');
    await expect(card).toContainText('1.2k of 4k');
    await expect(card).toContainText('$0.42');
    await expect(card).not.toContainText('Session tokens');
  });

  test('response.usage only renders session tokens without a ring', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.ResponseUsage);

    await helpers.sendMessage('report usage');
    await helpers.waitForUsage();

    // total_tokens 1500 -> "1.5k", and no context ring (no context reported).
    expect(await helpers.hasUsageRing()).toBe(false);
    expect(await helpers.usageChipText()).toBe('1.5k');

    // The popover lists the session-token breakdown but no context section.
    const card = await helpers.openUsageCard();
    await expect(card).toContainText('Session tokens');
    await expect(card).toContainText('Input');
    await expect(card).toContainText('Output');
    await expect(card).not.toContainText('Context');
  });

  test('both channels render a context ring and a token breakdown', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.BothUsage);

    await helpers.sendMessage('report usage');
    await helpers.waitForUsage();

    // 2000 / 8000 = 25% context fill -> ring + "25%".
    expect(await helpers.hasUsageRing()).toBe(true);
    expect(await helpers.usageChipText()).toBe('25%');

    // The popover shows both the context section and the session-token breakdown.
    const card = await helpers.openUsageCard();
    await expect(card).toContainText('Context');
    await expect(card).toContainText('2k of 8k');
    await expect(card).toContainText('$1.50');
    await expect(card).toContainText('Session tokens');
    await expect(card).toContainText('Input');
    await expect(card).toContainText('Output');
  });
});
