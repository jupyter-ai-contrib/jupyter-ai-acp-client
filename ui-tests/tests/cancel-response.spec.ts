/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fake persona installed into it.
const TEST_DIR = 'cancel-response';
const PERSONAS = [FixturePersona.SlowStream];

/**
 * Verifies the stop button interrupts an in-progress reply.
 *
 * The stop button calls the persona-manager cancel endpoint
 * (`/api/ai/personas/cancel`), which invokes `BasePersona.cancel_response()` on
 * each persona; `BaseAcpPersona` overrides it to cancel the agent's turn. The
 * fixture agent streams a number every ~200ms for ~60s (slow_stream_agent.py),
 * so there's a wide window to stop it mid-stream.
 *
 * After stopping we assert two things: the persona is no longer writing (the
 * stop button disables itself, driven by the chat's writers list, which the
 * persona leaves when its response ends), and the reply text stops growing —
 * no further chunks land once cancelled.
 */
test.describe('cancel-response', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('stopping a streaming reply halts writing and freezes the message', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.SlowStream);

    // Kick off the long stream and wait until the agent is actively writing.
    await helpers.send('count for me');
    await helpers.waitForWriting();

    // Let a few chunks accumulate so the message is non-empty and clearly mid-stream.
    await expect
      .poll(async () => (await helpers.lastMessageText()).trim().length, {
        timeout: 30000
      })
      .toBeGreaterThan(0);

    // Interrupt.
    await helpers.clickStop();

    // The persona stops writing: the stop button disables itself.
    await helpers.waitForNotWriting();

    // The message stops growing: capture it, wait, and confirm it's unchanged.
    const frozen = await helpers.lastMessageText();
    await page.waitForTimeout(3000);
    expect(await helpers.lastMessageText()).toBe(frozen);
  });
});
