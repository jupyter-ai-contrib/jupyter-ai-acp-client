/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import {
  expect,
  galata,
  IJupyterLabPageFixture,
  test
} from '@jupyterlab/galata';
import { Locator } from '@playwright/test';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fake persona installed into it.
const TEST_DIR = 'tool-calls';
const PERSONAS = [FixturePersona.ToolCall];

// The file the fixture agent asks to edit (see tool_call_agent.py).
const TARGET_FILE = 'tool-call-target.txt';

const TOOL_CALLS = '.jp-ai-tool-calls';
const COMPLETED_ITEM = '.jp-ai-tool-call-item-completed';
const FAILED_ITEM = '.jp-ai-tool-call-item-failed';
const DIFF_HEADER = '.jp-ai-tool-call-diff-header';
const PERMISSION_LABEL = '.jp-ai-tool-call-permission-label';

const TIMEOUT = 30000;

/**
 * Send a prompt to the fixture agent and return the helpers and the tool
 * calls block, once the permission request shows.
 */
async function requestEdit(
  page: IJupyterLabPageFixture
): Promise<{ helpers: TestHelpers; toolCalls: Locator }> {
  const helpers = new TestHelpers({ dir: TEST_DIR, page });
  await helpers.openChat();
  await helpers.selectPersona(FixturePersona.ToolCall);
  await helpers.send('edit the file');

  const toolCalls = helpers.chat.locator(TOOL_CALLS);
  await expect(toolCalls.getByRole('button', { name: 'Allow' })).toBeEnabled({
    timeout: TIMEOUT
  });
  return { helpers, toolCalls };
}

/**
 * Verifies the tool call UI that jupyter-chat-components renders for an ACP
 * permission request: the fixture agent asks to edit a file, then replies
 * "allowed" or "rejected" from the option the user clicks.
 */
test.describe('tool-calls', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
    await galata
      .newContentsHelper(request)
      .uploadContent('old line\n', 'text', `${TEST_DIR}/${TARGET_FILE}`);
  });

  test('allowing a permission request completes the tool call', async ({
    page
  }) => {
    const { helpers, toolCalls } = await requestEdit(page);
    const allow = toolCalls.getByRole('button', { name: 'Allow' });

    // The stylesheet comes from the jupyter-chat-components extension.
    await expect(allow).toHaveCSS('cursor', 'pointer');

    await allow.click();

    await expect(toolCalls.locator(PERMISSION_LABEL)).toContainText('Allow', {
      timeout: TIMEOUT
    });
    await expect(toolCalls.locator(COMPLETED_ITEM)).toHaveCount(1);
    await expect
      .poll(async () => helpers.lastMessageText(), { timeout: TIMEOUT })
      .toContain('allowed');
  });

  test('rejecting a permission request fails the tool call', async ({
    page
  }) => {
    const { helpers, toolCalls } = await requestEdit(page);

    await toolCalls.getByRole('button', { name: 'Reject' }).click();

    await expect(toolCalls.locator(PERMISSION_LABEL)).toContainText('Reject', {
      timeout: TIMEOUT
    });
    await expect(toolCalls.locator(FAILED_ITEM)).toHaveCount(1);
    await expect
      .poll(async () => helpers.lastMessageText(), { timeout: TIMEOUT })
      .toContain('rejected');
  });

  test('clicking a diff header opens the file', async ({ page }) => {
    const { toolCalls } = await requestEdit(page);
    await toolCalls.getByRole('button', { name: 'Allow' }).click();

    // A completed tool call collapses its diff.
    const completed = toolCalls.locator(COMPLETED_ITEM);
    await expect(completed).toHaveCount(1, { timeout: TIMEOUT });
    await completed.locator('summary').click();

    const header = toolCalls.locator(DIFF_HEADER);
    await expect(header).toHaveText(`${TEST_DIR}/${TARGET_FILE}`);
    await header.click();

    await page.waitForCondition(async () =>
      page.activity.isTabActive(TARGET_FILE)
    );
  });
});
