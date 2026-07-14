/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, IJupyterLabPageFixture, test } from '@jupyterlab/galata';
import { UUID } from '@lumino/coreutils';

/**
 * Verifies an ACP persona's reply reaches the chat: selects the "Hello Test
 * Agent" fixture persona (which always replies "hello") and asserts the reply
 * renders. The fixture personas are loaded by the shared test server via
 * JAI_TEST_PERSONAS (see jupyter_server_test_config.py / AGENTS.md).
 */

const PERSONA_NAME = 'Hello Test Agent';
const PICKER = '.jp-jupyter-ai-acp-client-personaControls-persona-btn';
// This test file's working directory. The server installs this suite's personas
// (see playwright.config.js SUITES) under `<TEST_DIR>/.jupyter/personas/`, so
// chats created here load only those personas.
const TEST_DIR = 'replies';

/** Create and open a chat file under TEST_DIR, returning its panel locator. */
async function openChat(page: IJupyterLabPageFixture) {
  const filepath = `${TEST_DIR}/chat-${UUID.uuid4()}.chat`;
  await page.filebrowser.contents.uploadContent('{}', 'text', filepath);
  await page.evaluate(async (name: string) => {
    await window.jupyterapp.commands.execute('jupyterlab-chat:open', {
      filepath: name
    });
  }, filepath);
  const tab = filepath.split('/').pop()!;
  await page.waitForCondition(async () => page.activity.isTabActive(tab));
  return page.activity.getPanelLocator(tab);
}

test.describe('hello test agent', () => {
  test('replies "hello" when selected as the persona', async ({ page }) => {
    const chat = await openChat(page);

    // The persona picker appears once the PersonaManager registers its personas.
    const picker = chat.locator(PICKER);
    await expect(picker).toBeVisible({ timeout: 30000 });

    // Select the fake persona from the picker menu.
    await picker.click();
    await page.getByRole('menuitem', { name: PERSONA_NAME }).click();
    await expect(picker).toContainText(PERSONA_NAME);

    // Send a message.
    const input = chat
      .locator('.jp-chat-input-container')
      .getByRole('combobox');
    await input.pressSequentially('hi there');
    await chat.locator('.jp-chat-input-container .jp-chat-send-button').click();

    // The agent streams back "hello": assert a rendered message contains it.
    // (The first rendered message is the human's "hi there"; the reply follows.)
    const messages = chat.locator('.jp-chat-rendered-message');
    await expect(messages.filter({ hasText: 'hello' })).toBeVisible({
      timeout: 30000
    });
  });
});
