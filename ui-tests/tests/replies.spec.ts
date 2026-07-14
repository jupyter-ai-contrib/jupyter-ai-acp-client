/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { installPersonas, openChat } from './persona-fixtures';

/**
 * Verifies an ACP persona's reply reaches the chat: selects the "Hello Test
 * Agent" fixture persona (which always replies "hello") and asserts the reply
 * renders. The suite declares its own personas below.
 */

const PERSONA_NAME = 'Hello Test Agent';
const PICKER = '.jp-jupyter-ai-acp-client-personaControls-persona-btn';
// This suite's working directory; its personas are installed under
// `<TEST_DIR>/.jupyter/personas/` so chats created here load only those.
const TEST_DIR = 'replies';

test.describe('hello test agent', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, ['hello']);
  });

  test('replies "hello" when selected as the persona', async ({ page }) => {
    const chat = await openChat(page, TEST_DIR);

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
