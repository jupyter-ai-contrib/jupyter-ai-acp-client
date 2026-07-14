/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, IJupyterLabPageFixture, test } from '@jupyterlab/galata';
import { UUID } from '@lumino/coreutils';

/**
 * This suite needs the "hello" fixture persona, which the server config installs
 * when JAI_TEST_PERSONAS includes it (see jupyter_server_test_config.py). The
 * webServer command sets JAI_TEST_PERSONAS=hello. To run a suite against a
 * different persona set, give it its own Playwright project with its own
 * webServer env — see ui-tests/AGENTS.md.
 */

const PERSONA_NAME = 'Hello Test Agent';
const PICKER = '.jp-jupyter-ai-acp-client-personaControls-persona-btn';

/** Create and open a chat file, returning its panel locator. */
async function openChat(page: IJupyterLabPageFixture) {
  const filename = `hello-${UUID.uuid4()}.chat`;
  await page.filebrowser.contents.uploadContent('{}', 'text', filename);
  await page.evaluate(async (name: string) => {
    await window.jupyterapp.commands.execute('jupyterlab-chat:open', {
      filepath: name
    });
  }, filename);
  const tab = filename.split('/').pop()!;
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
    const input = chat.locator('.jp-chat-input-container').getByRole('combobox');
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
