/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import {
  expect,
  IJupyterLabPageFixture,
  Locator,
  test
} from '@jupyterlab/galata';
import { UUID } from '@lumino/coreutils';

/**
 * Verifies the toolbar's model/settings controls drive the persona's session
 * config. Uses the "Echo Config Agent" fixture persona, which advertises two
 * select config options — `model` (default claude-haiku-45) and `effort_level`
 * (default medium) — and replies to every message with its current config as
 * YAML. Changing a control therefore shows up in the next reply, which is what
 * these tests assert. Fixture personas are loaded by the shared test server via
 * JAI_TEST_PERSONAS (see jupyter_server_test_config.py / AGENTS.md).
 */

const PERSONA_NAME = 'Echo Config Agent';
const PICKER = '.jp-jupyter-ai-acp-client-personaControls-persona-btn';
// The controls row renders each control twice: once in an aria-hidden,
// `inert` measurement copy (`-controls-measure`, used only to size the row) and
// once for real as a direct child of `-controls`. Use the direct-child
// combinator (`>`) so we target the visible buttons, not the measurement copy
// (whose duplicate buttons compute as `hidden`).
const VISIBLE_CONTROL_BTN =
  '.jp-jupyter-ai-acp-client-personaControls-controls > .jp-jupyter-ai-acp-client-personaControls-control-btn';
// This test file's working directory. The server installs this suite's personas
// (see playwright.config.js SUITES) under `<TEST_DIR>/.jupyter/personas/`, so
// chats created here load only those personas.
const TEST_DIR = 'ui-controls';

/** Create and open a chat file under TEST_DIR, returning its panel locator. */
async function openChat(page: IJupyterLabPageFixture): Promise<Locator> {
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

/** Select the Echo persona and wait for its config controls to load. */
async function selectEchoPersona(
  page: IJupyterLabPageFixture,
  chat: Locator
): Promise<void> {
  const picker = chat.locator(PICKER);
  await expect(picker).toBeVisible({ timeout: 30000 });
  await picker.click();
  await page.getByRole('menuitem', { name: PERSONA_NAME }).click();
  await expect(picker).toContainText(PERSONA_NAME);
  // The Model + Effort Level controls appear once the ACP session loads.
  await expect(chat.locator(VISIBLE_CONTROL_BTN).first()).toBeVisible({
    timeout: 30000
  });
}

/** Change a select control (identified by its `title`) to the given option. */
async function setControl(
  page: IJupyterLabPageFixture,
  chat: Locator,
  title: string,
  optionValue: string
): Promise<void> {
  // Target the visible control button by its title attribute (the `>` combinator
  // in VISIBLE_CONTROL_BTN excludes the hidden measurement copy).
  await chat.locator(`${VISIBLE_CONTROL_BTN}[title="${title}"]`).click();
  await page.getByRole('menuitem', { name: optionValue, exact: true }).click();
}

/** Send a message and return the text of the latest rendered (agent) message. */
async function sendAndReadReply(
  page: IJupyterLabPageFixture,
  chat: Locator,
  text: string
): Promise<string> {
  const before = await chat.locator('.jp-chat-rendered-message').count();
  const input = chat.locator('.jp-chat-input-container').getByRole('combobox');
  await input.pressSequentially(text);
  await chat.locator('.jp-chat-input-container .jp-chat-send-button').click();
  // Wait for two new rendered messages (the human echo + the agent reply).
  await expect
    .poll(async () => chat.locator('.jp-chat-rendered-message').count(), {
      timeout: 30000
    })
    .toBeGreaterThanOrEqual(before + 2);
  return (
    (await chat.locator('.jp-chat-rendered-message').last().textContent()) ?? ''
  );
}

/** Assert the echoed YAML reports the expected model and effort_level. */
function expectConfig(reply: string, model: string, effort: string): void {
  // The reply is a YAML block: `config_options:\n  model: ...\n  effort_level: ...`.
  expect(reply).toContain(`model: ${model}`);
  expect(reply).toContain(`effort_level: ${effort}`);
}

test.describe('echo config agent', () => {
  test('reports default config before any change', async ({ page }) => {
    const chat = await openChat(page);
    await selectEchoPersona(page, chat);
    const reply = await sendAndReadReply(page, chat, 'show config');
    expectConfig(reply, 'claude-haiku-45', 'medium');
  });

  test('changing just the model is reflected in the reply', async ({
    page
  }) => {
    const chat = await openChat(page);
    await selectEchoPersona(page, chat);
    await setControl(page, chat, 'Model', 'claude-opus-48');
    const reply = await sendAndReadReply(page, chat, 'show config');
    expectConfig(reply, 'claude-opus-48', 'medium');
  });

  test('changing just a non-model setting is reflected in the reply', async ({
    page
  }) => {
    const chat = await openChat(page);
    await selectEchoPersona(page, chat);
    await setControl(page, chat, 'Effort Level', 'high');
    const reply = await sendAndReadReply(page, chat, 'show config');
    expectConfig(reply, 'claude-haiku-45', 'high');
  });

  test('changing both model and setting is reflected in the reply', async ({
    page
  }) => {
    const chat = await openChat(page);
    await selectEchoPersona(page, chat);
    await setControl(page, chat, 'Model', 'claude-fable-5');
    await setControl(page, chat, 'Effort Level', 'low');
    const reply = await sendAndReadReply(page, chat, 'show config');
    expectConfig(reply, 'claude-fable-5', 'low');
  });
});
