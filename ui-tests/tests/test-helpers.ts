/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 *
 * Helpers for the ACP E2E suites.
 *
 * Each suite works in its own directory with its own fake personas. In
 * `beforeAll`, call `installPersonas(request, dir, [...])` to copy the named
 * fixture personas into `<dir>/.jupyter/personas/`; the PersonaManager loads the
 * nearest `.jupyter/personas` walking up from a chat's directory, so a chat
 * created under `<dir>/` (via `TestHelpers#openChat`) sees only that suite's
 * personas. One shared server, per-suite isolation by directory. See AGENTS.md.
 */

import { expect, galata, IJupyterLabPageFixture } from '@jupyterlab/galata';
import { APIRequestContext, Locator } from '@playwright/test';
import { UUID } from '@lumino/coreutils';
import * as fs from 'fs';
import * as path from 'path';

const PERSONAS_SRC = path.resolve(__dirname, '..', 'fixtures', 'personas');

/**
 * The fake personas available under `fixtures/personas/`. The value is the
 * fixture's file stem (`<value>_persona.py`) and the id a suite passes to
 * `installPersonas`.
 */
export enum FixturePersona {
  Hello = 'hello',
  Hello2 = 'hello-2',
  Hello3 = 'hello-3',
  EchoConfig = 'echo-config',
  SessionUsage = 'session-usage',
  ResponseUsage = 'response-usage',
  BothUsage = 'both-usage',
  SetMode = 'set-mode',
  ConfigMode = 'config-mode',
  BothMode = 'both-mode',
  DuplicateGroups = 'duplicate-groups',
  SlashCommands = 'slash-commands',
  SlowStream = 'slow-stream',
  KiroUsage = 'kiro-usage'
}

interface FixturePersonaInfo {
  /** Display name shown in the persona picker (its `PersonaDefaults.name`). */
  name: string;
}

/** Single source of truth for each fixture persona's metadata. */
export const FIXTURE_PERSONAS: Record<FixturePersona, FixturePersonaInfo> = {
  [FixturePersona.Hello]: { name: 'Hello Test Agent' },
  [FixturePersona.Hello2]: { name: 'Hello Agent Two' },
  [FixturePersona.Hello3]: { name: 'Hello Agent Three' },
  [FixturePersona.EchoConfig]: { name: 'Echo Config Agent' },
  [FixturePersona.SessionUsage]: { name: 'Session Usage Agent' },
  [FixturePersona.ResponseUsage]: { name: 'Response Usage Agent' },
  [FixturePersona.BothUsage]: { name: 'Both Usage Agent' },
  [FixturePersona.SetMode]: { name: 'Set Mode Agent' },
  [FixturePersona.ConfigMode]: { name: 'Config Mode Agent' },
  [FixturePersona.BothMode]: { name: 'Both Mode Agent' },
  [FixturePersona.DuplicateGroups]: { name: 'Duplicate Groups Agent' },
  [FixturePersona.SlashCommands]: { name: 'Slash Commands Agent' },
  [FixturePersona.SlowStream]: { name: 'Slow Stream Agent' },
  [FixturePersona.KiroUsage]: { name: 'Kiro Usage Agent' }
};

const PICKER = '.jp-jai-personaControls-persona-btn';
// The controls row renders each control twice: a real visible copy and an
// aria-hidden, `inert` measurement copy (used only to size the row). The
// direct-child combinator targets the visible buttons — the duplicates in the
// measurement copy are nested one level deeper and compute as `hidden`.
const VISIBLE_CONTROL_BTN =
  '.jp-jai-personaControls-controls > .jp-jai-personaControls-control-btn';
const INPUT = '.jp-chat-input-container';
const MESSAGE = '.jp-chat-rendered-message';
// Each slash-command completion in the input's autocomplete popup renders its
// name in a `.jp-chat-command-name` span (MUI list options, page-scoped portal).
const COMMAND_NAME = '.jp-chat-command-name';
// The toolbar's stop button: enabled only while an AI persona is writing, so its
// disabled state doubles as a "is the persona still streaming?" signal.
const STOP_BUTTON = '.jp-jupyter-ai-acp-client-stopButton';

// The usage chip and the parts that distinguish which usage channel an agent
// reported: a context ring + percent (session/usage) and/or a session-token
// breakdown in the popover card (response.usage). See persona-controls.tsx.
const USAGE_CHIP = '.jp-jai-usage-chip';
const USAGE_RING = '.jp-jai-usage-ring';
const USAGE_PCT = '.jp-jai-usage-pct';
const USAGE_CARD = '.jp-jai-usage-card';

const TIMEOUT = 30000;

/**
 * Install the named fixture personas into `<dir>/.jupyter/personas/`. Call from
 * `beforeAll` with the worker-scoped `request` fixture (no browser needed).
 */
export async function installPersonas(
  request: APIRequestContext,
  dir: string,
  personas: FixturePersona[]
): Promise<void> {
  const contents = galata.newContentsHelper(request);
  for (const persona of personas) {
    const file = `${persona}_persona.py`;
    const source = fs.readFileSync(path.join(PERSONAS_SRC, file), 'utf-8');
    await contents.uploadContent(
      source,
      'text',
      `${dir}/.jupyter/personas/${file}`
    );
  }
}

/**
 * Per-test helper bound to one suite directory and one page. Drives the chat UI:
 * open a chat, pick a persona, change controls, send and read messages.
 */
export class TestHelpers {
  readonly dir: string;
  readonly page: IJupyterLabPageFixture;
  private _chat: Locator | null = null;

  constructor(options: { dir: string; page: IJupyterLabPageFixture }) {
    this.dir = options.dir;
    this.page = options.page;
  }

  /** The current chat panel (throws if `openChat` hasn't run). */
  get chat(): Locator {
    if (!this._chat) {
      throw new Error('Call openChat() first.');
    }
    return this._chat;
  }

  /** Create and open a chat under this suite's directory. */
  async openChat(): Promise<Locator> {
    const filepath = `${this.dir}/chat-${UUID.uuid4()}.chat`;
    await this.page.filebrowser.contents.uploadContent('{}', 'text', filepath);
    await this.page.evaluate(async (name: string) => {
      await window.jupyterapp.commands.execute('jupyterlab-chat:open', {
        filepath: name
      });
    }, filepath);
    const tab = filepath.split('/').pop()!;
    await this.page.waitForCondition(async () =>
      this.page.activity.isTabActive(tab)
    );
    this._chat = (await this.page.activity.getPanelLocator(tab)) as Locator;
    return this._chat;
  }

  /** The persona picker button in the toolbar. */
  get personaPicker(): Locator {
    return this.chat.locator(PICKER);
  }

  /** Select a fixture persona from the picker and wait for it to take. */
  async selectPersona(persona: FixturePersona): Promise<void> {
    const { name } = FIXTURE_PERSONAS[persona];
    const picker = this.chat.locator(PICKER);
    await expect(picker).toBeVisible({ timeout: TIMEOUT });
    await picker.click();
    await this.page.getByRole('menuitem', { name }).click();
    await expect(picker).toContainText(name);
  }

  /** Pick "No one" in the persona picker, so messages go to no persona. */
  async selectNoOne(): Promise<void> {
    await expect(this.personaPicker).toBeVisible({ timeout: TIMEOUT });
    await this.personaPicker.click();
    await this.page.getByRole('menuitem', { name: 'No one' }).click();
    await expect(this.personaPicker).toContainText('No one');
  }

  /** Wait for the selected persona's session controls to render. */
  async waitForControls(): Promise<void> {
    await expect(this.chat.locator(VISIBLE_CONTROL_BTN).first()).toBeVisible({
      timeout: TIMEOUT
    });
  }

  /** Change a select control (identified by its `title`) to the given option. */
  async setControl(title: string, optionValue: string): Promise<void> {
    await this.chat.locator(`${VISIBLE_CONTROL_BTN}[title="${title}"]`).click();
    await this.page
      .getByRole('menuitem', { name: optionValue, exact: true })
      .click();
  }

  /** How many visible controls carry the given `title` (e.g. to assert a mode
   * advertised through two channels renders exactly once). */
  async controlCount(title: string): Promise<number> {
    return this.chat
      .locator(`${VISIBLE_CONTROL_BTN}[title="${title}"]`)
      .count();
  }

  /** The usage chip in the toolbar (present only once the agent reports usage). */
  get usageChip(): Locator {
    return this.chat.locator(USAGE_CHIP);
  }

  /** Wait for the usage chip to appear (the agent reported some usage). */
  async waitForUsage(): Promise<void> {
    await expect(this.usageChip).toBeVisible({ timeout: TIMEOUT });
  }

  /** Whether the chip shows a context ring (i.e. the agent reported context fill). */
  async hasUsageRing(): Promise<boolean> {
    return (await this.usageChip.locator(USAGE_RING).count()) > 0;
  }

  /** The chip's percent/token label text (e.g. "30%" or "1.5k"). */
  async usageChipText(): Promise<string> {
    return (await this.usageChip.locator(USAGE_PCT).textContent()) ?? '';
  }

  /**
   * Open the usage popover and return its card. The popover is a MUI portal at
   * the page root, so it's page-scoped rather than chat-scoped.
   */
  async openUsageCard(): Promise<Locator> {
    await this.usageChip.click();
    const card = this.page.locator(USAGE_CARD);
    await expect(card).toBeVisible({ timeout: TIMEOUT });
    return card;
  }

  /**
   * Send a message and return the text of the resulting agent reply (the latest
   * rendered message once the human message + reply have both rendered).
   */
  async sendMessage(text: string): Promise<string> {
    const before = await this.chat.locator(MESSAGE).count();
    await this.chat
      .locator(INPUT)
      .getByRole('combobox')
      .pressSequentially(text);
    await this.chat.locator(`${INPUT} .jp-chat-send-button`).click();
    // Two new rendered messages: the human echo + the agent reply.
    await expect
      .poll(async () => this.chat.locator(MESSAGE).count(), {
        timeout: TIMEOUT
      })
      .toBeGreaterThanOrEqual(before + 2);
    return (await this.chat.locator(MESSAGE).last().textContent()) ?? '';
  }

  /**
   * Send a message expecting no reply (e.g. with "No one" selected): waits
   * only for the human message to render.
   */
  async sendWithoutReply(text: string): Promise<void> {
    const before = await this.chat.locator(MESSAGE).count();
    await this.chat
      .locator(INPUT)
      .getByRole('combobox')
      .pressSequentially(text);
    await this.chat.locator(`${INPUT} .jp-chat-send-button`).click();
    await expect
      .poll(async () => this.chat.locator(MESSAGE).count(), {
        timeout: TIMEOUT
      })
      .toBeGreaterThanOrEqual(before + 1);
  }

  /**
   * Type `prefix` (e.g. "/" or "/c") into the chat input and return the slash
   * command names offered in the autocomplete popup (page-scoped, since the
   * popup is a MUI portal at the page root).
   *
   * A persona's commands arrive over awareness asynchronously after its ACP
   * session initializes, and the provider only re-queries awareness on a
   * keystroke — so this polls, clearing and retyping `prefix` each round to
   * re-trigger the query, until every name in `waitFor` is present (or it times
   * out). Pass the commands the agent advertises as `waitFor`.
   */
  async slashCompletions(
    prefix: string,
    waitFor: string[] = []
  ): Promise<string[]> {
    const combobox = this.chat.locator(INPUT).getByRole('combobox');
    await combobox.click();
    const names = this.page.locator(COMMAND_NAME);
    let last: string[] = [];
    await expect
      .poll(
        async () => {
          // Clear the previous round's text, then retype to re-query awareness.
          await combobox.press('ControlOrMeta+a');
          await combobox.press('Backspace');
          await combobox.pressSequentially(prefix);
          if ((await names.count()) === 0) {
            return false;
          }
          last = await names.allTextContents();
          return waitFor.every(w => last.includes(w));
        },
        { timeout: TIMEOUT }
      )
      .toBe(true);
    return last;
  }

  /**
   * Type a message and click send, without waiting for the reply to finish.
   * Use when the reply is long-running (e.g. a slow stream) and the test drives
   * what happens mid-stream.
   */
  async send(text: string): Promise<void> {
    await this.chat
      .locator(INPUT)
      .getByRole('combobox')
      .pressSequentially(text);
    await this.chat.locator(`${INPUT} .jp-chat-send-button`).click();
  }

  /** The toolbar's stop button (a single toolbar item, rendered once). */
  get stopButton(): Locator {
    return this.chat.locator(STOP_BUTTON);
  }

  /**
   * Wait until an AI persona is actively writing — the stop button becomes
   * enabled. Returns once the persona is streaming.
   */
  async waitForWriting(): Promise<void> {
    await expect(this.stopButton).toBeEnabled({ timeout: TIMEOUT });
  }

  /** Click the stop button to interrupt the in-progress response. */
  async clickStop(): Promise<void> {
    await this.stopButton.click();
  }

  /**
   * Wait until no AI persona is writing — the stop button is disabled again.
   * This is the awareness-driven signal that the persona stopped streaming.
   */
  async waitForNotWriting(): Promise<void> {
    await expect(this.stopButton).toBeDisabled({ timeout: TIMEOUT });
  }

  /** The text of the latest rendered message (the agent's streaming reply). */
  async lastMessageText(): Promise<string> {
    return (await this.chat.locator(MESSAGE).last().textContent()) ?? '';
  }
}
