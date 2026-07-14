/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 *
 * Helpers for E2E suites to declare their own fake ACP personas.
 *
 * A suite calls `installPersonas(request, testDir, [...])` from `beforeAll`. It
 * copies each named fixture persona (from ../fixtures/personas) into
 * `<testDir>/.jupyter/personas/` on the test server. The PersonaManager loads
 * the nearest `.jupyter/personas` walking up from a chat's directory, so a chat
 * created under `<testDir>/` (via `openChat`) sees only that suite's personas.
 * One shared server, per-suite isolation by directory — see AGENTS.md.
 */

import { galata, IJupyterLabPageFixture } from '@jupyterlab/galata';
import { APIRequestContext, Locator } from '@playwright/test';
import { UUID } from '@lumino/coreutils';
import * as fs from 'fs';
import * as path from 'path';

const PERSONAS_SRC = path.resolve(__dirname, '..', 'fixtures', 'personas');

/**
 * Install the named fixture personas into `<testDir>/.jupyter/personas/`.
 * Call from `beforeAll` with the worker-scoped `request` fixture.
 */
export async function installPersonas(
  request: APIRequestContext,
  testDir: string,
  personas: string[]
): Promise<void> {
  const contents = galata.newContentsHelper(request);
  for (const name of personas) {
    const source = fs.readFileSync(
      path.join(PERSONAS_SRC, `${name}_persona.py`),
      'utf-8'
    );
    await contents.uploadContent(
      source,
      'text',
      `${testDir}/.jupyter/personas/${name}_persona.py`
    );
  }
}

/**
 * Create and open a chat under `testDir` (so it loads that suite's personas),
 * returning its panel locator.
 */
export async function openChat(
  page: IJupyterLabPageFixture,
  testDir: string
): Promise<Locator> {
  const filepath = `${testDir}/chat-${UUID.uuid4()}.chat`;
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
