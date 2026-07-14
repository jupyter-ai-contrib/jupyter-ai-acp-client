/**
 * Shared test helper: build a real `PersonaAwareness` view over a plain
 * in-memory awareness slot, so specs exercise the actual persona-manager class
 * (its getters) rather than a hand-rolled shape.
 */

import { IAwareness } from '@jupyter/ydoc';
import { PersonaAwareness, PersonaOption } from '@jupyter-ai/persona-manager';

const CLIENT_ID = 42;

const OPTION: PersonaOption = {
  id: 'kiro',
  name: 'Kiro',
  avatar_url: null,
  yjs_client_id: CLIENT_ID
};

/**
 * A `PersonaAwareness` whose slot fields are the given partial (merged over
 * empty defaults). Pass `null` for "no persona / empty slot".
 */
export function personaAwareness(
  slot: Record<string, unknown> | null
): PersonaAwareness | null {
  if (slot === null) {
    return null;
  }
  const state = {
    id: 'kiro',
    model: { current: null, options: [], settings: [] },
    settings: [],
    usage: {},
    slash_commands: [],
    isWriting: false,
    ...slot
  };
  const awareness = {
    getStates: () => new Map<number, unknown>([[CLIENT_ID, state]])
  } as unknown as IAwareness;
  return PersonaAwareness.from(awareness, OPTION);
}
