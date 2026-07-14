/**
 * Reading persona session information from the chat's Yjs awareness channel.
 *
 * The persona-manager server extension broadcasts session info over awareness
 * instead of REST:
 *
 * - Under a fixed, hardcoded client ID, the `PersonaManager` publishes a
 *   `PersonaManagerAwarenessState` (the list of personas in the chat). A
 *   readiness endpoint hands the browser that client ID once the manager is
 *   registered; we cache it and read the persona list from that slot.
 *
 * - Under each persona's own (dynamic) Yjs client ID, the persona publishes a
 *   `PersonaAwarenessState` (its model configuration, settings, usage, and
 *   slash commands). The manager's persona list reports each persona's client
 *   ID so we can look its state up in O(1).
 *
 * This module mirrors those Pydantic models as TypeScript types and provides
 * helpers to read them out of an awareness `getStates()` map. All selections
 * ride on message metadata (see `metadata.ts`); nothing here writes awareness.
 */

import { Awareness } from 'y-protocols/awareness';

/**
 * A selectable model. Mirrors `ModelOption` in persona-manager.
 */
export type ModelOption = {
  id: string;
  name: string | null;
  description: string | null;
};

/**
 * A selectable value for a setting. Mirrors `SettingOption`.
 */
export type SettingOption = {
  id: string;
  name: string | null;
  description: string | null;
};

/**
 * A single setting: its current value and all options. Used both for model
 * settings (rendered near the model picker) and general settings. Mirrors
 * `SettingConfiguration`. `current` is null when the persona's default is in
 * effect.
 */
export type SettingConfiguration = {
  id: string;
  current: string | null;
  name: string | null;
  description: string | null;
  options: SettingOption[];
};

/**
 * The persona's current model, its options, and its model settings. Mirrors
 * `ModelConfiguration`.
 */
export type ModelConfiguration = {
  current: string | null;
  options: ModelOption[];
  settings: SettingConfiguration[];
};

/**
 * Token and cost usage reported by a persona. Mirrors `Usage`. Every field is
 * null until the persona reports it.
 */
export type Usage = {
  context_tokens: number | null;
  context_size: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  cached_read_tokens: number | null;
  cached_write_tokens: number | null;
  thought_tokens: number | null;
  total_tokens: number | null;
  cost_amount: number | null;
  cost_currency: string | null;
};

/**
 * One slash command advertised by a persona. Mirrors `CommandOption`.
 */
export type CommandOption = {
  name: string;
  description: string | null;
};

/**
 * A single persona's awareness state. Mirrors `PersonaAwarenessState`, and is
 * the persona's awareness slot itself — each field is a top-level entry of the
 * slot. `isWriting` (false when idle, else the ID of the message being written)
 * is written on the streaming hot path and read by jupyter-chat; other slot
 * entries (e.g. the `user` object) are not part of this state.
 */
export type PersonaAwarenessState = {
  id: string;
  model: ModelConfiguration;
  settings: SettingConfiguration[];
  usage: Usage;
  slash_commands: CommandOption[];
  isWriting: boolean | string;
};

/**
 * One persona in the chat, as advertised by the manager. Mirrors
 * `PersonaOption`.
 */
export type PersonaOption = {
  id: string;
  name: string;
  avatar_url: string | null;
  /** The Yjs client ID of this persona's awareness slot. */
  yjs_client_id: number;
};

export const EMPTY_USAGE: Usage = {
  context_tokens: null,
  context_size: null,
  input_tokens: null,
  output_tokens: null,
  cached_read_tokens: null,
  cached_write_tokens: null,
  thought_tokens: null,
  total_tokens: null,
  cost_amount: null,
  cost_currency: null
};

/**
 * The awareness field the manager writes its persona list under. Kept in sync
 * with the persona-manager `PersonaManagerAwarenessState`.
 */
const MANAGER_PERSONAS_FIELD = 'personas';

/**
 * Read the persona list the `PersonaManager` published under `managerClientId`.
 * Returns an empty list until the manager slot is present.
 */
export function readPersonaList(
  awareness: Awareness,
  managerClientId: number
): PersonaOption[] {
  const state = awareness.getStates().get(managerClientId);
  const personas = state?.[MANAGER_PERSONAS_FIELD];
  return Array.isArray(personas) ? (personas as PersonaOption[]) : [];
}

/**
 * Read a single persona's `PersonaAwarenessState` from the awareness slot named
 * by its `yjs_client_id`. The slot itself is the state (each field is a
 * top-level entry). Returns null when the slot is absent or has no `model` yet
 * (the persona has not published its configuration, or reloaded under a new
 * client ID).
 */
export function readPersonaState(
  awareness: Awareness,
  yjsClientId: number
): PersonaAwarenessState | null {
  const state = awareness.getStates().get(yjsClientId);
  // `model` is present once the persona has broadcast its configuration; use it
  // to tell a populated persona slot from an empty/other client's slot.
  return state && state.model ? (state as PersonaAwarenessState) : null;
}

/**
 * Find the manager's persona list without a cached client ID, by scanning
 * awareness for the slot that carries a persona list. Used where the manager's
 * client ID isn't already known (e.g. the stateless slash-command provider);
 * the toolbar, which caches the client ID from the readiness endpoint, uses
 * `readPersonaList` directly.
 */
export function findPersonaList(awareness: Awareness): PersonaOption[] {
  for (const state of awareness.getStates().values()) {
    const personas = state?.[MANAGER_PERSONAS_FIELD];
    if (Array.isArray(personas)) {
      return personas as PersonaOption[];
    }
  }
  return [];
}

/**
 * Read a persona's `PersonaAwarenessState` given its persona ID, resolving the
 * Yjs client ID through the manager's persona list. Returns null when the
 * persona or its state is not present in awareness.
 */
export function readPersonaStateById(
  awareness: Awareness,
  personaId: string
): PersonaAwarenessState | null {
  const persona = findPersonaList(awareness).find(p => p.id === personaId);
  if (!persona) {
    return null;
  }
  return readPersonaState(awareness, persona.yjs_client_id);
}
