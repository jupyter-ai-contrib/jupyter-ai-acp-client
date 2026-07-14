/**
 * Building the message metadata that carries a user's persona/model/settings
 * selection.
 *
 * Selections are per-user and never touch shared state: when the user picks a
 * persona, model, model setting, or general setting, the choice is stamped onto
 * the outgoing message's metadata. The `PersonaManager` reads `to_persona` to
 * route the message, and `BasePersona.apply_message_metadata` applies the model
 * and settings specs before processing.
 *
 * The shape mirrors the persona-manager `ModelSpec` and the message-metadata
 * contract in the issue:
 *
 *   type ModelSpec = { id: string | null; settings: Record<string, string | null> }
 *   type MessageMetadata = {
 *     to_persona: string | null;
 *     model: ModelSpec;
 *     settings: Record<string, string | null>;
 *   }
 *
 * A `null` value for the model id or any setting means "use the persona's
 * current/default value", so the persona leaves it untouched.
 */

import { IMessageMetadata } from '@jupyter/chat';
import { PersonaAwarenessState } from './awareness';

/**
 * A user's per-message selection: which persona to address, and (relative to
 * that persona's current values) which model, model settings, and general
 * settings to use. A null anywhere means "keep the persona's current value".
 */
export type PersonaSelection = {
  personaId: string | null;
  modelId: string | null;
  modelSettings: { [id: string]: string | null };
  settings: { [id: string]: string | null };
};

/**
 * A selection that changes nothing: no persona, and every value left at the
 * persona's default.
 */
export function emptySelection(personaId: string | null): PersonaSelection {
  return {
    personaId,
    modelId: null,
    modelSettings: {},
    settings: {}
  };
}

/**
 * Seed a selection for a persona from its current awareness state: every
 * control starts at "default" (null), which the UI renders as the persona's
 * current value. This is the initial per-user selection — the user only
 * diverges from it by picking a non-default option.
 */
export function selectionForPersona(
  personaId: string | null,
  state: PersonaAwarenessState | null
): PersonaSelection {
  const selection = emptySelection(personaId);
  if (!state) {
    return selection;
  }
  // Initialize the settings keys so the picker knows which controls exist, all
  // at the default (null) value.
  for (const setting of state.model.settings) {
    selection.modelSettings[setting.id] = null;
  }
  for (const setting of state.settings) {
    selection.settings[setting.id] = null;
  }
  return selection;
}

/**
 * Build the message metadata for a selection. With no persona selected ("no
 * one"), only `to_persona` is stamped — there is nothing to configure. The
 * model spec and settings are always included for a real persona so a message
 * fully describes the selection it was sent with (a null value means default).
 */
export function buildSelectionMetadata(
  selection: PersonaSelection
): IMessageMetadata {
  const metadata: IMessageMetadata = { to_persona: selection.personaId };
  if (!selection.personaId) {
    return metadata;
  }
  metadata.model = {
    id: selection.modelId,
    settings: { ...selection.modelSettings }
  };
  metadata.settings = { ...selection.settings };
  return metadata;
}
