/**
 * Building the message metadata that carries a user's persona/model/settings
 * selection.
 *
 * Selections are per-user and never touch shared state: when the user picks a
 * persona, model, model setting, or general setting, the choice is stamped onto
 * the outgoing message's metadata. The `PersonaManager` reads `to_persona` to
 * route the message, and `BasePersona.apply_specs_in_message` applies the model
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

/**
 * The settings a user has chosen for one persona: which model, model settings,
 * and general settings to use, relative to that persona's current values. A key
 * absent (or null) means "keep the persona's current value". Which persona this
 * applies to is tracked separately (the cache key / `to_persona`), so it isn't
 * repeated here.
 */
export type PersonaSettings = {
  modelId: string | null;
  modelSettings: { [id: string]: string | null };
  settings: { [id: string]: string | null };
};

/**
 * Empty settings: every control left at "default" (the persona's current
 * value). The user diverges from this by picking a non-default option; controls
 * the user hasn't touched simply stay absent.
 */
export function emptyPersonaSettings(): PersonaSettings {
  return { modelId: null, modelSettings: {}, settings: {} };
}

/**
 * Build the message metadata for a persona and the user's settings for it. With
 * no persona ("no one"), only `to_persona` is stamped — nothing to configure.
 * For a real persona the model spec and settings are always included so a
 * message fully describes the selection it was sent with (null = default).
 */
export function buildMessageMetadata(
  personaId: string | null,
  settings: PersonaSettings
): IMessageMetadata {
  const metadata: IMessageMetadata = { to_persona: personaId };
  if (!personaId) {
    return metadata;
  }
  metadata.model = {
    id: settings.modelId,
    settings: { ...settings.modelSettings }
  };
  metadata.settings = { ...settings.settings };
  return metadata;
}
