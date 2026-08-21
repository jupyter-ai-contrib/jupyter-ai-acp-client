import { Event } from '@jupyterlab/services';

export const PERMISSION_RESPONSE_EVENT_SCHEMA_ID =
  'https://schema.jupyter.org/jupyter_ai_persona_manager/permission_response/v1';

/**
 * Send the user's permission decision to the server by emitting a
 * `permission_response` Jupyter Event (client -> server). This replaces the
 * previous bespoke REST endpoint: the request lifecycle is now owned by
 * jupyter-ai-persona-manager, which routes the event by (chat_id, persona_id,
 * request_id) to the requesting persona.
 */
export async function submitPermissionDecision(
  events: Event.IManager,
  ids: { chat_id: string; persona_id: string; request_id: string },
  optionId: string | null
): Promise<void> {
  await events.emit({
    schema_id: PERMISSION_RESPONSE_EVENT_SCHEMA_ID,
    version: '1',
    data: {
      chat_id: ids.chat_id,
      persona_id: ids.persona_id,
      request_id: ids.request_id,
      option_id: optionId
    }
  });
}
