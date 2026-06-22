import { URLExt } from '@jupyterlab/coreutils';

import { ServerConnection } from '@jupyterlab/services';

/**
 * Call the server extension
 *
 * @param endPoint API REST end point for the extension
 * @param init Initial values for the request
 * @returns The response body interpreted as JSON
 */
export async function requestAPI<T>(
  endPoint = '',
  init: RequestInit = {}
): Promise<T> {
  // Make request to Jupyter API
  const settings = ServerConnection.makeSettings();
  const requestUrl = URLExt.join(
    settings.baseUrl,
    'ai/acp', // our server extension's API namespace
    endPoint
  );

  let response: Response;
  try {
    response = await ServerConnection.makeRequest(requestUrl, init, settings);
  } catch (error) {
    throw new ServerConnection.NetworkError(error as any);
  }

  let data: any = await response.text();

  if (data.length > 0) {
    try {
      data = JSON.parse(data);
    } catch (error) {
      console.log('Not a JSON response body.', response);
    }
  }

  if (!response.ok) {
    throw new ServerConnection.ResponseError(response, data.message || data);
  }

  return data;
}

type AcpSlashCommand = {
  name: string;
  description: string;
};

type AcpSlashCommandsResponse = {
  commands: AcpSlashCommand[];
};

export async function getAcpSlashCommands(
  chatPath: string,
  personaMentionName: string | null = null
): Promise<AcpSlashCommand[]> {
  let response: AcpSlashCommandsResponse;
  try {
    if (personaMentionName === null) {
      response = await requestAPI(`/slash_commands?chat_path=${chatPath}`);
    } else {
      response = await requestAPI(
        `/slash_commands/${personaMentionName}?chat_path=${chatPath}`
      );
    }
  } catch (e) {
    console.warn('Error retrieving ACP slash commands: ', e);
    return [];
  }

  return response.commands;
}
export type AcpControlChoice = {
  value: string;
  label: string;
  description: string | null;
};

export type AcpControl = {
  id: string;
  source: 'model' | 'mode' | 'config_option';
  kind: 'select' | 'boolean';
  label: string;
  current_value: string | boolean | null;
  choices: AcpControlChoice[];
};

export type ActivePersonaInfo = {
  id: string;
  name: string;
  mention_name: string;
  is_acp: boolean;
  avatar_url: string | null;
};

export type ActivePersonaResponse = {
  personas: ActivePersonaInfo[];
  active_id: string | null;
  active_name: string | null;
  controls: AcpControl[];
};

/**
 * Fetch the chat's personas, which one is active, and the active persona's
 * session controls, in one call.
 */
export async function getActivePersona(
  chatPath: string
): Promise<ActivePersonaResponse> {
  const empty: ActivePersonaResponse = {
    personas: [],
    active_id: null,
    active_name: null,
    controls: []
  };
  try {
    return await requestAPI<ActivePersonaResponse>(
      `/active_persona?chat_path=${encodeURIComponent(chatPath)}`
    );
  } catch (e) {
    console.warn('Error retrieving active persona: ', e);
    return empty;
  }
}

/**
 * Set the active persona (who replies). A null personaId means "no one".
 */
export async function setActivePersona(
  chatPath: string,
  personaId: string | null
): Promise<void> {
  try {
    await requestAPI(
      `/active_persona?chat_path=${encodeURIComponent(chatPath)}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ persona_id: personaId })
      }
    );
  } catch (e) {
    console.warn('Error setting active persona: ', e);
  }
}

/**
 * Set a session control (model, mode, or config option) on the chat's active
 * ACP persona.
 */
export async function setAcpControl(
  chatPath: string,
  controlId: string,
  source: string,
  value: string | boolean
): Promise<void> {
  try {
    await requestAPI(`/control?chat_path=${encodeURIComponent(chatPath)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ control_id: controlId, source, value })
    });
  } catch (e) {
    console.warn('Error setting ACP control: ', e);
  }
}

/**
 * Send the user's permission decision to the backend.
 */
export async function submitPermissionDecision(
  sessionId: string,
  toolCallId: string,
  optionId: string
): Promise<void> {
  await requestAPI('/permissions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      tool_call_id: toolCallId,
      option_id: optionId
    })
  });
}

export async function stopStreaming(
  chatPath: string,
  personaMentionName: string | null = null
): Promise<void> {
  try {
    if (personaMentionName === null) {
      await requestAPI(`/stop?chat_path=${chatPath}`, { method: 'POST' });
    } else {
      await requestAPI(`/stop/${personaMentionName}?chat_path=${chatPath}`, {
        method: 'POST'
      });
    }
  } catch (e) {
    console.warn('Error stopping stream: ', e);
  }
}
