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