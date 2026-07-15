import { URLExt } from '@jupyterlab/coreutils';

import { ServerConnection } from '@jupyterlab/services';

/**
 * Call a server extension endpoint under the given API namespace.
 *
 * @param namespace API namespace, e.g. 'ai/acp' (this extension) or 'api/ai'
 *   (the persona-manager extension).
 * @param endPoint API REST endpoint, appended to the namespace.
 * @param init Initial values for the request.
 * @returns The response body interpreted as JSON.
 */
export async function requestAPI<T>(
  namespace: string,
  endPoint = '',
  init: RequestInit = {}
): Promise<T> {
  // Make request to Jupyter API
  const settings = ServerConnection.makeSettings();
  const requestUrl = URLExt.join(settings.baseUrl, namespace, endPoint);

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

/**
 * Send the user's permission decision to the backend.
 */
export async function submitPermissionDecision(
  sessionId: string,
  toolCallId: string,
  optionId: string
): Promise<void> {
  await requestAPI('ai/acp', 'permissions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      tool_call_id: toolCallId,
      option_id: optionId
    })
  });
}

/**
 * Interrupt/stop an in-progress agent response. With no persona name, the
 * backend stops all personas in the chat.
 */
export async function stopStreaming(
  chatPath: string,
  personaMentionName: string | null = null
): Promise<void> {
  try {
    if (personaMentionName === null) {
      await requestAPI('ai/acp', `stop?chat_path=${chatPath}`, {
        method: 'POST'
      });
    } else {
      await requestAPI(
        'ai/acp',
        `stop/${personaMentionName}?chat_path=${chatPath}`,
        {
          method: 'POST'
        }
      );
    }
  } catch (e) {
    console.warn('Error stopping stream: ', e);
  }
}
