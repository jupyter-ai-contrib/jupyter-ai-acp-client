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
