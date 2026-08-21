import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

import { IMessagePreambleRegistry } from '@jupyter/chat';

import { ToolCallsComponent, setPermissionEventManager } from './tool-calls';

/**
 * Plugin registering the ACP tool-call UI (tool calls, permission requests,
 * diffs) with the message preamble registry, so it renders above agent
 * messages.
 */
export const toolCallsPlugin: JupyterFrontEndPlugin<void> = {
  id: '@jupyter-ai/acp-client:tool-calls',
  description: 'Renders ACP tool calls in chat message preambles.',
  autoStart: true,
  optional: [IMessagePreambleRegistry],
  activate: (
    app: JupyterFrontEnd,
    preambleRegistry: IMessagePreambleRegistry | null
  ) => {
    // The permission buttons emit a `permission_response` Jupyter Event; give
    // the renderer the event manager to emit with.
    setPermissionEventManager(app.serviceManager.events);
    if (preambleRegistry) {
      preambleRegistry.addComponent(ToolCallsComponent);
    } else {
      console.warn(
        '[ACP] IMessagePreambleRegistry not available — tool call UI disabled'
      );
    }
  }
};

export default [toolCallsPlugin];
