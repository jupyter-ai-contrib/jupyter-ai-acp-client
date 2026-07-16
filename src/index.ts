import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

import { IMessagePreambleRegistry } from '@jupyter/chat';

import { ToolCallsComponent } from './tool-calls';

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
