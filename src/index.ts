import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

import { IComponentsRendererFactory } from 'jupyter-chat-components';

import { submitPermissionDecision } from './request';

const TOOL_CALL_COMPONENTS_PLUGIN_ID =
  '@jupyter-ai/acp-client:tool-call-components';

/**
 * Plugin that wires ACP-specific callbacks into jupyter-chat-components so
 * grouped tool call MIME bundles can open files and resolve permissions.
 */
export const toolCallComponentsPlugin: JupyterFrontEndPlugin<void> = {
  id: TOOL_CALL_COMPONENTS_PLUGIN_ID,
  description:
    'Connects ACP grouped tool call actions to jupyter-chat-components.',
  autoStart: true,
  requires: [IComponentsRendererFactory],
  activate: (
    app: JupyterFrontEnd,
    componentsRendererFactory: IComponentsRendererFactory
  ) => {
    componentsRendererFactory.addCallbacks({
      toolCallPermissionDecision: submitPermissionDecision,
      openToolCallPath: (path: string) => {
        // The component sends a server-relative path. A leading '/' means
        // the file is outside the server root and cannot be opened.
        if (path.startsWith('/')) {
          return;
        }

        app.commands
          .execute('docmanager:open', { path })
          .catch((error: unknown) => {
            console.error(`Failed to open tool call path: ${path}`, error);
          });
      }
    });
  }
};

export default [toolCallComponentsPlugin];
