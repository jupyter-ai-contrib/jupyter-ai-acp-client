import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

import { IComponentsRendererFactory } from 'jupyter-chat-components';

import { submitPermissionDecision } from './request';
import { getOpenableToolCallPath } from './tool-call-paths';

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
    componentsRendererFactory.groupedToolCallCallbacks = {
      toolCallPermissionDecision: submitPermissionDecision,
      openToolCallPath: (path: string) => {
        const openPath = getOpenableToolCallPath(path);

        if (!openPath) {
          return;
        }

        void app.commands.execute('docmanager:open', { path: openPath });
      }
    };
  }
};

export default [toolCallComponentsPlugin];
