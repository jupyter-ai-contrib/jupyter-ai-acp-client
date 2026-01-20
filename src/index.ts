import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

import { requestAPI } from './request';

/**
 * Initialization data for the @jupyter-ai/acp-client extension.
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id: '@jupyter-ai/acp-client:plugin',
  description: 'The ACP client for Jupyter AI, allowing for ACP agents to be used in JupyterLab',
  autoStart: true,
  activate: (app: JupyterFrontEnd) => {
    console.log('JupyterLab extension @jupyter-ai/acp-client is activated!');

    requestAPI<any>('hello')
      .then(data => {
        console.log(data);
      })
      .catch(reason => {
        console.error(
          `The jupyter_ai_acp_client server extension appears to be missing.\n${reason}`
        );
      });
  }
};

export default plugin;
