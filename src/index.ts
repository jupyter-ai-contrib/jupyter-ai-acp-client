import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

import {
  IChatCommandProvider,
  IChatCommandRegistry,
  IInputModel,
  IMessagePreambleRegistry,
  IInputToolbarRegistryFactory,
  InputToolbarRegistry,
  ChatCommand
} from '@jupyter/chat';

import { getAcpSlashCommands, submitPermissionDecision } from './request';
import { AcpStopButton } from './stop-button';
import { createToolCallsPreamble } from './tool-call-preamble';
import { getOpenableToolCallPath } from './tool-call-paths';

const SLASH_COMMAND_PROVIDER_ID =
  '@jupyter-ai/acp-client:slash-command-provider';
const TOOL_CALL_COMPONENTS_PLUGIN_ID =
  '@jupyter-ai/acp-client:tool-call-components';

/**
 * A command provider that provides completions for slash commands and handles
 * slash command calls.
 *
 * - Slash commands are intended for "chat-level" operations that aren't
 * specific to any persona.
 *
 * - Slash commands may only appear one-at-a-time, and only do something if the
 * first word of the input specifies a slash command `/{slash-command-id}`.
 *
 * - Note: In v2, slash commands were reserved for specific tasks like
 * 'generate' or 'learn'. But because tasks are handled by AI personas via agent
 * tools in v3, slash commands in v3 are reserved for "chat-level" operations
 * that are not specific to an AI persona.
 */
export class SlashCommandProvider implements IChatCommandProvider {
  public id: string = SLASH_COMMAND_PROVIDER_ID;

  /**
   * Regex that matches a potential slash command. The first capturing group
   * captures the ID of the slash command. Slash command IDs may be any
   * combination of: `\`, `-`.
   */
  _regex: RegExp = /\/([\w-]*)/g;

  constructor() {}

  /**
   * Returns slash command completions for the current input.
   */
  async listCommandCompletions(
    inputModel: IInputModel
  ): Promise<ChatCommand[]> {
    const currentWord = inputModel.currentWord || '';

    // return early if current word doesn't start with '/'.
    if (!currentWord.startsWith('/')) {
      return [];
    }

    if (!inputModel.chatContext) {
      return [];
    }
    const chatPath = inputModel.chatContext.name;
    const existingMentions = this._getExistingMentions(inputModel);

    // return early if >1 persona is mentioned in the input. we never show ACP
    // slash command suggestions in this case.
    if (existingMentions.size > 1) {
      return [];
    }

    // otherwise, call the `/ai/acp/slash_commands` endpoint to get slash
    // command suggestions
    let personaMentionName: string | null = null;
    if (existingMentions.size) {
      personaMentionName = existingMentions.values().next().value ?? null;
    }
    const response = await getAcpSlashCommands(chatPath, personaMentionName);
    const commandSuggestions: ChatCommand[] = [];
    for (const cmd of response) {
      // continue if command does not match current word
      if (!cmd.name.startsWith(currentWord)) {
        continue;
      }

      // otherwise add it as a suggestion
      commandSuggestions.push({
        name: cmd.name,
        providerId: this.id,
        description: cmd.description,
        spaceOnAccept: true
      });
    }

    return commandSuggestions;
  }

  /**
   * Returns the set of mention names that have already been @-mentioned in the
   * input.
   */
  _getExistingMentions(inputModel: IInputModel): Set<string> {
    const matches = inputModel.value?.matchAll(/@([\w-]*)/g);
    const existingMentions = new Set<string>();
    for (const match of matches) {
      const mention = match?.[1];
      // ignore if 1st group capturing the mention name is an empty string
      if (!mention) {
        continue;
      }
      existingMentions.add(mention);
    }

    return existingMentions;
  }

  async onSubmit(inputModel: IInputModel): Promise<void> {
    // no-op. ACP slash commands are handled by the ACP agent
    return;
  }
}

export const slashCommandPlugin: JupyterFrontEndPlugin<void> = {
  id: SLASH_COMMAND_PROVIDER_ID,
  description: 'Adds support for slash commands in Jupyter AI.',
  autoStart: true,
  requires: [IChatCommandRegistry],
  activate: (_app: JupyterFrontEnd, registry: IChatCommandRegistry) => {
    registry.addProvider(new SlashCommandProvider());
  }
};

/**
 * Plugin that renders ACP tool calls using jupyter-chat-components through the
 * chat preamble registry.
 */
export const toolCallComponentsPlugin: JupyterFrontEndPlugin<void> = {
  id: TOOL_CALL_COMPONENTS_PLUGIN_ID,
  description: 'Renders ACP grouped tool calls with jupyter-chat-components.',
  autoStart: true,
  optional: [IMessagePreambleRegistry],
  activate: (
    app: JupyterFrontEnd,
    preambleRegistry: IMessagePreambleRegistry | null
  ) => {
    if (!preambleRegistry) {
      console.warn(
        '[ACP] IMessagePreambleRegistry not available; tool call UI disabled.'
      );
      return;
    }

    preambleRegistry.addComponent(
      createToolCallsPreamble({
        openToolCallPath: (path: string) => {
          const openPath = getOpenableToolCallPath(path);

          if (!openPath) {
            return;
          }

          void app.commands.execute('docmanager:open', { path: openPath });
        },
        toolCallPermissionDecision: submitPermissionDecision
      })
    );
  }
};

/**
 * Plugin that provides a custom input toolbar factory with the ACP stop button.
 * The chat panel picks this up and uses it to build the toolbar for each chat.
 */
export const toolbarPlugin: JupyterFrontEndPlugin<IInputToolbarRegistryFactory> =
  {
    id: '@jupyter-ai/acp-client:toolbar',
    description:
      'Provides a chat input toolbar with ACP stop streaming button.',
    autoStart: true,
    provides: IInputToolbarRegistryFactory,
    activate: (): IInputToolbarRegistryFactory => {
      return {
        create: () => {
          // Start with the default toolbar (Send, Attach, Cancel, SaveEdit)
          const registry = InputToolbarRegistry.defaultToolbarRegistry();
          // Add our stop button near the beginning of the default toolbar.
          registry.addItem('stop', {
            element: AcpStopButton,
            position: 10
          });
          return registry;
        }
      };
    }
  };

export default [slashCommandPlugin, toolCallComponentsPlugin, toolbarPlugin];

export { stopStreaming } from './request';
