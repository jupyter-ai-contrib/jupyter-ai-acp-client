import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

import {
  IChatCommandProvider,
  IChatCommandRegistry,
  IInputModel,
  IMessagePreambleRegistry,
  ChatCommand
} from '@jupyter/chat';

import { ToolCallsComponent } from './tool-calls';

import { getAcpSlashCommands } from './request';

const SLASH_COMMAND_PROVIDER_ID =
  '@jupyter-ai/acp-client:slash-command-provider';

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
    const chatPath = inputModel.chatContext.name;
    const existingMentions = this._getExistingMentions(inputModel);

    // return early if current word doesn't start with '/'.
    if (!currentWord.startsWith('/')) {
      return [];
    }

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
  optional: [IMessagePreambleRegistry],
  activate: (
    app: JupyterFrontEnd,
    registry: IChatCommandRegistry,
    preambleRegistry: IMessagePreambleRegistry | null
  ) => {
    registry.addProvider(new SlashCommandProvider());
    if (preambleRegistry) {
      console.warn(
        '[ACP] Registered ToolCallsComponent with preamble registry'
      );
      preambleRegistry.addComponent(ToolCallsComponent);
    } else {
      console.warn(
        '[ACP] IMessagePreambleRegistry not available — tool call UI disabled'
      );
    }
  }
};

export default slashCommandPlugin;
