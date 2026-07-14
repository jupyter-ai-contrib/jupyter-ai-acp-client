import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

import {
  IChatCommandProvider,
  IChatCommandRegistry,
  IChatContext,
  IInputModel,
  IMessagePreambleRegistry,
  IInputToolbarRegistryFactory,
  InputToolbarRegistry,
  ChatCommand
} from '@jupyter/chat';

import { IAwareness } from '@jupyter/ydoc';
import {
  PersonaAwareness,
  PersonaManagerAwareness
} from '@jupyter-ai/persona-manager';

import { ToolCallsComponent } from './tool-calls';

import { AcpStopButton } from './stop-button';
import { AcpPersonaControls } from './persona-controls';

/**
 * Reach the Yjs awareness channel through the input's chat context. The
 * concrete `LabChatContext` wraps the `LabChatModel`, whose `sharedModel`
 * carries the `awareness` object; the generic `IChatContext` type does not
 * surface either, so we read them structurally.
 */
function getAwarenessFromContext(
  chatContext: IChatContext | undefined
): IAwareness | null {
  const model = (chatContext as { _model?: unknown })?._model;
  const shared = (model as { sharedModel?: { awareness?: IAwareness } })
    ?.sharedModel;
  return shared?.awareness ?? null;
}

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
   *
   * Slash commands are read from the selected persona's awareness slot — the
   * same source the toolbar reads — with no REST call. The target persona is the
   * one the picker stamped onto the input metadata as `to_persona`.
   */
  async listCommandCompletions(
    inputModel: IInputModel
  ): Promise<ChatCommand[]> {
    const currentWord = inputModel.currentWord || '';

    // return early if current word doesn't start with '/'.
    if (!currentWord.startsWith('/')) {
      return [];
    }

    const awareness = getAwarenessFromContext(inputModel.chatContext);
    if (!awareness) {
      return [];
    }

    const personaId = inputModel.getMetadata?.().to_persona ?? null;
    if (!personaId) {
      return [];
    }

    // `from()` resolves immediately when the manager is already registered
    // (the normal case). If it never registers, this rejects after a bounded
    // wait; a late/failed completion query is harmless.
    let manager: PersonaManagerAwareness;
    try {
      manager = await PersonaManagerAwareness.from(awareness);
    } catch {
      return [];
    }
    const option = manager.personas.find(p => p.id === personaId);
    if (!option) {
      return [];
    }
    const persona = PersonaAwareness.from(awareness, option);

    const commandSuggestions: ChatCommand[] = [];
    for (const cmd of persona.slash_commands) {
      const name = cmd.name.startsWith('/') ? cmd.name : `/${cmd.name}`;
      // continue if command does not match current word
      if (!name.startsWith(currentWord)) {
        continue;
      }
      commandSuggestions.push({
        name,
        providerId: this.id,
        description: cmd.description ?? undefined,
        spaceOnAccept: true
      });
    }

    return commandSuggestions;
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
          // Add the active-persona controls (persona + model), leftmost.
          registry.addItem('persona', {
            element: AcpPersonaControls,
            position: 5
          });
          // Add our stop button (position 90 = just before Send at 100)
          registry.addItem('stop', {
            element: AcpStopButton,
            position: 10
          });
          return registry;
        }
      };
    }
  };

export default [slashCommandPlugin, toolbarPlugin];

export { stopStreaming } from './request';
