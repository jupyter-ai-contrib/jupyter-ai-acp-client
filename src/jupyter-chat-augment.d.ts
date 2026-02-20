/**
 * Type augmentation for @jupyter/chat.
 *
 * Extends IMessageMetadata with ACP-specific tool call fields.
 * Remove IChatMessage and MessagePreambleProps blocks once @jupyter/chat
 * ships IMessageMetadata and IMessagePreambleRegistry.
 */

/* eslint-disable @typescript-eslint/naming-convention */
import { Token } from '@lumino/coreutils';
import { IChatModel } from '@jupyter/chat';

declare module '@jupyter/chat' {
  export interface IToolCall {
    /**
     * Unique identifier for this tool call, used to correlate events
     * across the tool call lifecycle.
     */
    tool_call_id: string;
    /**
     * Human-readable label displayed in the message preamble.
     */
    title: string;
    /**
     * The category of tool operation.
     */
    kind?:
      | 'read'
      | 'edit'
      | 'delete'
      | 'move'
      | 'search'
      | 'execute'
      | 'think'
      | 'fetch'
      | 'switch_mode'
      | (string & {});
    /**
     * Current execution status.
     */
    status?: 'in_progress' | 'completed' | (string & {});
    /**
     * Raw return value from tool execution.
     */
    raw_output?: unknown;
    /**
     * File paths or resource URIs involved in this tool call.
     */
    locations?: string[];
  }

  export interface IMessageMetadata {
    tool_calls?: IToolCall[];
  }

  export interface IChatMessage {
    metadata?: IMessageMetadata;
  }

  export type MessagePreambleProps = {
    model: IChatModel;
    message: IChatMessage;
  };

  export interface IMessagePreambleRegistry {
    addComponent(
      component: (props: MessagePreambleProps) => JSX.Element | null
    ): void;
    getComponents(): ((props: MessagePreambleProps) => JSX.Element | null)[];
  }

  export const IMessagePreambleRegistry: Token<IMessagePreambleRegistry>;
}
