/**
 * Type augmentations for @jupyter/chat types not yet published to npm.
 *
 * Remove once @jupyter/chat ships IMessagePreambleRegistry, MessagePreambleProps,
 * and IMessageMetadata with tool_calls support.
 */

/* eslint-disable @typescript-eslint/naming-convention */
import { Token } from '@lumino/coreutils';
import React from 'react';
import { IChatModel } from '@jupyter/chat';

declare module '@jupyter/chat' {
  export interface IToolCall {
    tool_call_id: string;
    title: string;
    kind?: string;
    status?: string;
    raw_output?: unknown;
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
    addComponent(component: React.FC<MessagePreambleProps>): void;
    getComponents(): React.FC<MessagePreambleProps>[];
  }

  export const IMessagePreambleRegistry: Token<IMessagePreambleRegistry>;
}
