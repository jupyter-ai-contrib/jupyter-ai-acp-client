/**
 * Type augmentations for @jupyter/chat types that exist in the local
 * jupyter-chat build but are not yet published to npm.
 *
 * This file should be removed once @jupyter/chat is published with
 * IToolCall, IMessagePreambleRegistry, MessagePreambleProps, and
 * tool_calls on IChatMessage.
 */

/* eslint-disable @typescript-eslint/naming-convention */
import { Token } from '@lumino/coreutils';
import React from 'react';
import { IChatModel, IChatMessage, IUser, IAttachment } from '@jupyter/chat';

declare module '@jupyter/chat' {
  export interface IToolCall {
    tool_call_id: string;
    title: string;
    kind?: string;
    status?: string;
    raw_output?: unknown;
    locations?: string[];
  }

  // Augment IChatMessage with tool_calls field
  export interface IChatMessage<T = IUser, U = IAttachment> {
    tool_calls?: IToolCall[];
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
