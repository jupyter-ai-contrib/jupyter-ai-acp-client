import { MessagePreambleProps } from '@jupyter/chat';

import { nullTranslator } from '@jupyterlab/translation';

import {
  GroupedToolCalls,
  IToolCallDiff,
  IToolCallPermissionOption,
  IToolCallsEntry,
  OpenToolCallPath,
  ToolCallPermissionDecision
} from 'jupyter-chat-components';

import * as React from 'react';

const trans = nullTranslator.load('jupyterlab');

interface IAcpToolCallPreambleOptions {
  openToolCallPath?: OpenToolCallPath;
  toolCallPermissionDecision?: ToolCallPermissionDecision;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function readString(value: unknown): string | undefined {
  return typeof value === 'string' ? value : undefined;
}

function readStringArray(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }

  const strings = value.filter(
    (item): item is string => typeof item === 'string'
  );
  return strings.length === value.length ? strings : undefined;
}

function toPermissionOption(value: unknown): IToolCallPermissionOption | null {
  if (!isRecord(value)) {
    return null;
  }

  const optionId = readString(value.option_id);
  const name = readString(value.name);

  if (!optionId || !name) {
    return null;
  }

  const kind = readString(value.kind);

  return kind ? { optionId, name, kind } : { optionId, name };
}

function toDiff(value: unknown): IToolCallDiff | null {
  if (!isRecord(value)) {
    return null;
  }

  const path = readString(value.path);
  const newText = readString(value.new_text);

  if (!path || newText === undefined) {
    return null;
  }

  const oldText = readString(value.old_text);

  return oldText !== undefined ? { path, newText, oldText } : { path, newText };
}

function toPermissionStatus(
  value: unknown
): 'pending' | 'resolved' | undefined {
  return value === 'pending' || value === 'resolved' ? value : undefined;
}

function toToolCallsEntry(value: unknown): IToolCallsEntry | null {
  if (!isRecord(value)) {
    return null;
  }

  const toolCallId = readString(value.tool_call_id);

  if (!toolCallId) {
    return null;
  }

  const entry: IToolCallsEntry = { toolCallId };
  const title = readString(value.title);
  const kind = readString(value.kind);
  const status = readString(value.status);
  const locations = readStringArray(value.locations);
  const permissionStatus = toPermissionStatus(value.permission_status);
  const selectedOptionId = readString(value.selected_option_id);
  const sessionId = readString(value.session_id);

  if (title !== undefined) {
    entry.title = title;
  }
  if (kind !== undefined) {
    entry.kind = kind;
  }
  if (status !== undefined) {
    entry.status = status;
  }
  if (locations !== undefined) {
    entry.locations = locations;
  }
  if (permissionStatus !== undefined) {
    entry.permissionStatus = permissionStatus;
  }
  if (selectedOptionId !== undefined) {
    entry.selectedOptionId = selectedOptionId;
  }
  if (sessionId !== undefined) {
    entry.sessionId = sessionId;
  }
  if (Object.prototype.hasOwnProperty.call(value, 'raw_input')) {
    entry.rawInput = value.raw_input;
  }
  if (Object.prototype.hasOwnProperty.call(value, 'raw_output')) {
    entry.rawOutput = value.raw_output;
  }

  if (Array.isArray(value.permission_options)) {
    entry.permissionOptions = value.permission_options
      .map(toPermissionOption)
      .filter((option): option is IToolCallPermissionOption => option !== null);
  }

  if (Array.isArray(value.diffs)) {
    entry.diffs = value.diffs
      .map(toDiff)
      .filter((diff): diff is IToolCallDiff => diff !== null);
  }

  return entry;
}

function getToolCalls(
  message: MessagePreambleProps['message']
): IToolCallsEntry[] {
  if (!isRecord(message.metadata)) {
    return [];
  }

  const toolCalls = message.metadata.tool_calls;
  if (!Array.isArray(toolCalls)) {
    return [];
  }

  return toolCalls
    .map(toToolCallsEntry)
    .filter((entry): entry is IToolCallsEntry => entry !== null);
}

/**
 * Build a preamble component that renders ACP tool calls with the shared chat
 * components package.
 */
export function createToolCallsPreamble(options: IAcpToolCallPreambleOptions) {
  return function ToolCallsPreamble(
    props: MessagePreambleProps
  ): JSX.Element | null {
    const toolCalls = getToolCalls(props.message);

    if (!toolCalls.length) {
      return null;
    }

    return (
      <GroupedToolCalls
        trans={trans}
        toolCalls={toolCalls}
        openToolCallPath={options.openToolCallPath ?? null}
        toolCallPermissionDecision={options.toolCallPermissionDecision ?? null}
      />
    );
  };
}
