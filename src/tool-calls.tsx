import React from 'react';
import {
  IToolCall,
  IPermissionOption,
  MessagePreambleProps
} from '@jupyter/chat';
import { submitPermissionDecision } from './request';
import clsx from 'clsx';
import { DiffView } from './diff-view';

/**
 * Preamble component that renders tool call status lines above message body.
 * Returns null if the message has no tool calls.
 */
export function ToolCallsComponent(
  props: MessagePreambleProps
): JSX.Element | null {
  const { message, model } = props;
  if (!message.metadata?.tool_calls?.length) {
    return null;
  }

  const onOpenFile = (path: string) => {
    model.documentManager?.openOrReveal(path);
  };

  return (
    <div className="jp-jupyter-ai-acp-client-tool-calls">
      {(message.metadata?.tool_calls ?? []).map((tc: IToolCall) => (
        <ToolCallLine
          key={tc.tool_call_id}
          toolCall={tc}
          onOpenFile={onOpenFile}
        />
      ))}
    </div>
  );
}

/**
 * Format raw_output for display. Handles string, object, and array values.
 */
function formatOutput(rawOutput: unknown): string {
  if (typeof rawOutput === 'string') {
    return rawOutput;
  }
  // Extract text from ACP content block arrays: [{type, text}, ...]
  if (Array.isArray(rawOutput) && rawOutput.every(i => i?.text)) {
    return rawOutput.map(i => i.text).join('\n');
  }
  return JSON.stringify(rawOutput, null, 2);
}

/**
 * Format tool input for display. Flat objects (all primitive values) render as
 * clean key-value pairs; nested/complex values fall back to JSON.
 */
function formatToolInput(input: unknown): string {
  if (typeof input === 'string') {
    return input;
  }
  if (typeof input !== 'object' || input === null || Array.isArray(input)) {
    return JSON.stringify(input, null, 2);
  }
  const entries = Object.entries(input as Record<string, unknown>);
  const isFlat = entries.every(
    ([, v]) =>
      typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean'
  );
  if (isFlat) {
    return entries.map(([k, v]) => `${k}: ${v}`).join('\n');
  }
  return JSON.stringify(input, null, 2);
}

/**
 * Compute the pre-permission detail text for a tool call, or null if nothing
 * to show beyond the title. Returns a plain string so callers can check null.
 *
 * Priority: kind-specific rendering (locations for file ops, raw_input.command
 * for execute) → key-value pairs for unknown/MCP tools → null.
 */
function buildPermissionDetail(toolCall: IToolCall): string | null {
  const { kind, title, locations, raw_input } = toolCall;

  if (kind === 'execute') {
    // Prefer raw_input.command (ACP-compliant agents); fall back to stripping
    // "Running: " from title (Kiro-specific fallback).
    const rawObj =
      typeof raw_input === 'object' && raw_input !== null
        ? (raw_input as Record<string, unknown>)
        : null;
    const cmd =
      rawObj && typeof rawObj.command === 'string'
        ? rawObj.command
        : title
            ?.replace(/^Running:\s*/i, '')
            .replace(/\.\.\.$/, '')
            .trim() || null;
    // Guard: if stripping produced nothing new, don't show.
    if (!cmd || cmd === title) {
      return null;
    }
    return '$ ' + cmd;
  }

  if (
    (kind === 'delete' || kind === 'move' || kind === 'read') &&
    locations?.length
  ) {
    return kind === 'move' && locations.length >= 2
      ? locations[0] + '  \u2192  ' + locations[1]
      : locations.join('\n');
  }

  // Generic fallback for unknown/MCP kinds with raw_input.
  if (
    raw_input !== null &&
    typeof raw_input === 'object' &&
    !Array.isArray(raw_input)
  ) {
    const obj = raw_input as Record<string, unknown>;

    // Extract __tool_use_purpose — agent's plain-English intent for the tool call.
    const purpose =
      typeof obj.__tool_use_purpose === 'string'
        ? obj.__tool_use_purpose
        : null;

    // Filter remaining __-prefixed internal keys for the params display.
    const paramEntries = Object.entries(obj).filter(
      ([k]) => !k.startsWith('__')
    );
    const params =
      paramEntries.length > 0
        ? formatToolInput(Object.fromEntries(paramEntries))
        : null;

    // Compose: purpose + params, or whichever is available.
    if (purpose && params) {
      return purpose + '\n' + params;
    }
    if (purpose) {
      return purpose;
    }
    if (params) {
      return params;
    }
    return null;
  }

  // Non-object raw_input (string, array, primitive) — pass through.
  if (raw_input !== null && raw_input !== undefined) {
    return formatToolInput(raw_input);
  }

  return null;
}

/** Tool kinds where expanded view shows full file path(s) from locations. */
const FILE_KINDS = new Set(['read', 'edit', 'delete', 'move']);

/** Tool kinds where expanded view shows raw_output (stdout, search results, etc.). */
const OUTPUT_KINDS = new Set(['search', 'execute', 'think', 'fetch']);

/**
 * Build the expandable details content for a tool call.
 * Returns lines of metadata to display, or empty array if nothing to show.
 *
 * File operations show full paths; output operations show raw_output;
 * switch_mode/other/None show nothing (clean title only).
 */
function buildDetailsLines(toolCall: IToolCall): string[] {
  const lines: string[] = [];
  const kind = toolCall.kind;

  if (kind && FILE_KINDS.has(kind) && toolCall.locations?.length) {
    for (const loc of toolCall.locations) {
      lines.push(loc);
    }
  } else if (kind && OUTPUT_KINDS.has(kind) && toolCall.raw_output) {
    lines.push(formatOutput(toolCall.raw_output));
  } else if (toolCall.raw_output && typeof toolCall.raw_output === 'string') {
    // Fallback: show raw_output only if it's a plain string
    lines.push(toolCall.raw_output);
  }

  return lines;
}

/**
 * Renders a single tool call line with status icon and optional expandable output.
 */
function ToolCallLine({
  toolCall,
  onOpenFile
}: {
  toolCall: IToolCall;
  onOpenFile?: (path: string) => void;
}): JSX.Element {
  const { title, status, kind } = toolCall;
  const displayTitle =
    title ||
    (kind
      ? `${kind.charAt(0).toUpperCase()}${kind.slice(1)}...`
      : 'Working...');
  const selectedOpt = toolCall.permission_options?.find(
    opt => opt.option_id === toolCall.selected_option_id
  );
  const isRejected =
    toolCall.permission_status === 'resolved' &&
    !!selectedOpt?.kind?.includes('reject');
  const hasPendingPermission = toolCall.permission_status === 'pending';
  const isInProgress =
    !isRejected &&
    (status === 'in_progress' || status === 'pending' || hasPendingPermission);
  const isCompleted = status === 'completed';
  const isFailed = status === 'failed' || isRejected;

  // Unicode text glyphs — consistent across OS/browser
  const icon = isInProgress
    ? '\u2022'
    : isCompleted
      ? '\u2713'
      : isFailed
        ? '\u2717'
        : '\u2022';
  // Force 'failed' class when rejected
  const effectiveStatus = isRejected ? 'failed' : status || 'in_progress';

  const cssClass = clsx(
    'jp-jupyter-ai-acp-client-tool-call',
    `jp-jupyter-ai-acp-client-tool-call-${effectiveStatus}`
  );

  const hasDiffs = !!toolCall.diffs?.length;

  // Pending permission with diffs: expanded diff + permission buttons outside
  if (hasDiffs && hasPendingPermission) {
    return (
      <div className={cssClass}>
        <details open>
          <summary>
            <span className="jp-jupyter-ai-acp-client-tool-call-icon">
              {icon}
            </span>{' '}
            <em>{displayTitle}</em>
          </summary>
          <DiffView diffs={toolCall.diffs!} onOpenFile={onOpenFile} />
        </details>
        <PermissionButtons toolCall={toolCall} />
      </div>
    );
  }

  // Pending permission without diffs: show kind-specific detail if available
  if (!hasDiffs && hasPendingPermission) {
    const permissionDetail = buildPermissionDetail(toolCall);
    if (permissionDetail !== null) {
      return (
        <div className={cssClass}>
          <details open>
            <summary>
              <span className="jp-jupyter-ai-acp-client-tool-call-icon">
                {icon}
              </span>{' '}
              <em>{displayTitle}</em>
            </summary>
            <div className="jp-jupyter-ai-acp-client-tool-call-detail">
              {permissionDetail}
            </div>
          </details>
          <PermissionButtons toolCall={toolCall} />
        </div>
      );
    }
  }

  // Completed/failed with expandable content (diffs or metadata)
  const detailsLines =
    !hasDiffs && (isCompleted || isFailed) ? buildDetailsLines(toolCall) : [];
  const hasExpandableContent = hasDiffs || detailsLines.length > 0;

  if ((isCompleted || isFailed) && hasExpandableContent) {
    return (
      <details className={cssClass}>
        <summary>
          <span className="jp-jupyter-ai-acp-client-tool-call-icon">
            {icon}
          </span>{' '}
          {displayTitle}
          <PermissionLabel toolCall={toolCall} />
        </summary>
        {hasDiffs ? (
          <DiffView diffs={toolCall.diffs!} onOpenFile={onOpenFile} />
        ) : (
          <div className="jp-jupyter-ai-acp-client-tool-call-detail">
            {detailsLines.join('\n')}
          </div>
        )}
      </details>
    );
  }

  // In-progress — italic
  if (isInProgress) {
    return (
      <div className={cssClass}>
        <span className="jp-jupyter-ai-acp-client-tool-call-icon">{icon}</span>{' '}
        <em>{displayTitle}</em>
        <PermissionButtons toolCall={toolCall} />
      </div>
    );
  }

  // Completed/failed without metadata
  return (
    <div className={cssClass}>
      <span className="jp-jupyter-ai-acp-client-tool-call-icon">{icon}</span>{' '}
      {displayTitle}
      <PermissionLabel toolCall={toolCall} />
    </div>
  );
}

/**
 * Shows the user's permission selection.
 */
function PermissionLabel({
  toolCall
}: {
  toolCall: IToolCall;
}): JSX.Element | null {
  if (
    toolCall.permission_status !== 'resolved' ||
    !toolCall.selected_option_id
  ) {
    return null;
  }
  const selectedName = toolCall.permission_options?.find(
    opt => opt.option_id === toolCall.selected_option_id
  )?.name;
  if (!selectedName) {
    return null;
  }
  return (
    <span className="jp-jupyter-ai-acp-client-permission-label">
      {' '}
      — {selectedName}
    </span>
  );
}

/**
 * Renders the permission buttons.
 */
function PermissionButtons({
  toolCall
}: {
  toolCall: IToolCall;
}): JSX.Element | null {
  const [submitting, setSubmitting] = React.useState(false);

  if (
    !toolCall.permission_options?.length ||
    toolCall.permission_status !== 'pending' ||
    !toolCall.session_id
  ) {
    return null;
  }

  const handleClick = async (optionId: string) => {
    setSubmitting(true);
    try {
      await submitPermissionDecision(
        toolCall.session_id!,
        toolCall.tool_call_id,
        optionId
      );
    } catch (err) {
      console.error('Failed to submit permission decision:', err);
      setSubmitting(false);
    }
  };

  return (
    <div className="jp-jupyter-ai-acp-client-permission-buttons">
      <span className="jp-jupyter-ai-acp-client-permission-tree">└─</span>
      <span>Allow?</span>
      {toolCall.permission_options.map((opt: IPermissionOption) => (
        <button
          key={opt.option_id}
          className={clsx(
            'jp-jupyter-ai-acp-client-permission-btn',
            opt.kind &&
              `jp-jupyter-ai-acp-client-permission-btn-${opt.kind.replace(/_/g, '-')}`
          )}
          onClick={() => handleClick(opt.option_id)}
          disabled={submitting}
          title={opt.kind}
        >
          {opt.name}
        </button>
      ))}
    </div>
  );
}
