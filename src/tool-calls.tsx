import React from 'react';
import { IToolCall, MessagePreambleProps } from '@jupyter/chat';

/**
 * Preamble component that renders tool call status lines above message body.
 * Returns null if the message has no tool calls.
 */
export function ToolCallsComponent(
  props: MessagePreambleProps
): JSX.Element | null {
  const { message } = props;
  // TEMPORARY: remove after tool call UI is verified working
  console.warn('[ACP] ToolCallsComponent render:', message.tool_calls);
  if (!message.tool_calls?.length) {
    return null;
  }

  return (
    <div className="jp-jupyter-ai-acp-client-tool-calls">
      {message.tool_calls.map((tc: IToolCall) => (
        <ToolCallLine key={tc.tool_call_id} toolCall={tc} />
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
  return JSON.stringify(rawOutput, null, 2);
}

/**
 * Renders a single tool call line with status icon and optional expandable output.
 */
function ToolCallLine({ toolCall }: { toolCall: IToolCall }): JSX.Element {
  const { title, status, kind, raw_output } = toolCall;
  const displayTitle =
    title ||
    (kind
      ? `${kind.charAt(0).toUpperCase()}${kind.slice(1)}...`
      : 'Working...');
  const isInProgress = status === 'in_progress' || status === 'pending';
  const isCompleted = status === 'completed';
  const isFailed = status === 'failed';

  // Unicode text glyphs — consistent across OS/browser
  const icon = isInProgress
    ? '\u2022'
    : isCompleted
      ? '\u2713'
      : isFailed
        ? '\u2717'
        : '\u2022';
  const cssClass = `jp-jupyter-ai-acp-client-tool-call jp-jupyter-ai-acp-client-tool-call-${status || 'in_progress'}`;

  // Show <details> for: execute with output, OR any failed tool call with output
  const showDetails =
    raw_output &&
    ((kind === 'execute' && (isCompleted || isFailed)) || isFailed);

  if (showDetails) {
    return (
      <details className={cssClass}>
        <summary>
          {icon} {displayTitle}
        </summary>
        <pre className="jp-jupyter-ai-acp-client-tool-call-output">
          {formatOutput(raw_output)}
        </pre>
      </details>
    );
  }

  // In-progress — italic
  if (isInProgress) {
    return (
      <div className={cssClass}>
        {icon} <em>{displayTitle}</em>
      </div>
    );
  }

  // Completed/failed without output
  return (
    <div className={cssClass}>
      {icon} {displayTitle}
    </div>
  );
}
