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
  // Extract text from ACP content block arrays: [{type, text}, ...]
  if (Array.isArray(rawOutput) && rawOutput.every(i => i?.text)) {
    return rawOutput.map(i => i.text).join('\n');
  }
  return JSON.stringify(rawOutput, null, 2);
}

/** Tool kinds where expanded view shows full file path(s) from locations. */
const FILE_KINDS = new Set(['read', 'edit', 'delete', 'move']);

/** Tool kinds where expanded view shows raw_output (stdout, search results, etc.). */
const OUTPUT_KINDS = new Set(['search', 'execute', 'think', 'fetch']);

/** Display tier for expanded details content. */
type DetailStyle = 'inline' | 'block';

/**
 * Determine display tier based on kind and content length.
 * File paths and short output get lightweight inline treatment;
 * long command output gets a bordered scrollable code block.
 */
function getDetailStyle(toolCall: IToolCall, lines: string[]): DetailStyle {
  const kind = toolCall.kind;
  // File operations are always inline (just paths)
  if (kind && FILE_KINDS.has(kind)) return 'inline';
  // Think is always inline (prose)
  if (kind === 'think') return 'inline';
  // Count actual newlines in content
  const totalLines = lines.join('\n').split('\n').length;
  // Short output (≤ 3 lines) gets inline treatment
  if (totalLines <= 3) return 'inline';
  // Long output gets scrollable code block
  return 'block';
}

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
function ToolCallLine({ toolCall }: { toolCall: IToolCall }): JSX.Element {
  const { title, status, kind } = toolCall;
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

  // Progressive disclosure: completed/failed tool calls with metadata get expandable details
  const detailsLines = (isCompleted || isFailed) ? buildDetailsLines(toolCall) : [];
  const showDetails = detailsLines.length > 0;

  if (showDetails) {
    const detailStyle = getDetailStyle(toolCall, detailsLines);
    return (
      <details className={cssClass}>
        <summary>
          <span className="jp-jupyter-ai-acp-client-tool-call-icon">{icon}</span> {displayTitle}
        </summary>
        {detailStyle === 'block' ? (
          <pre className="jp-jupyter-ai-acp-client-tool-call-output">
            {detailsLines.join('\n')}
          </pre>
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
        <span className="jp-jupyter-ai-acp-client-tool-call-icon">{icon}</span> <em>{displayTitle}</em>
      </div>
    );
  }

  // Completed/failed without metadata
  return (
    <div className={cssClass}>
      <span className="jp-jupyter-ai-acp-client-tool-call-icon">{icon}</span> {displayTitle}
    </div>
  );
}
