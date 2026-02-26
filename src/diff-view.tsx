import React from 'react';
import { IToolCallDiff } from '@jupyter/chat';
import { diffLines, Change } from 'diff';

/** Maximum number of diff lines shown before truncation. */
const MAX_DIFF_LINES = 20;

/** A single flattened diff line with its styling metadata. */
interface IDiffLineInfo {
  cls: string;
  prefix: string;
  text: string;
  key: string;
}

/**
 * Renders a single file diff block with filename header, line-level
 * highlighting, and click-to-expand truncation for long diffs.
 */
function DiffBlock({ diff }: { diff: IToolCallDiff }): JSX.Element {
  const changes = diffLines(diff.old_text ?? '', diff.new_text);
  const filename = diff.path.split('/').pop() ?? diff.path;
  const [expanded, setExpanded] = React.useState(false);

  // Flatten all change hunks into individual lines
  const allLines: IDiffLineInfo[] = [];
  changes.forEach((change: Change, i: number) => {
    const cls = change.added
      ? 'jp-jupyter-ai-acp-client-diff-added'
      : change.removed
        ? 'jp-jupyter-ai-acp-client-diff-removed'
        : 'jp-jupyter-ai-acp-client-diff-context';
    const prefix = change.added ? '+' : change.removed ? '-' : ' ';
    const lines = change.value.replace(/\n$/, '').split('\n');
    lines.forEach((line: string, j: number) => {
      allLines.push({ cls, prefix, text: line, key: `${i}-${j}` });
    });
  });

  const canTruncate = allLines.length > MAX_DIFF_LINES;
  const visible =
    canTruncate && !expanded ? allLines.slice(0, MAX_DIFF_LINES) : allLines;
  const hiddenCount = allLines.length - MAX_DIFF_LINES;

  return (
    <div className="jp-jupyter-ai-acp-client-diff-block">
      <div className="jp-jupyter-ai-acp-client-diff-header">{filename}</div>
      <pre className="jp-jupyter-ai-acp-client-diff-content">
        {visible.map((line: IDiffLineInfo) => (
          <span key={line.key} className={line.cls}>
            {line.prefix} {line.text}
            {'\n'}
          </span>
        ))}
        {canTruncate && !expanded && (
          <span
            className="jp-jupyter-ai-acp-client-diff-toggle"
            onClick={() => setExpanded(true)}
          >
            ... {hiddenCount} more lines
          </span>
        )}
        {canTruncate && expanded && (
          <span
            className="jp-jupyter-ai-acp-client-diff-toggle"
            onClick={() => setExpanded(false)}
          >
            show less
          </span>
        )}
      </pre>
    </div>
  );
}

/**
 * Renders one or more file diffs.
 */
export function DiffView({ diffs }: { diffs: IToolCallDiff[] }): JSX.Element {
  return (
    <div className="jp-jupyter-ai-acp-client-diff-container">
      {diffs.map((d, i) => (
        <DiffBlock key={i} diff={d} />
      ))}
    </div>
  );
}
