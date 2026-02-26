import React from 'react';
import { IToolCallDiff } from '@jupyter/chat';
import { structuredPatch } from 'diff';

/** Maximum number of diff lines shown before truncation. */
const MAX_DIFF_LINES = 20;

/** A single flattened diff line with its styling and line number metadata. */
interface IDiffLineInfo {
  cls: string;
  prefix: string;
  text: string;
  key: string;
  oldLineNum: number | null;
  newLineNum: number | null;
}

/**
 * Renders a single file diff block with filename header, dual line number
 * gutters, line-level highlighting, and click-to-expand truncation.
 */
function DiffBlock({
  diff,
  onOpenFile
}: {
  diff: IToolCallDiff;
  onOpenFile?: (path: string) => void;
}): JSX.Element {
  const patch = structuredPatch(
    diff.path,
    diff.path,
    diff.old_text ?? '',
    diff.new_text,
    undefined,
    undefined,
    { context: Infinity }
  );
  const filename = diff.path.split('/').pop() ?? diff.path;
  const [expanded, setExpanded] = React.useState(false);

  // Flatten hunks into renderable lines
  const allLines: IDiffLineInfo[] = [];
  for (const hunk of patch.hunks) {
    let oldLine = hunk.oldStart;
    let newLine = hunk.newStart;
    hunk.lines
      .filter(line => !line.startsWith('\\'))
      .forEach((line, j) => {
        const prefix = line[0];
        const text = line.slice(1);
        const isAdded = prefix === '+';
        const isRemoved = prefix === '-';
        allLines.push({
          cls: isAdded
            ? 'jp-jupyter-ai-acp-client-diff-added'
            : isRemoved
              ? 'jp-jupyter-ai-acp-client-diff-removed'
              : 'jp-jupyter-ai-acp-client-diff-context',
          prefix,
          text,
          key: `${hunk.oldStart}-${j}`,
          oldLineNum: isAdded ? null : oldLine,
          newLineNum: isRemoved ? null : newLine
        });
        if (!isAdded) {
          oldLine++;
        }
        if (!isRemoved) {
          newLine++;
        }
      });
  }

  const canTruncate = allLines.length > MAX_DIFF_LINES;
  const visible =
    canTruncate && !expanded ? allLines.slice(0, MAX_DIFF_LINES) : allLines;
  const hiddenCount = allLines.length - MAX_DIFF_LINES;

  return (
    <div className="jp-jupyter-ai-acp-client-diff-block">
      <div
        className="jp-jupyter-ai-acp-client-diff-header"
        onClick={onOpenFile ? () => onOpenFile(diff.path) : undefined}
        title={diff.path}
      >
        {filename}
      </div>
      <div className="jp-jupyter-ai-acp-client-diff-content">
        {visible.map((line: IDiffLineInfo) => (
          <div
            key={line.key}
            className={`jp-jupyter-ai-acp-client-diff-line ${line.cls}`}
          >
            <span className="jp-jupyter-ai-acp-client-diff-gutter">
              {line.oldLineNum ?? ' '}
            </span>
            <span className="jp-jupyter-ai-acp-client-diff-gutter">
              {line.newLineNum ?? ' '}
            </span>
            <span className="jp-jupyter-ai-acp-client-diff-line-text">
              {line.prefix} {line.text}
            </span>
          </div>
        ))}
        {canTruncate && !expanded && (
          <div
            className="jp-jupyter-ai-acp-client-diff-toggle"
            onClick={() => setExpanded(true)}
          >
            ... {hiddenCount} more lines
          </div>
        )}
        {canTruncate && expanded && (
          <div
            className="jp-jupyter-ai-acp-client-diff-toggle"
            onClick={() => setExpanded(false)}
          >
            show less
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Renders one or more file diffs.
 */
export function DiffView({
  diffs,
  onOpenFile
}: {
  diffs: IToolCallDiff[];
  onOpenFile?: (path: string) => void;
}): JSX.Element {
  return (
    <div className="jp-jupyter-ai-acp-client-diff-container">
      {diffs.map((d, i) => (
        <DiffBlock key={i} diff={d} onOpenFile={onOpenFile} />
      ))}
    </div>
  );
}
