import React from 'react';
import { IToolCallDiff } from '@jupyter/chat';
import { PathExt } from '@jupyterlab/coreutils';
import { structuredPatch } from 'diff';
import clsx from 'clsx';

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
 * highlighting, and click-to-expand truncation.
 */
function DiffBlock({
  diff,
  onOpenFile,
  toDisplayPath,
  pendingPermission
}: {
  diff: IToolCallDiff;
  onOpenFile?: (path: string) => void;
  toDisplayPath?: (path: string) => string;
  pendingPermission?: boolean;
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
  const displayPath = toDisplayPath
    ? toDisplayPath(diff.path)
    : PathExt.basename(diff.path);
  const isClickable =
    !!onOpenFile && !(pendingPermission && diff.old_text === undefined);
  const [expanded, setExpanded] = React.useState(false);

  // Flatten hunks into renderable lines
  const allLines: IDiffLineInfo[] = [];
  for (const hunk of patch.hunks) {
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
          key: `${hunk.oldStart}-${j}`
        });
      });
  }

  const canTruncate = allLines.length > MAX_DIFF_LINES;
  const visible =
    canTruncate && !expanded ? allLines.slice(0, MAX_DIFF_LINES) : allLines;
  const hiddenCount = allLines.length - MAX_DIFF_LINES;

  return (
    <div className="jp-jupyter-ai-acp-client-diff-block">
      <div
        className={clsx('jp-jupyter-ai-acp-client-diff-header', {
          'jp-jupyter-ai-acp-client-diff-header-clickable': isClickable
        })}
        onClick={isClickable ? () => onOpenFile!(diff.path) : undefined}
        title={diff.path}
      >
        {displayPath}
      </div>
      <div className="jp-jupyter-ai-acp-client-diff-content">
        {visible.map((line: IDiffLineInfo) => (
          <div
            key={line.key}
            className={`jp-jupyter-ai-acp-client-diff-line ${line.cls}`}
          >
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
  onOpenFile,
  toDisplayPath,
  pendingPermission
}: {
  diffs: IToolCallDiff[];
  onOpenFile?: (path: string) => void;
  toDisplayPath?: (path: string) => string;
  pendingPermission?: boolean;
}): JSX.Element {
  return (
    <div className="jp-jupyter-ai-acp-client-diff-container">
      {diffs.map((d, i) => (
        <DiffBlock
          key={i}
          diff={d}
          onOpenFile={onOpenFile}
          toDisplayPath={toDisplayPath}
          pendingPermission={pendingPermission}
        />
      ))}
    </div>
  );
}
