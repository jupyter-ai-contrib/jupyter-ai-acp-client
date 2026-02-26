import React from 'react';
import { IToolCallDiff } from '@jupyter/chat';
import { diffLines, Change } from 'diff';

/**
 * Renders a single file diff block with filename header and line-level highlighting.
 */
function DiffBlock({ diff }: { diff: IToolCallDiff }): JSX.Element {
  const changes = diffLines(diff.old_text ?? '', diff.new_text);
  const filename = diff.path.split('/').pop() ?? diff.path;

  return (
    <div className="jp-jupyter-ai-acp-client-diff-block">
      <div className="jp-jupyter-ai-acp-client-diff-header">{filename}</div>
      <pre className="jp-jupyter-ai-acp-client-diff-content">
        {changes.map((change: Change, i: number) => {
          const cls = change.added
            ? 'jp-jupyter-ai-acp-client-diff-added'
            : change.removed
              ? 'jp-jupyter-ai-acp-client-diff-removed'
              : 'jp-jupyter-ai-acp-client-diff-context';
          const prefix = change.added ? '+' : change.removed ? '-' : ' ';
          const lines = change.value.replace(/\n$/, '').split('\n');
          return lines.map((line: string, j: number) => (
            <span key={`${i}-${j}`} className={cls}>
              {prefix} {line}
              {'\n'}
            </span>
          ));
        })}
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
