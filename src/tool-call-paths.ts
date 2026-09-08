import { PageConfig, PathExt } from '@jupyterlab/coreutils';

function getConfiguredServerRoot(): string | null {
  const rootUri = PageConfig.getOption('rootUri');

  if (rootUri) {
    try {
      return new URL(rootUri, 'http://localhost').pathname;
    } catch (error) {
      console.warn(
        'Could not parse rootUri while opening tool call path.',
        error
      );
    }
  }

  const serverRoot = PageConfig.getOption('serverRoot');
  return serverRoot || null;
}

/**
 * Convert an absolute filesystem path to a server-relative path when possible.
 */
export function toServerRelativePath(path: string): string {
  if (!path.startsWith('/')) {
    return path;
  }

  const serverRoot = getConfiguredServerRoot();

  if (!serverRoot) {
    return path;
  }

  const relativePath = PathExt.relative(serverRoot, path);
  if (relativePath.startsWith('..')) {
    return path;
  }

  return relativePath;
}

/**
 * Return the path that can be opened through the document manager, if any.
 */
export function getOpenableToolCallPath(path: string): string | null {
  const openPath = toServerRelativePath(path);

  return openPath.startsWith('/') ? null : openPath;
}
