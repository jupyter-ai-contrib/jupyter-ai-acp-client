# jupyter_ai_acp_client

[![Github Actions Status](https://github.com/jupyter-ai-contrib/jupyter-ai-acp-client/workflows/Build/badge.svg)](https://github.com/jupyter-ai-contrib/jupyter-ai-acp-client/actions/workflows/build.yml)

A proof-of-concept package providing a client implementation of the Agent Client
Protocol (ACP) in Jupyter AI v3, as well as helper classes for other developers
to use when custom AI personas wrapping ACP agents.

This package provides a default ACP client implementation as `JaiAcpClient`.
This client provides a `prompt_and_reply()` method which calls the ACP server
and streams the reply back to the chat. In addition, it provides file read, file
write, and terminal use capabilities.

This package also provides a default `BaseAcpPersona` class which can be easily
extended to add ACP agents as AI personas in JupyterLab. This base class takes
an additional `executable` argument which starts the ACP agent server. This
package also provides a default ACP client implementation as `JaiAcpClient`.

- `BaseAcpPersona` automatically creates new subprocesses for the ACP agent and
  client when needed. These are stored as class attributes, so all instances of
  the same ACP persona share a common ACP agent subprocess.

- Since `BaseAcpPersona` inherits from `BasePersona`, subclasses can be provided
  simply as entry points to become available for use in Jupyter AI. (see
  [documentation](https://jupyter-ai.readthedocs.io/en/v3/developers/entry_points_api/personas_group.html))

- Personas based on ACP now just need to derive from `BaseAcpPersona` and define
  the persona name, the persona avatar, and the `executable` starting the ACP
  agent server.

For example, the `@Claude-ACP` persona is defined in `claude.py` using less than
20 lines of code:

```py
class ClaudeAcpPersona(BaseAcpPersona):
    def __init__(self, *args, **kwargs):
        executable = ["claude-code-acp"]
        super().__init__(*args, executable=executable, **kwargs)

    @property
    def defaults(self) -> PersonaDefaults:
        avatar_path = str(os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "static", "claude.svg")
        ))

        return PersonaDefaults(
            name="Claude-ACP",
            description="Claude Code as an ACP agent persona.",
            avatar_path=avatar_path,
            system_prompt="unused"
        )
```

Currently, this package provides 3 personas:

1. `@Test-ACP` (a test persona that echoes responses)
2. `@Claude-ACP`
   - requires `claude-code-acp`, installed via `npm install -g @zed-industries/claude-code-acp`
3. `@Kiro`
   - requires `kiro-cli`, installed from https://kiro.dev

## Dependencies

**Required**:
- JupyterLab >= 4.0.0
- `jupyter-ai-persona-manager>=0.0.5`
- `agent_client_protocol`

**Optional**
- `claude-code-acp` (enables `@Claude-ACP`)
- `kiro-cli` (enables `@Kiro`)

## Install

To install the extension, execute:

```bash
pip install jupyter_ai_acp_client
```

## Uninstall

To remove the extension, execute:

```bash
pip uninstall jupyter_ai_acp_client
```

## Troubleshoot

If you are seeing the frontend extension, but it is not working, check
that the server extension is enabled:

```bash
jupyter server extension list
```

If the server extension is installed and enabled, but you are not seeing
the frontend extension, check the frontend extension is installed:

```bash
jupyter labextension list
```

## Contributing

### Development install

Note: You will need NodeJS to build the extension package.

The `jlpm` command is JupyterLab's pinned version of
[yarn](https://yarnpkg.com/) that is installed with JupyterLab. You may use
`yarn` or `npm` in lieu of `jlpm` below.

```bash
# Clone the repo to your local environment
# Change directory to the jupyter_ai_acp_client directory

# Set up a virtual environment and install package in development mode
python -m venv .venv
source .venv/bin/activate
pip install --editable ".[dev,test]"

# Link your development version of the extension with JupyterLab
jupyter labextension develop . --overwrite
# Server extension must be manually installed in develop mode
jupyter server extension enable jupyter_ai_acp_client

# Rebuild extension Typescript source after making changes
# IMPORTANT: Unlike the steps above which are performed only once, do this step
# every time you make a change.
jlpm build
```

You can watch the source directory and run JupyterLab at the same time in different terminals to watch for changes in the extension's source and automatically rebuild the extension.

```bash
# Watch the source directory in one terminal, automatically rebuilding when needed
jlpm watch
# Run JupyterLab in another terminal
jupyter lab
```

With the watch command running, every saved change will immediately be built locally and available in your running JupyterLab. Refresh JupyterLab to load the change in your browser (you may need to wait several seconds for the extension to be rebuilt).

By default, the `jlpm build` command generates the source maps for this extension to make it easier to debug using the browser dev tools. To also generate source maps for the JupyterLab core extensions, you can run the following command:

```bash
jupyter lab build --minimize=False
```

### Development uninstall

```bash
# Server extension must be manually disabled in develop mode
jupyter server extension disable jupyter_ai_acp_client
pip uninstall jupyter_ai_acp_client
```

In development mode, you will also need to remove the symlink created by `jupyter labextension develop`
command. To find its location, you can run `jupyter labextension list` to figure out where the `labextensions`
folder is located. Then you can remove the symlink named `@jupyter-ai/acp-client` within that folder.

### Testing the extension

#### Server tests

This extension is using [Pytest](https://docs.pytest.org/) for Python code testing.

Install test dependencies (needed only once):

```sh
pip install -e ".[test]"
# Each time you install the Python package, you need to restore the front-end extension link
jupyter labextension develop . --overwrite
```

To execute them, run:

```sh
pytest -vv -r ap --cov jupyter_ai_acp_client
```

#### Frontend tests

This extension is using [Jest](https://jestjs.io/) for JavaScript code testing.

To execute them, execute:

```sh
jlpm
jlpm test
```

#### Integration tests

This extension uses [Playwright](https://playwright.dev/docs/intro) for the integration tests (aka user level tests).
More precisely, the JupyterLab helper [Galata](https://github.com/jupyterlab/jupyterlab/tree/master/galata) is used to handle testing the extension in JupyterLab.

More information are provided within the [ui-tests](./ui-tests/README.md) README.

## AI Coding Assistant Support

This project includes an `AGENTS.md` file with coding standards and best practices for JupyterLab extension development. The file follows the [AGENTS.md standard](https://agents.md) for cross-tool compatibility.

### Compatible AI Tools

`AGENTS.md` works with AI coding assistants that support the standard, including Cursor, GitHub Copilot, Windsurf, Aider, and others. For a current list of compatible tools, see [the AGENTS.md standard](https://agents.md).
This project also includes symlinks for tool-specific compatibility:

- `CLAUDE.md` → `AGENTS.md` (for Claude Code)

- `GEMINI.md` → `AGENTS.md` (for Gemini Code Assist)

Other conventions you might encounter:

- `.cursorrules` - Cursor's YAML/JSON format (Cursor also supports AGENTS.md natively)
- `CONVENTIONS.md` / `CONTRIBUTING.md` - For CodeConventions.ai and GitHub bots
- Project-specific rules in JetBrains AI Assistant settings

All tool-specific files should be symlinks to `AGENTS.md` as the single source of truth.

### What's Included

The `AGENTS.md` file provides guidance on:

- Code quality rules and file-scoped validation commands
- Naming conventions for packages, plugins, and files
- Coding standards (TypeScript, Python)
- Development workflow and debugging
- Backend-frontend integration patterns (`APIHandler`, `requestAPI()`, routing)
- Common pitfalls and how to avoid them

### Customization

You can edit `AGENTS.md` to add project-specific conventions or adjust guidelines to match your team's practices. The file uses plain Markdown with Do/Don't patterns and references to actual project files.

**Note**: `AGENTS.md` is living documentation. Update it when you change conventions, add dependencies, or discover new patterns. Include `AGENTS.md` updates in commits that modify workflows or coding standards.

### Packaging the extension

See [RELEASE](RELEASE.md)
