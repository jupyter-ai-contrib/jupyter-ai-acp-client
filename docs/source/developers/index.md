# jupyter-ai-acp-client

## Additional prompt context

Extensions can add context to ACP prompts without changing the user's message
body or inserting messages into the conversation. Register a server-side callable
through a Python entry point:

```toml
[project.entry-points."jupyter_ai_acp_client.prompt_context"]
example = "example_extension.context:prompt_context"
```

```python
import json


def prompt_context(message, history):
    if message is None:
        return None
    project = (message.metadata or {}).get("example_project")
    if not isinstance(project, str):
        return None
    return f"Attached project: {json.dumps(project)}"
```

`prompt_context(message, history)` may be synchronous or asynchronous. It returns
text, or `None`/empty text when the extension does not apply. Both arguments are
independent deep copies of chat messages. `history` excludes deleted messages;
it may contain the current message. `message` is `None` for the automatic prompt
after an agent signs in, so providers with conversation-level bindings should
recover those from history. Validate persisted metadata before using it.

Providers are loaded once per server process and called in entry-point name
order for each normal prompt, including session recovery, and for the automatic
post-authentication prompt. Duplicate names, loading errors, invalid results and
provider exceptions stop the prompt through the persona's normal error handling;
required context is never silently omitted. Install/update providers and restart
the Jupyter server to change the registry.

Returned text is appended to the outgoing prompt as additional **user context**,
not as a system-role message. User-authored text, mentions, attachments and chat
metadata stay unchanged. Context is not rendered, copied or edited as part of the
chat message, though an agent may refer to it in its response. Providers should
not return secrets and should return nothing for unrelated chats. Only installed
server packages can register providers; metadata cannot name executable providers.

This API applies to personas using `BaseAcpPersona.process_message`. Custom
personas which replace that method can call
`jupyter_ai_acp_client.prompt_context.get_prompt_context(message, history)`
explicitly before constructing their own agent prompt. The module exports
`PROMPT_CONTEXT_API_VERSION = 1` and `get_prompt_context_providers()` for hosts
that need to verify support and registration before accepting a message.
