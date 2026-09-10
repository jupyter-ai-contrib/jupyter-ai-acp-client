"""Extension-provided context sent to ACP without changing chat messages."""

from copy import deepcopy
from functools import lru_cache
from importlib.metadata import entry_points
from inspect import isawaitable
from collections.abc import Awaitable, Callable, Sequence

from jupyterlab_chat.models import Message

PROMPT_CONTEXT_API_VERSION = 1
ENTRY_POINT_GROUP = "jupyter_ai_acp_client.prompt_context"
PromptContextProvider = Callable[
    [Message | None, Sequence[Message]], str | None | Awaitable[str | None]
]


@lru_cache(maxsize=1)
def get_prompt_context_providers() -> dict[str, PromptContextProvider]:
    """Load installed providers once per process, ordered by entry-point name."""
    providers = {}
    for entry in sorted(entry_points(group=ENTRY_POINT_GROUP), key=lambda e: e.name):
        if entry.name in providers:
            raise ValueError(f"Duplicate prompt context provider: {entry.name}")
        provider = entry.load()
        if not callable(provider):
            raise TypeError(f"Prompt context provider {entry.name!r} is not callable")
        providers[entry.name] = provider
    return providers


async def get_prompt_context(
    message: Message | None, history: Sequence[Message]
) -> str:
    """Collect additional prompt text from server-installed providers.

    Providers receive independent snapshots, not the mutable chat model.
    ``message`` is None for the automatic prompt after agent authentication.
    Deleted messages are excluded from history. Return None/empty text for
    unrelated conversations. Errors abort the prompt through the persona's
    usual error handling rather than silently dropping required context.
    This text is additional user context, not a system-role instruction.
    """
    parts = []
    history = tuple(item for item in history if not item.deleted)
    for name, provider in get_prompt_context_providers().items():
        try:
            result = provider(deepcopy(message), deepcopy(history))
            if isawaitable(result):
                result = await result
            if result is not None and not isinstance(result, str):
                raise TypeError("Expected a string or None")
            if result and result.strip():
                parts.append(result.strip())
        except Exception as error:
            raise ValueError(
                f"Prompt context provider {name!r} failed: {error}"
            ) from error
    return "\n\n".join(parts)
