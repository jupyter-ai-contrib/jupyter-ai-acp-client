import asyncio
import json
import shutil

from jupyter_ai_persona_manager import PersonaRequirementsUnmet
if shutil.which("claude-agent-acp") is None:
    raise PersonaRequirementsUnmet(
        "This persona requires the Claude Agent ACP adapter to be installed."
        " Install it via `npm install -g @zed-industries/claude-agent-acp`"
        " then restart."
    )

import os
from jupyter_ai_persona_manager import PersonaDefaults
from jupyterlab_chat.models import Message
from acp.exceptions import RequestError

from ..base_acp_persona import BaseAcpPersona


def _is_auth_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        keyword in message
        for keyword in (
            "auth",
            "login",
            "not signed in",
            "not authenticated",
            "token",
            "credential",
            "forbidden",
            "unauthorized",
        )
    )


async def _check_claude_auth() -> bool:
    """Check if claude is authenticated by running `claude auth status`."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "claude", "auth", "status", "--json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        result = json.loads(stdout.decode())
        return result.get("loggedIn", False)
    except (asyncio.TimeoutError, json.JSONDecodeError, FileNotFoundError, OSError):
        return False


class ClaudeAcpPersona(BaseAcpPersona):
    def __init__(self, *args, **kwargs):
        executable = ["claude-agent-acp"]
        super().__init__(*args, executable=executable, **kwargs)

    @property
    def defaults(self) -> PersonaDefaults:
        avatar_path = str(os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "static", "claude.svg")
        ))

        return PersonaDefaults(
            name="Claude",
            description="Claude Code as an ACP agent persona.",
            avatar_path=avatar_path,
            system_prompt="unused"
        )

    async def before_agent_subprocess(self):
        """Block subprocess startup until Claude is authenticated.
        Polls `claude auth status` every 3 seconds. Once auth passes,
        the subprocess starts with valid credentials — matching Kiro's pattern.
        """
        if await _check_claude_auth():
            return

        # Not authenticated — wait and poll until user logs in
        self.log.info("[Claude] Waiting for user to authenticate...")
        while not await _check_claude_auth():
            await asyncio.sleep(3)
        self.log.info("[Claude] Authentication detected, starting subprocess.")

    async def is_authed(self) -> bool:
        # If before_agent_subprocess hasn't completed yet, the user hasn't
        # authenticated. This mirrors Kiro's pattern.
        if not self._before_subprocess_future.done():
            return False
        # Once the subprocess is running, we trust it has valid credentials.
        # Tokens can still expire mid-session, but we catch that via the
        # try/except in process_message() as a fallback.
        return True

    def _needs_auth_before_subprocess(self) -> bool:
        return True

    async def process_message(self, message: Message) -> None:
        try:
            await super().process_message(message)
        except RequestError as e:
            if _is_auth_error(e):
                self.log.info("[Claude] User is not logged in.")
                await self.handle_no_auth(message)
            else:
                raise e

    async def handle_no_auth(self, message: Message) -> None:
        await super().handle_no_auth(message)
        self.send_message(
            "You're not authenticated with Claude."
            "\n\n- If you want to log in with a Claude.ai account, you may log in via `claude /login` in a new terminal."
            "\n\n- For cloud provider authentication and other options, see the [Claude.ai documentation](https://code.claude.com/docs/en/authentication)."
        )