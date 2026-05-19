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
    """Check if Claude is authenticated via Anthropic OAuth or AWS Bedrock."""
    # Check Anthropic OAuth via claude auth status
    try:
        proc = await asyncio.create_subprocess_exec(
            "claude", "auth", "status", "--json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        result = json.loads(stdout.decode())
        if result.get("loggedIn", False):
            return True
    except (asyncio.TimeoutError, json.JSONDecodeError, FileNotFoundError, OSError):
        pass

    # Check AWS/Bedrock credentials via aws sts get-caller-identity
    try:
        proc = await asyncio.create_subprocess_exec(
            "aws", "sts", "get-caller-identity",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=10.0)
        if proc.returncode == 0:
            return True
    except (asyncio.TimeoutError, FileNotFoundError, OSError):
        pass

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
        Polls both `claude auth status` (Anthropic OAuth) and
        `aws sts get-caller-identity` (Bedrock) every 3 seconds.
        """
        if await _check_claude_auth():
            return

        self.log.info("[Claude] Waiting for user to authenticate...")
        while not await _check_claude_auth():
            await asyncio.sleep(3)
        self.log.info("[Claude] Authentication detected, starting subprocess.")

    async def is_authed(self) -> bool:
        if not self._before_subprocess_future.done():
            return False
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
            "\n\n- If you want to log in with a Claude.ai account, run the following in a new terminal:"
            "\n\n```\nclaude /login\n```"
            "\n\n- For cloud provider authentication and other options, see the [Claude.ai documentation](https://code.claude.com/docs/en/authentication)."
        )