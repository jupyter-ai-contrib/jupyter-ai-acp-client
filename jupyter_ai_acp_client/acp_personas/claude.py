import shutil
from jupyter_ai_persona_manager import PersonaRequirementsUnmet
if shutil.which("claude-code-acp") is None:
    raise PersonaRequirementsUnmet(
        "This persona requires the Claude Code ACP adapter to be installed."
        " Install it via `npm install -g @zed-industries/claude-code-acp`"
        " then restart."
    )

import os
from jupyter_ai_persona_manager import PersonaDefaults
from jupyterlab_chat.models import Message
from acp.exceptions import RequestError

from ..base_acp_persona import BaseAcpPersona
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
    
    async def before_agent_subprocess(self):
        # The Claude ACP agent server seems to always be able to start as long
        # as `claude-code-acp` is installed, so this method does not need to be
        # implemented.
        return None

    async def is_authed(self) -> bool:
        # Unfortunately, we cannot check the exit code of `claude auth status`
        # as documented to implement this method. This command may claim the
        # user is authenticated even if their token is expired. So we have to
        # always return `True` here.
        # 
        # Upon auth failure, the `process_message()` method raises
        # `acp.exceptions.RequestError: Authentication required` when the user
        # is not logged in. We use that to inform the user to log in.
        return True
    
    async def process_message(self, message: Message) -> None:
        try:
            await super().process_message(message)
        except RequestError as e:
            if "Authentication required" in str(e):
                self.log.info("[Claude] User is not logged in.")
                await self.handle_no_auth(message)
                pass
            else:
                raise e


    async def handle_no_auth(self, message: Message):
        # Claude supports several authentication options so we just send a
        # canned response and let the user choose for themselves.
        self.send_message(
            "You're not authenticated with Claude."
            "\n\n- If you want to log in with a Claude.ai account, you may log in via `claude /login` in a new terminal."
            "\n\n- For cloud provider authentication and other options, see the [Claude.ai documentation](https://code.claude.com/docs/en/authentication)."
        )