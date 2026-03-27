import os
import shutil

from acp.exceptions import RequestError
from jupyter_ai_persona_manager import PersonaDefaults, PersonaRequirementsUnmet
from jupyterlab_chat.models import Message

from ..base_acp_persona import BaseAcpPersona


def _is_auth_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        keyword in message
        for keyword in (
            "api key",
            "api_key",
            "authentication",
            "authorized",
            "credential",
            "not configured",
        )
    )


if shutil.which("codex-acp") is None:
    raise PersonaRequirementsUnmet(
        "This persona requires `codex-acp`, the ACP adapter for OpenAI Codex."
        " Install it via `npm install -g @zed-industries/codex-acp`"
        " then restart."
    )


class CodexAcpPersona(BaseAcpPersona):
    def __init__(self, *args, **kwargs):
        executable = ["codex-acp"]
        super().__init__(*args, executable=executable, **kwargs)

    @property
    def defaults(self) -> PersonaDefaults:
        avatar_path = str(
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "..", "static", "codex.svg"
                )
            )
        )

        return PersonaDefaults(
            name="Codex",
            description="OpenAI Codex as an ACP agent persona.",
            avatar_path=avatar_path,
            system_prompt="unused",
        )

    async def is_authed(self) -> bool:
        return True

    async def process_message(self, message: Message) -> None:
        try:
            await super().process_message(message)
        except RequestError as error:
            if not _is_auth_error(error):
                raise

            self.log.info(
                "[Codex] Authentication or configuration required: %s",
                error,
            )
            await self.handle_no_auth(message)

    async def handle_no_auth(self, message: Message) -> None:
        self.send_message(
            "Codex isn't configured yet."
            "\n\n- Set `OPENAI_API_KEY` (or `CODEX_API_KEY`) before starting JupyterLab."
            "\n\n- Or install the Codex CLI (`npm i -g @openai/codex`)"
            " and run `codex login` to log in with your ChatGPT account."
            "\n\nRestart the JupyterLab server after either step."
        )
