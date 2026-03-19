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
            "authentication",
            "unauthorized",
            "credential",
            "not configured",
            "mistral_api_key",
        )
    )


if shutil.which("vibe-acp") is None:
    raise PersonaRequirementsUnmet(
        "This persona requires `vibe-acp`, which is provided by the `mistral-vibe` package."
        " Install it via `uv tool install mistral-vibe`, `pip install mistral-vibe`,"
        " or Mistral's install script, then restart."
    )


class MistralVibeAcpPersona(BaseAcpPersona):
    def __init__(self, *args, **kwargs):
        executable = ["vibe-acp"]
        super().__init__(*args, executable=executable, **kwargs)

    @property
    def defaults(self) -> PersonaDefaults:
        avatar_path = str(
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "..", "static", "mistral_vibe.svg"
                )
            )
        )

        return PersonaDefaults(
            name="Mistral Vibe",
            description="Mistral Vibe as an ACP agent persona.",
            avatar_path=avatar_path,
            system_prompt="unused",
        )

    async def before_agent_subprocess(self) -> None:
        return None

    async def is_authed(self) -> bool:
        return True

    async def process_message(self, message: Message) -> None:
        try:
            await super().process_message(message)
        except RequestError as error:
            if not _is_auth_error(error):
                raise

            self.log.info(
                "[Mistral Vibe] Authentication or configuration required: %s",
                error,
            )
            await self.handle_no_auth(message)

    async def handle_no_auth(self, message: Message) -> None:
        self.send_message(
            "Mistral Vibe isn't configured yet."
            "\n\n- Run `vibe --setup` in a terminal to configure your API key."
            "\n\n- Or set `MISTRAL_API_KEY` before starting JupyterLab."
            "\n\n- If you set `MISTRAL_API_KEY` in a new shell, restart the JupyterLab server so this process can see it."
        )
