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


def _check_copilot() -> None:
    if shutil.which("copilot") is None:
        raise PersonaRequirementsUnmet(
            "This persona requires the GitHub Copilot CLI."
            " Install it via https://docs.github.com/en/copilot/how-tos/copilot-cli/set-up-copilot-cli/install-copilot-cli"
            " and restart."
        )


_check_copilot()


class CopilotAcpPersona(BaseAcpPersona):
    def __init__(self, *args, **kwargs):
        executable = ["copilot", "--acp", "--stdio"]
        super().__init__(*args, executable=executable, **kwargs)

    @property
    def defaults(self) -> PersonaDefaults:
        avatar_path = str(
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "..", "static", "copilot.svg"
                )
            )
        )

        return PersonaDefaults(
            name="Copilot",
            description="GitHub Copilot as an ACP agent persona.",
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
                "[Copilot] Authentication or configuration required: %s",
                error,
            )
            await self.handle_no_auth(message)

    async def handle_no_auth(self, message: Message) -> None:
        self.send_message(
            "GitHub Copilot isn't configured yet."
            "\n\n- Run `copilot login` in a terminal to sign in with GitHub."
            "\n\n- On headless or non-interactive setups, set `COPILOT_GITHUB_TOKEN`, `GH_TOKEN`, or `GITHUB_TOKEN` before starting JupyterLab."
            "\n\n- Restart the JupyterLab server after changing Copilot CLI authentication or environment variables."
        )
