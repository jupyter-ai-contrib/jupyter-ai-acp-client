from jupyter_ai_persona_manager import BasePersona
from jupyterlab_chat.models import Message
import asyncio
from asyncio.subprocess import Process
from typing import Awaitable, ClassVar
from acp import NewSessionResponse
from acp.schema import AvailableCommand

from .default_acp_client import JaiAcpClient


class BaseAcpPersona(BasePersona):
    _subprocess_future: ClassVar[Awaitable[Process] | None] = None
    """
    The task that yields the agent subprocess once complete. This is a class
    attribute because multiple instances of the same ACP persona may share an
    ACP agent subprocess.
    
    Developers should always use `self.get_agent_subprocess()`.
    """

    _client_future: ClassVar[Awaitable[JaiAcpClient] | None] = None
    """
    The future that yields the ACP Client once complete. This is a class
    attribute because multiple instances of the same ACP persona may share an
    ACP client as well. ACP agent subprocesses and clients map 1-to-1.

    Developers should always use `self.get_client()`.
    """

    _client_session_future: Awaitable[NewSessionResponse]
    """
    The future that yields the ACP client session info. Each instance of an ACP
    persona has a unique session ID, i.e. each chat reserves a unique session.

    Developers should always call `self.get_session()` or `self.get_session_id()`.
    """

    _acp_slash_commands: list[AvailableCommand]

    def __init__(self, *args, executable: list[str], **kwargs):
        super().__init__(*args, **kwargs)

        self._executable = executable
        if self.__class__._subprocess_future is None:
            self.__class__._subprocess_future = self.event_loop.create_task(
                self._init_agent_subprocess()
            )
        if self.__class__._client_future is None:
            self.__class__._client_future = self.event_loop.create_task(
                self._init_client()
            )
        self._client_session_future = self.event_loop.create_task(
            self._init_client_session()
        )
        self._acp_slash_commands = []

    async def _init_agent_subprocess(self) -> Process:
        process = await asyncio.create_subprocess_exec(
            *self._executable,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        self.log.info(f"Spawned ACP agent subprocess for '{self.__class__.__name__}'.")
        return process

    async def _init_client(self) -> JaiAcpClient:
        agent_subprocess = await self.get_agent_subprocess()
        client = JaiAcpClient(agent_subprocess=agent_subprocess, event_loop=self.event_loop)
        self.log.info(f"Initialized ACP client for '{self.__class__.__name__}'.")
        return client
    
    async def _init_client_session(self) -> NewSessionResponse:
        client = await self.get_client()
        session = await client.create_session(persona=self)
        self.log.info(
            f"Initialized new ACP client session for '{self.__class__.__name__}'"
            f" with ID '{session.session_id}'."
        )
        return session

    async def get_agent_subprocess(self) -> asyncio.subprocess.Process:
        """
        Safely returns the ACP agent subprocess for this persona.
        """
        return await self.__class__._subprocess_future
    
    async def get_client(self) -> JaiAcpClient:
        """
        Safely returns the ACP client for this persona.
        """
        return await self.__class__._client_future
    
    async def get_session(self) -> NewSessionResponse:
        """
        Safely returns the ACP client session for this chat.
        """
        return await self._client_session_future
    
    async def get_session_id(self) -> str:
        """
        Safely returns the ACP client ID assigned to this chat.
        """
        session = await self._client_session_future
        return session.session_id
    
    async def process_message(self, message: Message) -> None:
        """
        A default implementation for the `BasePersona.process_message()` method
        for ACP agents.

        This method may be overriden by child classes.
        """
        client = await self.get_client()
        session_id = await self.get_session_id()

        # TODO: add attachments!
        prompt = message.body.replace("@" + self.as_user().mention_name, "").strip()
        await client.prompt_and_reply(
            session_id=session_id,
            prompt=prompt,
        )
    
    @property
    def acp_slash_commands(self) -> list[AvailableCommand]:
        """
        Returns the list of slash commands advertised by the ACP agent in the
        current session.

        This initializes to an empty list, and should be updated **only** by the
        ACP client upon receiving a `session/update` request containing an
        `AvailableCommandsUpdate` payload from the ACP agent.
        """
        return self._acp_slash_commands
    
    @acp_slash_commands.setter
    def acp_slash_commands(self, commands: list[AvailableCommand]):
        self.log.info(
            f"Setting {len(commands)} slash commands for '{self.name}' in room '{self.parent.room_id}'."
        )
        self._acp_slash_commands = commands

    def shutdown(self):
        # TODO: allow shutdown() to be async
        self.event_loop.create_task(self._shutdown())

    async def _shutdown(self):
        self.log.info(f"Closing ACP agent and client for '{self.__class__.__name__}'.")
        client = await self.get_client()
        conn = await client.get_connection()
        await conn.close()
        subprocess = await self.get_agent_subprocess()
        try:
            subprocess.kill()
        except ProcessLookupError:
            pass
        self.log.info(f"Completed closed ACP agent and client for '{self.__class__.__name__}'.")
