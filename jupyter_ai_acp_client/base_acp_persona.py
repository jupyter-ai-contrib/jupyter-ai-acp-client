from jupyter_ai_persona_manager import BasePersona
from jupyterlab_chat.models import Message
import asyncio
import sys
from asyncio import Task
from asyncio.subprocess import Process
from typing import Awaitable, ClassVar
from acp import NewSessionResponse
from acp.schema import AvailableCommand

from .default_acp_client import JaiAcpClient


class BaseAcpPersona(BasePersona):
    _auth_future: ClassVar[Task[None] | None] = None
    """
    The future that resolves only once the user is authenticated.
    """

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
    """
    List of slash commands broadcast by the ACP agent in the current session.
    This attribute is set automatically by the default ACP client.
    """

    def __init__(self, *args, executable: list[str], **kwargs):
        super().__init__(*args, **kwargs)

        self._executable = executable

        # Ensure each subclass has its own subprocess and client by checking if the
        # class variable is defined directly on this class (not inherited)
        if '_auth_future' not in self.__class__.__dict__ or self.__class__._auth_future is None:
            self.__class__._auth_future = self.event_loop.create_task(
                self.ensure_auth()
            )
        if '_subprocess_future' not in self.__class__.__dict__ or self.__class__._subprocess_future is None:
            self.__class__._subprocess_future = self.event_loop.create_task(
                self._init_agent_subprocess()
            )
        if '_client_future' not in self.__class__.__dict__ or self.__class__._client_future is None:
            self.__class__._client_future = self.event_loop.create_task(
                self._init_client()
            )

        self._client_session_future = self.event_loop.create_task(
            self._init_client_session()
        )
        self._acp_slash_commands = []

    async def ensure_auth(self) -> None:
        """
        Defines a task that resolves only once the client is authenticated with
        this agent. While this task is pending, the ACP agent subprocess is not
        started, and any messages sent to this persona return a canned response
        defined in `self.handle_no_auth()`.

        The `BaseAcpPersona` does not implement this method by default.
        Subclasses are expected to provide a custom implementation of this
        method if auth is required prior to usage. 
        """
        self.log.info(
            f"No auth method is defined in '{self.__class__.__name__}'."
            " To require auth, implement the `ensure_auth()` method."
        )
        return None

    async def _init_agent_subprocess(self) -> Process:
        # Wait until user is authenticated
        await self._auth_future
        process = await asyncio.create_subprocess_exec(
            *self._executable,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=sys.stderr,
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
    
    def is_authed(self) -> bool:
        """
        Returns whether the client is authenticated to use this agent. Helper
        alias for `self.__class__._auth_future.done()`.
        """
        if getattr(self.__class__, '_auth_future', None) is None:
            return False
        return self.__class__._auth_future.done()
    
    async def handle_no_auth(self, message: Message) -> None:
        """
        Method called when the persona receives a message while the user is not
        authenticated. This method should return a canned response to the chat
        asking the user to log in. Subclasses may override this method to
        customize the help message sent.
        """
        self.send_message("You are not authenticated. Please log in.")
    
    async def process_message(self, message: Message) -> None:
        """
        A default implementation for the `BasePersona.process_message()` method
        for ACP agents.

        This method may be overriden by child classes.
        """
        # If not authenticated, return early
        if not self.is_authed():
            await self.handle_no_auth(message)
            return

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
