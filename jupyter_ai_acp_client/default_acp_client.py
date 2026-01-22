import asyncio
import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, AsyncGenerator

from acp import (
    PROTOCOL_VERSION,
    Client,
    RequestError,
    connect_to_agent,
    text_block,
)
from acp.core import ClientSideConnection
from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AudioContentBlock,
    AvailableCommandsUpdate,
    ClientCapabilities,
    CreateTerminalResponse,
    CurrentModeUpdate,
    EmbeddedResourceContentBlock,
    EnvVariable,
    FileSystemCapability,
    ImageContentBlock,
    Implementation,
    KillTerminalCommandResponse,
    NewSessionResponse,
    PermissionOption,
    PromptResponse,
    ReadTextFileResponse,
    ReleaseTerminalResponse,
    RequestPermissionResponse,
    ResourceContentBlock,
    TerminalOutputResponse,
    TextContentBlock,
    ToolCall,
    ToolCallProgress,
    ToolCallStart,
    UserMessageChunk,
    WaitForTerminalExitResponse,
    WriteTextFileResponse,
    AllowedOutcome
)
from jupyter_ai_persona_manager import BasePersona
from jupyterlab_chat.ychat import YChat
from typing import Awaitable, ClassVar
from asyncio.subprocess import Process

async def queue_to_iterator(queue: asyncio.Queue[str], sentinel: str = "__end__") -> AsyncGenerator[str]:
    """Convert an asyncio queue to an async iterator."""
    while True:
        item = await queue.get()
        if item == sentinel:
            break
        yield item

class JaiAcpClient(Client):
    """
    The default ACP client. The client should be stored as a class attribute on each
    ACP persona, such that each ACP agent subprocess is communicated through
    exactly one ACP client (an instance of this class).
    """

    agent_subprocess: Process
    _connection_future: ClassVar[Awaitable[ClientSideConnection] | None] = None
    event_loop: asyncio.AbstractEventLoop
    _personas_by_session: dict[str, BasePersona]
    _queues_by_session: dict[str, asyncio.Queue[str]]

    def __init__(self, *args, agent_subprocess: Awaitable[Process], event_loop: asyncio.AbstractEventLoop, **kwargs):
        """
        :param agent_subprocess: The ACP agent subprocess
        (`asyncio.subprocess.Process`) assigned to this client.

        :param event_loop: The `asyncio` event loop running this process.
        """
        self.agent_subprocess = agent_subprocess
        if self.__class__._connection_future is None:
            self.__class__._connection_future = event_loop.create_task(
                self._init_connection()
            )
        self.event_loop = event_loop
        self._personas_by_session = {}
        self._queues_by_session = {}
        super().__init__(*args, **kwargs)
    

    async def _init_connection(self) -> ClientSideConnection:
        proc = self.agent_subprocess
        conn = connect_to_agent(self, proc.stdin, proc.stdout)
        await conn.initialize(
            protocol_version=PROTOCOL_VERSION,
            client_capabilities=ClientCapabilities(
                fs=FileSystemCapability(read_text_file=True, write_text_file=True),
            ),
            client_info=Implementation(name="Jupyter AI", title="Jupyter AI ACP Client", version="0.1.0"),
        )
        return conn
    
    async def get_connection(self) -> ClientSideConnection:
        return await self.__class__._connection_future

    async def create_session(self, persona: BasePersona) -> NewSessionResponse:
        """
        Create an ACP agent session through this client scoped to a
        `BasePersona` instance.
        """
        conn = await self.get_connection()
        # TODO: change this to Jupyter preferred dir
        session = await conn.new_session(mcp_servers=[], cwd=os.getcwd())
        self._personas_by_session[session.session_id] = persona
        return session
    
    async def prompt_and_reply(self, session_id: str, prompt: str, attachments: list[dict] = []) -> PromptResponse:
        """
        A helper method that sends a prompt with an optional list of attachments
        to the assigned ACP server. This method writes back to the chat by
        calling methods on the persona corresponding to this session ID.
        """
        assert session_id in self._personas_by_session
        conn = await self.get_connection()

        # ensure an asyncio Queue exists for this session
        # the `session_update()` method will push chunks to this queue
        queue = self._queues_by_session.get(session_id, None)
        if queue is None:
            queue: asyncio.Queue[str] = asyncio.Queue()
            self._queues_by_session[session_id] = queue

        # create async iterator that yields until the response is complete
        aiter = queue_to_iterator(queue)

        # create background task to stream message back to client using the
        # dedicated persona method 
        persona = self._personas_by_session[session_id]
        self.event_loop.create_task(
            persona.stream_message(aiter)
        )

        # call the model and await
        # TODO: add attachments!
        response = await conn.prompt(
            prompt=[
                TextContentBlock(text=prompt, type="text"),
            ],
            session_id=session_id
        )

        # push sentinel value to queue to close the async iterator
        queue.put_nowait("__end__")

        return response

    async def session_update(
        self,
        session_id: str,
        update: UserMessageChunk
        | AgentMessageChunk
        | AgentThoughtChunk
        | ToolCallStart
        | ToolCallProgress
        | AgentPlanUpdate
        | AvailableCommandsUpdate
        | CurrentModeUpdate,
        **kwargs: Any,
    ) -> None:
        """
        Handles `session/update` requests from the ACP agent. There must be an
        `asyncio.Queue` corresponding to this session ID - this should be set by
        the `prompt_and_reply()` method.
        """

        if not isinstance(update, AgentMessageChunk):
            return

        assert session_id in self._queues_by_session

        content = update.content
        text: str
        if isinstance(content, TextContentBlock):
            text = content.text
        elif isinstance(content, ImageContentBlock):
            text = "<image>"
        elif isinstance(content, AudioContentBlock):
            text = "<audio>"
        elif isinstance(content, ResourceContentBlock):
            text = content.uri or "<resource>"
        elif isinstance(content, EmbeddedResourceContentBlock):
            text = "<resource>"
        else:
            text = "<content>"
        
        queue = self._queues_by_session[session_id]
        queue.put_nowait(text)

    async def request_permission(
        self, options: list[PermissionOption], session_id: str, tool_call: ToolCall, **kwargs: Any
    ) -> RequestPermissionResponse:
        """
        Handles `session/request_permission` requests from the ACP agent.

        TODO: This currently always gives the agent permission. We will need to
        add some tool call approval UI and handle permission requests properly.
        """
        option_id = ""
        for o in options:
            if "allow" in o.option_id.lower():
                option_id = o.option_id
                break

        return RequestPermissionResponse(
            outcome=AllowedOutcome(option_id=option_id, outcome='selected')
        )

    async def write_text_file(
        self, content: str, path: str, session_id: str, **kwargs: Any
    ) -> WriteTextFileResponse | None:
        # Validate path parameter
        if not path or not path.strip():
            raise RequestError.invalid_params({"path": "path cannot be empty"})

        file_path = Path(path)

        # Check if path is a directory
        if file_path.is_dir():
            raise RequestError.invalid_params({"path": "path cannot be a directory"})

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(file_path.write_text, content, encoding="utf-8")
        except PermissionError as e:
            raise RequestError.internal_error({"path": path, "error": f"Permission denied: {e}"})
        except OSError as e:
            raise RequestError.internal_error({"path": path, "error": str(e)})

        return WriteTextFileResponse()

    async def read_text_file(
        self, path: str, session_id: str, limit: int | None = None, line: int | None = None, **kwargs: Any
    ) -> ReadTextFileResponse:
        # Validate path parameter
        if not path or not path.strip():
            raise RequestError.invalid_params({"path": "path cannot be empty"})

        # Validate line parameter (must be >= 1 if provided)
        if line is not None and line < 1:
            raise RequestError.invalid_params({"line": "line must be >= 1 (1-indexed)"})

        # Validate limit parameter (must be >= 1 if provided)
        if limit is not None and limit < 1:
            raise RequestError.invalid_params({"limit": "limit must be >= 1"})

        file_path = Path(path)

        # Check if file exists
        if not file_path.exists():
            raise RequestError.resource_not_found(path)

        # Check if path is a directory
        if file_path.is_dir():
            raise RequestError.invalid_params({"path": "path cannot be a directory"})

        try:
            text = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
        except PermissionError as e:
            raise RequestError.internal_error({"path": path, "error": f"Permission denied: {e}"})
        except OSError as e:
            raise RequestError.internal_error({"path": path, "error": str(e)})

        lines = text.splitlines(keepends=True)

        # line is 1-indexed; default to line 1 if not specified
        start_index = (line - 1) if line is not None else 0

        if limit is not None:
            lines = lines[start_index : start_index + limit]
        else:
            lines = lines[start_index:]

        content = "".join(lines)
        return ReadTextFileResponse(content=content)

    ##############################
    # Terminal methods
    ##############################

    async def create_terminal(
        self,
        command: str,
        session_id: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: list[EnvVariable] | None = None,
        output_byte_limit: int | None = None,
        **kwargs: Any,
    ) -> CreateTerminalResponse:
        raise RequestError.method_not_found("terminal/create")

    async def terminal_output(self, session_id: str, terminal_id: str, **kwargs: Any) -> TerminalOutputResponse:
        raise RequestError.method_not_found("terminal/output")

    async def release_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> ReleaseTerminalResponse | None:
        raise RequestError.method_not_found("terminal/release")

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> WaitForTerminalExitResponse:
        raise RequestError.method_not_found("terminal/wait_for_exit")

    async def kill_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> KillTerminalCommandResponse | None:
        raise RequestError.method_not_found("terminal/kill")

    async def ext_method(self, method: str, params: dict) -> dict:
        raise RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: dict) -> None:
        raise RequestError.method_not_found(method)

