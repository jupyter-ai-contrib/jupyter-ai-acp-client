import asyncio
import os
from pathlib import Path
from typing import Any, Awaitable
from time import time

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
    McpServerStdio as AcpMcpServerStdio,
    HttpMcpServer as AcpMcpServerHttp,
    AllowedOutcome
)
from jupyter_ai_persona_manager import BasePersona, McpServerStdio
from jupyterlab_chat.models import Message, NewMessage
from jupyterlab_chat.utils import find_mentions
from asyncio.subprocess import Process

from .terminal_manager import TerminalManager
from .tool_call_manager import ToolCallManager
from .tool_call_renderer import extract_diffs
from .permission_manager import PermissionManager

import traceback as tb_mod

class JaiAcpClient(Client):
    """
    The default ACP client. The client should be stored as a class attribute on each
    ACP persona, such that each ACP agent subprocess is communicated through
    exactly one ACP client (an instance of this class).
    """

    agent_subprocess: Process
    _connection_future: Awaitable[ClientSideConnection]
    event_loop: asyncio.AbstractEventLoop
    _personas_by_session: dict[str, BasePersona]
    _terminal_manager: TerminalManager
    _tool_call_manager: ToolCallManager
    _prompt_locks_by_session: dict[str, asyncio.Lock]

    def __init__(
            self,
            *args,
            agent_subprocess: Awaitable[Process],
            event_loop: asyncio.AbstractEventLoop,
            **kwargs,
    ):
        """
        :param agent_subprocess: The ACP agent subprocess
        (`asyncio.subprocess.Process`) assigned to this client.

        :param event_loop: The `asyncio` event loop running this process.
        """
        self.agent_subprocess = agent_subprocess
        # Each client instance needs its own connection to its own subprocess
        self._connection_future = event_loop.create_task(
            self._init_connection()
        )
        self.event_loop = event_loop
        # Each client instance maintains its own session mappings
        self._personas_by_session = {}
        self._prompt_locks_by_session: dict[str, asyncio.Lock] = {}
        self._terminal_manager = TerminalManager(event_loop)
        self._tool_call_manager = ToolCallManager()
        self._permission_manager = PermissionManager(event_loop)
        super().__init__(*args, **kwargs)


    async def _init_connection(self) -> ClientSideConnection:
        proc = self.agent_subprocess
        conn = connect_to_agent(self, proc.stdin, proc.stdout)
        await conn.initialize(
            protocol_version=PROTOCOL_VERSION,
            client_capabilities=ClientCapabilities(
                fs=FileSystemCapability(read_text_file=True, write_text_file=True),
                terminal=True,
            ),
            client_info=Implementation(name="Jupyter AI", title="Jupyter AI ACP Client", version="0.1.0"),
        )
        return conn

    async def get_connection(self) -> ClientSideConnection:
        return await self._connection_future

    async def create_session(self, persona: BasePersona) -> NewSessionResponse:
        """
        Create an ACP agent session through this client scoped to a
        `BasePersona` instance.
        """
        conn = await self.get_connection()

        # read MCP settings from persona
        mcp_settings = persona.get_mcp_settings()

        # Parse MCP servers from `.jupyter/mcp_settings.json`.
        # We need to cast each from the PersonaManager model to the ACP model
        # here. The models are the exact same, but we still need to do this to
        # avoid a Pydantic error. 
        mcp_servers: list[AcpMcpServerStdio | AcpMcpServerHttp] = []
        if mcp_settings:
            for mcp_server in mcp_settings.mcp_servers:
                if isinstance(mcp_server, McpServerStdio):
                    mcp_servers.append(AcpMcpServerStdio(**mcp_server.model_dump()))
                else:
                    mcp_servers.append(AcpMcpServerHttp(**mcp_server.model_dump()))

        # TODO: change this to Jupyter preferred dir
        session = await conn.new_session(mcp_servers=mcp_servers, cwd=os.getcwd())
        self._personas_by_session[session.session_id] = persona
        return session

    async def prompt_and_reply(self, session_id: str, prompt: str, attachments: list[dict] | None = None) -> PromptResponse:
        """
        A helper method that sends a prompt with an optional list of attachments
        to the assigned ACP server. This method writes back to the chat by
        handling all events in session_update().

        Uses a per-session lock to serialize concurrent calls, preventing
        state corruption if multiple messages arrive before the first completes.
        """
        assert session_id in self._personas_by_session
        lock = self._prompt_locks_by_session.setdefault(session_id, asyncio.Lock())

        # Auto-reject any pending permission requests
        rejected = self._permission_manager.reject_all_pending(session_id)
        if rejected:
            persona = self._personas_by_session.get(session_id)
            if persona:
                persona.log.info(
                    f"prompt_and_reply: auto-rejected {rejected} pending permission(s) for session {session_id}"
                )

        async with lock:
            conn = await self.get_connection()
            persona = self._personas_by_session[session_id]

            # Reset session state for this prompt
            self._tool_call_manager.reset(session_id)

            persona.log.info(f"prompt_and_reply: starting for session {session_id}")

            # Set awareness to indicate writing
            persona.awareness.set_local_state_field("isWriting", True)

            try:
                # Call the model and await — session_update() handles all events
                response = await conn.prompt(
                    prompt=[
                        TextContentBlock(text=prompt, type="text"),
                    ],
                    session_id=session_id
                )

                # Trigger find_mentions on the final message
                message_id = self._tool_call_manager.get_message_id(session_id)
                if message_id:
                    msg = persona.ychat.get_message(message_id)
                    if msg:
                        persona.ychat.update_message(
                            msg,
                            trigger_actions=[find_mentions],
                        )

                persona.log.info(f"prompt_and_reply: completed for session {session_id}")
                return response
            except Exception:
                persona.log.exception(f"prompt_and_reply: failed for session {session_id}")
                raise
            finally:
                # Clear awareness writing state
                persona.awareness.set_local_state_field("isWriting", False)

    def _handle_agent_message_chunk(self, session_id: str, update: AgentMessageChunk) -> None:
        """Handle an AgentMessageChunk event by appending text to the message."""
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

        persona = self._personas_by_session[session_id]
        message_id = self._tool_call_manager.get_or_create_message(session_id, persona)
        serialized_tool_calls = self._tool_call_manager.serialize(session_id)
        persona.log.info(f"agent_message_chunk: {len(text)} chars, tool_calls={len(serialized_tool_calls)}")

        msg = Message(
            id=message_id,
            body=text,
            time=time(),
            sender=persona.id,
            raw_time=False,
            metadata={"tool_calls": serialized_tool_calls},
        )
        persona.ychat.update_message(msg, append=True, trigger_actions=[])

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
        Handles `session/update` requests from the ACP agent. All event types
        are handled directly here — tool calls, text chunks, and slash commands.
        """
        persona = self._personas_by_session.get(session_id)
        if persona:
            persona.log.info(f"session_update: {type(update).__name__} for session {session_id}")

        if isinstance(update, AvailableCommandsUpdate):
            if not update.available_commands:
                return
            if persona and hasattr(persona, 'acp_slash_commands'):
                persona.acp_slash_commands = update.available_commands
            return

        if persona is None:
            return

        if isinstance(update, ToolCallStart):
            self._tool_call_manager.handle_start(session_id, update, persona)
            return

        if isinstance(update, ToolCallProgress):
            self._tool_call_manager.handle_progress(session_id, update, persona)
            return

        if isinstance(update, AgentMessageChunk):
            self._handle_agent_message_chunk(session_id, update)
            return

    def resolve_permission(self, session_id: str, tool_call_id: str, option_id: str) -> bool:
        """
        Called by the REST endpoint when the user clicks a permission button.
        Delegates to PermissionManager to resolve the pending Future.
        """
        return self._permission_manager.resolve(session_id, tool_call_id, option_id)

    async def request_permission(
        self, options: list[PermissionOption], session_id: str, tool_call: ToolCall, **kwargs: Any
    ) -> RequestPermissionResponse:
        """
        Handles `session/request_permission` requests from the ACP agent.
        """
        persona = self._personas_by_session.get(session_id)
        try:
            if persona:
                locations_paths = (
                    [loc.path for loc in tool_call.locations] if tool_call.locations else None
                )
                persona.log.info(
                    f"request_permission: CALLED session={session_id} "
                    f"tool_call_id={tool_call.tool_call_id} "
                    f"options_count={len(options)} "
                    f"options={[{'id': o.option_id, 'name': o.name, 'kind': o.kind} for o in options]} "
                    f"persona_class={persona.__class__.__name__}"
                )

            # Convert agent-provided options to dicts for the frontend
            permission_options = [
                {"option_id": opt.option_id, "title": opt.name, "description": opt.kind or ""}
                for opt in options
            ]

            # Create a Future via PermissionManager
            future = self._permission_manager.create_request(
                session_id, tool_call.tool_call_id, options=permission_options
            )

            if persona:
                persona.log.info(
                    f"request_permission: {len(options)} agent options -> {len(permission_options)} permission_options"
                )

            # Set the permission options + pending status on the tool call state,
            # then flush to Yjs so the frontend renders the buttons.
            session_state = self._tool_call_manager._ensure_session(session_id)
            tc = session_state.tool_calls.get(tool_call.tool_call_id)
            tc.permission_options = permission_options
            tc.permission_status = "pending"
            tc.session_id = session_id

            # Extract diffs from tool_call.content — agents may send
            # FileEditToolCallContent here rather than on ToolCallStart
            diffs = extract_diffs(tool_call.content)
            if diffs:
                tc.diffs = diffs
            if persona:
                persona.log.info(
                    f"request_permission: diffs={len(diffs) if diffs else 0}"
                    f" content_types={[type(c).__name__ for c in tool_call.content] if tool_call.content else None}"
                )

            if persona:
                self._tool_call_manager.get_or_create_message(session_id, persona)
                self._tool_call_manager._flush_to_message(session_id, persona) #Yjs sync and re-renders with the buttons

            # Suspend until the user clicks a permission button
            selected_option_id = await future
            self._permission_manager.cleanup(session_id, tool_call.tool_call_id)

            tc.permission_status = "resolved"
            tc.selected_option_id = selected_option_id
            if persona:
                self._tool_call_manager._flush_to_message(session_id, persona)

            return RequestPermissionResponse(
                outcome=AllowedOutcome(option_id=selected_option_id, outcome='selected')
            )
        except Exception as e:
            if persona:
                persona.log.error(f"request_permission FAILED: {e}\n{tb_mod.format_exc()}")
            else:
                import logging
                logging.error(f"request_permission FAILED: {e}\n{tb_mod.format_exc()}")
            raise

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
        return await self._terminal_manager.create_terminal(
            command=command,
            session_id=session_id,
            args=args,
            cwd=cwd,
            env=env,
            output_byte_limit=output_byte_limit,
            **kwargs,
        )

    async def terminal_output(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> TerminalOutputResponse:
        return await self._terminal_manager.terminal_output(
            session_id=session_id,
            terminal_id=terminal_id,
            **kwargs,
        )

    async def release_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> ReleaseTerminalResponse | None:
        return await self._terminal_manager.release_terminal(
            session_id=session_id,
            terminal_id=terminal_id,
            **kwargs,
        )

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> WaitForTerminalExitResponse:
        return await self._terminal_manager.wait_for_terminal_exit(
            session_id=session_id,
            terminal_id=terminal_id,
            **kwargs,
        )

    async def kill_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> KillTerminalCommandResponse | None:
        return await self._terminal_manager.kill_terminal(
            session_id=session_id,
            terminal_id=terminal_id,
            **kwargs,
        )

    async def ext_method(self, method: str, params: dict) -> dict:
        raise RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: dict) -> None:
        raise RequestError.method_not_found(method)
