import asyncio
import os
import signal
import sys
from asyncio import Task
from asyncio.subprocess import Process
from typing import Any, ClassVar, Optional

from acp import NewSessionResponse, LoadSessionResponse
from acp.schema import AvailableCommand
from jupyter_ai_persona_manager import BasePersona
from jupyterlab_chat.models import Message

from .default_acp_client import JaiAcpClient
from .telemetry import emit_event, auto_emit_event


class BaseAcpPersona(BasePersona):
    _before_subprocess_future: ClassVar[Task[None] | None] = None
    """
    The task that blocks the agent subprocess from starting until resolved.

    By default this resolves immediately. Developers may define this task in
    `self.before_agent_subprocess()` - see method documentation for details.
    """

    _subprocess_future: ClassVar[Task[Process] | None] = None
    """
    The task that yields the agent subprocess once complete. This is a class
    attribute because multiple instances of the same ACP persona may share an
    ACP agent subprocess.

    Developers should always use `self.get_agent_subprocess()`.
    """

    _client_future: ClassVar[Task[JaiAcpClient] | None] = None
    """
    The future that yields the ACP Client once complete. This is a class
    attribute because multiple instances of the same ACP persona may share an
    ACP client as well. ACP agent subprocesses and clients map 1-to-1.

    Developers should always use `self.get_client()`.
    """

    _client_session_future: Task[NewSessionResponse | LoadSessionResponse]
    """
    The future that yields the ACP client session info. Each instance of an ACP
    persona has a unique session ID, i.e. each chat reserves a unique session.

    Developers should always call `self.get_session_response()` or `self.get_session_id()`.
    """

    _acp_slash_commands: list[AvailableCommand]
    """
    List of slash commands broadcast by the ACP agent in the current session.
    This attribute is set automatically by the default ACP client.
    """

    _MAX_HISTORY_MESSAGES: ClassVar[int] = 50
    """
    Maximum number of recent messages to include in the history context injected
    after load-session recovery. Caps prompt size to avoid exceeding agent
    context window limits.
    """

    def __init__(self, *args, executable: list[str], **kwargs):
        super().__init__(*args, **kwargs)

        self._executable = executable
        self._pending_session_recovery_context: bool = False
        self._was_initially_unauthenticated: bool = False

        # Ensure each subclass has its own subprocess and client by checking if the
        # class variable is defined directly on this class (not inherited)
        if (
            "_before_subprocess_future" not in self.__class__.__dict__
            or self.__class__._before_subprocess_future is None
        ):
            self.__class__._before_subprocess_future = self.event_loop.create_task(
                self.before_agent_subprocess()
            )
        if (
            "_subprocess_future" not in self.__class__.__dict__
            or self.__class__._subprocess_future is None
        ):
            self.__class__._subprocess_future = self.event_loop.create_task(
                self._init_agent_subprocess()
            )
        if (
            "_client_future" not in self.__class__.__dict__
            or self.__class__._client_future is None
        ):
            self.__class__._client_future = self.event_loop.create_task(
                self._init_client()
            )

        self._client_session_future = self.event_loop.create_task(
            self._init_client_session()
        )
        self._acp_slash_commands = []

    async def before_agent_subprocess(self) -> None:
        """
        Defines a task that blocks the ACP agent subprocess from starting until
        resolved. This is useful for when the ACP agent subprocess cannot be
        started until certain requirements are met (e.g. Kiro).

        The `BaseAcpPersona` does not implement this method by default.
        Subclasses are expected to provide a custom implementation of this
        method if required.
        """
        return None

    async def _init_agent_subprocess(
        self, env: Optional[dict[str, str]] = None
    ) -> Process:
        # Wait until user is authenticated
        await self._before_subprocess_future
        self.log.info("Spawning ACP agent subprocess for '%s'.", self.__class__.__name__)
        kwargs: dict[str, Any] = dict(
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=sys.stderr,
            limit=50 * 1024 * 1024,
            start_new_session=True,
        )
        if env is not None:
            kwargs["env"] = env
        process = await asyncio.create_subprocess_exec(*self._executable, **kwargs)
        self.log.info("Spawned ACP agent subprocess for '%s'.", self.__class__.__name__)
        return process

    @auto_emit_event("acp_server_init")
    async def _init_client(self) -> JaiAcpClient:
        agent_subprocess = await self.get_agent_subprocess()
        client = JaiAcpClient(
            agent_subprocess=agent_subprocess, event_loop=self.event_loop
        )
        self.log.info("Initialized ACP client for '%s'.", self.__class__.__name__)
        return client

    def _get_existing_sessions(self) -> dict[str, str]:
        """
        Returns ACP session IDs from this chat's metadata, keyed by persona ID.
        """
        sessions = self.ychat.get_metadata().get("acp_session_ids", {})
        return sessions

    def _record_new_session(self, new_session_id: str) -> None:
        """
        Adds a new ACP session ID into this chat's metadata. Always use this
        method to avoid deleting other clients' sessions.

        Updates the `ychat._ydoc["metadata"]` shared type internally.
        """
        existing_session_ids = self._get_existing_sessions()
        self.ychat.set_metadata(
            "acp_session_ids", {**existing_session_ids, self.id: new_session_id}
        )

    @auto_emit_event("acp_session_init", lambda self: {"session_operation": "load"})
    async def _load_session(self, client, existing_session_id) -> LoadSessionResponse:
        response = await client.load_session(
            persona=self, session_id=existing_session_id
        )
        self.log.info(
            "Loaded existing ACP client session for '%s' with ID '%s'.",
            self.__class__.__name__,
            existing_session_id,
        )
        return response

    @auto_emit_event("acp_session_init", lambda self: {"session_operation": "new"})
    async def _create_session(self, client) -> NewSessionResponse:
        response = await client.create_session(persona=self)
        self.log.info(
            "Initialized new ACP client session for '%s' with ID '%s'.",
            self.__class__.__name__,
            response.session_id,
        )
        self._record_new_session(response.session_id)
        return response

    async def _init_client_session(self) -> NewSessionResponse | LoadSessionResponse:
        # get client
        client = await self.get_client()

        # check for an existing session ID
        existing_session_id = self._get_existing_sessions().get(self.id, None)
        supports_session_load = (await client.get_agent_capabilities()).load_session

        if existing_session_id and supports_session_load:
            try:
                return await self._load_session(client, existing_session_id)
            except Exception:
                self.log.warning(
                    "Failed to load ACP client session for '%s' with ID '%s'; "
                    "creating a new session.",
                    self.__class__.__name__,
                    existing_session_id,
                    exc_info=True,
                )
                self._pending_session_recovery_context = True
                return await self._create_session(client)
        else:
            response = await self._create_session(client)

            # If the user was initially unauthenticated and the session was
            # blocked on auth (e.g. Kiro, Gemini), proactively resume their
            # original request now that the session is ready.
            if self._was_initially_unauthenticated:
                self._was_initially_unauthenticated = False
                await self._resume_after_auth(client, response.session_id)

            return response

    async def _resume_after_auth(
        self, client: JaiAcpClient, session_id: str
    ) -> None:
        """
        After the user signs in, send a hidden prompt with chat history so the
        agent can proactively offer to continue with the user's original request.
        """
        history = self._build_history_context(
            preamble=(
                "You just became available after the user signed in. "
                "Here are the messages they sent while you were unavailable:"
            )
        )
        if history:
            prompt = (
                history + "\n\n"
                "If the user made a request, ask them if they'd like you to "
                "proceed with it. Otherwise, greet them and let them know "
                "you're ready to help."
            )
        else:
            prompt = (
                "You just became available after the user signed in. "
                "Greet them and let them know you're ready to help."
            )

        await client.prompt_and_reply(
            session_id=session_id,
            prompt=prompt,
            root_dir=self.parent.root_dir,
        )

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

    async def get_session_response(self) -> NewSessionResponse | LoadSessionResponse:
        """
        Safely returns the ACP session response for this chat.
        """
        return await self._client_session_future

    async def get_session_id(self) -> str:
        """
        Safely returns the ACP session ID assigned to this chat.
        """
        await self._client_session_future
        # session ID should always be stored in chat metadata after client
        # session was created or loaded.
        session_ids = self._get_existing_sessions()
        assert self.id in session_ids
        return session_ids[self.id]

    async def is_authed(self) -> bool:
        """
        Returns whether the client is authenticated to use this agent. Returns
        `True` by default. Subclasses should override this if possible.
        """
        return True

    async def handle_no_auth(self, message: Message) -> None:
        """
        Method called when the persona receives a message while the user is not
        authenticated. Sets the `_was_initially_unauthenticated` flag so the
        agent can proactively resume the user's request after signing in.

        Subclasses should call `await super().handle_no_auth(message)` first,
        then send a custom message asking the user to log in and perform any
        additional setup (e.g. opening a login terminal).
        """
        self._was_initially_unauthenticated = True
        self.log.warning(
            "[%s] Received message while unauthenticated.", self.__class__.__name__
        )

    def _build_history_context(
        self,
        exclude_id: str | None = None,
        preamble: str = (
            "The previous ACP session could not be loaded. Use this recent chat "
            "transcript as historical context for continuity."
        ),
    ) -> str:
        """
        Builds a plain-text summary of recent chat history for context injection.
        Caps at _MAX_HISTORY_MESSAGES to avoid exceeding agent context window
        limits.
        """
        all_messages = self.ychat.get_messages()
        recent = [
            m for m in all_messages
            if not m.deleted and m.id != exclude_id
        ][-self._MAX_HISTORY_MESSAGES:]
        if not recent:
            return ""
        users = self.ychat.get_users()
        lines = []
        for msg in recent:
            user = users.get(msg.sender)
            name = user.display_name if user else msg.sender
            lines.append(f"{name}: {msg.body}")
        return (
            preamble + "\n"
            "<conversation_history>\n"
            + "\n".join(lines)
            + "\n</conversation_history>"
        )

    @auto_emit_event("acp_chat_message")
    async def process_message(self, message: Message) -> None:
        """
        A default implementation for the `BasePersona.process_message()` method
        for ACP agents.

        This method may be overriden by child classes.
        """
        # If not authenticated, return early
        if not await self.is_authed():
            await self.handle_no_auth(message)
            return

        # If the user was previously unauthenticated, proactively resume their
        # original request instead of processing this message normally.
        if self._was_initially_unauthenticated:
            self._was_initially_unauthenticated = False
            client = await self.get_client()
            session_id = await self.get_session_id()
            await self._resume_after_auth(client, session_id)
            return

        client = await self.get_client()
        session_id = await self.get_session_id()
        prompt = message.body.strip()

        if self._pending_session_recovery_context:
            self._pending_session_recovery_context = False
            history = self._build_history_context(exclude_id=message.id)
            if history:
                emit_event(
                    self.event_logger,
                    "acp_session_recovery",
                    "success",
                    {"persona_class": self.__class__.__name__},
                )
                prompt = history + "\n\nCurrent user message:\n" + prompt

        # Resolve attachments from YChat by ID
        attachments: list[dict] | None = None
        if message.attachments:
            all_attachments = self.ychat.get_attachments()
            resolved = []
            for aid in message.attachments:
                raw = all_attachments.get(aid)
                if raw is None:
                    self.log.warning("Attachment %s not found in YChat", aid)
                    continue
                resolved.append(raw)
            attachments = resolved or None

        await client.prompt_and_reply(
            session_id=session_id,
            prompt=prompt,
            attachments=attachments,
            root_dir=self.parent.root_dir,
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
            "Setting %d slash commands for '%s' in room '%s'.",
            len(commands),
            self.name,
            self.parent.room_id,
        )
        self._acp_slash_commands = commands

    async def restart(self) -> None:
        """
        Restart the ACP agent subprocess shared by all instances of this persona
        class. Kills the existing subprocess, resets class-level futures, and
        re-initializes the subprocess, client, and sessions for all active
        persona instances.
        """
        self.log.info("[restart] Starting for '%s'.", self.__class__.__name__)

        # Collect all persona instances that share this subprocess before teardown
        siblings: list[BaseAcpPersona] = []
        try:
            client = await self.get_client()
            for persona in client._personas_by_session.values():
                if isinstance(persona, BaseAcpPersona) and persona not in siblings:
                    siblings.append(persona)
        except (asyncio.CancelledError, Exception):
            siblings = [self]

        # Step 1: Close connection
        try:
            client = await self.get_client()
            conn = await client.get_connection()
            await conn.close()
        except (asyncio.CancelledError, Exception):
            self.log.warning("[restart] Failed to close connection.", exc_info=True)

        # Step 2: Kill subprocess
        try:
            subprocess = await self.get_agent_subprocess()
            pgid = os.getpgid(subprocess.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                await asyncio.wait_for(subprocess.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                os.killpg(pgid, signal.SIGKILL)
        except (asyncio.CancelledError, ProcessLookupError, PermissionError, OSError):
            pass
        except Exception:
            self.log.warning("[restart] Failed to kill subprocess.", exc_info=True)

        # Step 3: Reset class-level futures
        self.__class__._before_subprocess_future = None
        self.__class__._subprocess_future = None
        self.__class__._client_future = None

        # Step 4: Re-initialize subprocess and client (class-level, done once)
        self.__class__._before_subprocess_future = self.event_loop.create_task(
            self.before_agent_subprocess()
        )
        self.__class__._subprocess_future = self.event_loop.create_task(
            self._init_agent_subprocess()
        )
        self.__class__._client_future = self.event_loop.create_task(
            self._init_client()
        )

        # Step 5: Re-initialize sessions for all sibling persona instances
        for persona in siblings:
            persona._client_session_future = persona.event_loop.create_task(
                persona._init_client_session()
            )

        self.log.info("[restart] Complete for '%s'.", self.__class__.__name__)

    async def shutdown(self):
        if getattr(self, "_shutting_down", False):
            return
        self._shutting_down = True
        await super().shutdown()
        await self._shutdown()

    async def _shutdown(self):
        self.log.info("[shutdown] Starting for '%s'.", self.__class__.__name__)

        # Cancel any pending startup futures to avoid hanging on auth-gated
        # personas (e.g. Kiro, Gemini) that never finished startup.
        for future in [
            self.__class__._before_subprocess_future,
            self.__class__._subprocess_future,
            self.__class__._client_future,
            self._client_session_future,
        ]:
            if isinstance(future, Task) and not future.done():
                future.cancel()

        # Step 1: Session cleanup
        try:
            client = await self.get_client()
            session_id = await self.get_session_id()
            await client.end_session(session_id)
            self.log.info(
                "[shutdown] Step 1: session ended for '%s'.",
                self.__class__.__name__,
            )
        except asyncio.CancelledError:
            pass
        except Exception:
            self.log.warning(
                "[shutdown] Step 1: failed for '%s'.",
                self.__class__.__name__,
                exc_info=True,
            )

        # Skip connection/subprocess teardown if other sessions are still active
        try:
            client = await self.get_client()
            if client.list_sessions():
                self.log.info(
                    "[shutdown] Other sessions still active, skipping subprocess teardown for '%s'.",
                    self.__class__.__name__,
                )
                return
        except (asyncio.CancelledError, Exception):
            pass

        # Step 2: Close connection
        try:
            client = await self.get_client()
            conn = await client.get_connection()
            await conn.close()
            self.log.info(
                "[shutdown] Step 2: connection closed for '%s'.",
                self.__class__.__name__,
            )
        except asyncio.CancelledError:
            pass
        except Exception:
            self.log.warning(
                "[shutdown] Step 2: failed for '%s'.",
                self.__class__.__name__,
                exc_info=True,
            )

        # Step 3: Stop the subprocess gracefully, falling back to SIGKILL
        try:
            subprocess = await self.get_agent_subprocess()
            pgid = os.getpgid(subprocess.pid)
            os.killpg(pgid, signal.SIGINT)
            os.killpg(pgid, signal.SIGTERM)
            try:
                await asyncio.wait_for(subprocess.wait(), timeout=5.0)
                self.log.info(
                    "[shutdown] Step 3: subprocess terminated for '%s'.",
                    self.__class__.__name__,
                )
            except asyncio.TimeoutError:
                os.killpg(pgid, signal.SIGKILL)
                self.log.info(
                    "[shutdown] Step 3: subprocess killed after timeout for '%s'.",
                    self.__class__.__name__,
                )
        except asyncio.CancelledError:
            pass
        except (ProcessLookupError, PermissionError, OSError):
            self.log.info(
                "[shutdown] Step 3: subprocess already dead for '%s'.",
                self.__class__.__name__,
            )
        except Exception:
            self.log.warning(
                "[shutdown] Step 3: failed for '%s'.",
                self.__class__.__name__,
                exc_info=True,
            )

        # Reset class attributes to `None` after cleaning up the global
        # resources they store
        self.__class__._before_subprocess_future = None
        self.__class__._subprocess_future = None
        self.__class__._client_future = None

        self.log.info("[shutdown] Complete for '%s'.", self.__class__.__name__)

    @property
    def event_logger(self):
        """Return the Jupyter EventLogger, or None if unavailable."""
        try:
            from jupyter_events import EventLogger

            extension_app = self.parent.parent  # ExtensionApp instance
            event_logger: EventLogger = extension_app.serverapp.event_logger
            return event_logger
        except Exception:
            self.log.warning("EventLogger unavailable; event logging will be skipped.")
            return None
