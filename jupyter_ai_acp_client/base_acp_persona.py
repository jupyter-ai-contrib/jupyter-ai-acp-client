import asyncio
import html
import os
import signal
import sys
from asyncio import Task
from asyncio.subprocess import Process
from typing import Any, ClassVar, Optional

from acp import NewSessionResponse, LoadSessionResponse
from acp.exceptions import RequestError
from acp.schema import (
    AvailableCommand,
    SessionConfigOptionBoolean,
    SessionConfigOptionSelect,
    SessionMode,
    SessionModeState,
    Usage,
    UsageUpdate,
)
from jupyter_ai_persona_manager import (
    BasePersona,
    ModelConfiguration,
    ModelOption,
    SettingConfiguration,
    SettingOption,
)
from jupyter_ai_persona_manager import Usage as AwarenessUsage
from jupyterlab_chat.models import Message

from .default_acp_client import JaiAcpClient
from .telemetry import emit_event, auto_emit_event

# A single session config option the agent advertises: either a select (one of
# several values) or a boolean toggle. Set via `session/set_config_option`.
AcpConfigOption = SessionConfigOptionSelect | SessionConfigOptionBoolean

# Stable setting ID for the ACP session mode in the awareness settings list.
# Mirrors the REST toolbar's control ID so both surfaces agree. Defined here
# (rather than in `routes.py`) so `routes.py` can import it without a cycle.
MODE_CONTROL_ID = "__mode__"

# ACP config-option categories (https://agentclientprotocol.com/protocol/v1/
# session-config-options#option-categories). Categories are optional semantic
# labels; the client uses them to place an option in the right toolbar group.
# When several options share a category, ACP says to break the tie by array
# order — the earliest wins the prominent slot.
_MODEL_CATEGORY = "model"
_MODE_CATEGORY = "mode"
_MODEL_CONFIG_CATEGORY = "model_config"


def _flatten_select_options(options) -> list:
    """
    Yield the flat choice items of an ACP select option, whether the agent sent
    a flat list or grouped options. Unknown shapes are skipped.
    """
    flat = []
    for item in options or []:
        if hasattr(item, "value"):
            flat.append(item)
        elif hasattr(item, "options"):
            flat.extend(item.options or [])
    return flat


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

    _acp_modes: list[SessionMode]
    """
    Modes the ACP agent advertises for the current session. Set by the persona
    when a session is created or loaded.
    """

    _acp_current_mode_id: Optional[str]
    """
    The mode ID currently selected for the ACP session, if any.
    """

    _acp_config_options: list[AcpConfigOption]
    """
    Session config options (selects and toggles) the ACP agent advertises for
    the current session, each carrying its current value. Set by the persona
    when a session is created or loaded.
    """

    _acp_context_usage: Optional[UsageUpdate]
    """
    How full the agent's context window is right now: tokens currently in
    context, window size, and optional cumulative session cost, from the
    latest `usage_update`. A live snapshot that can decrease (e.g. after
    compaction). Distinct from `acp_session_usage`, which counts total
    session throughput. `None` until the agent sends one; some agents
    never do. Set by the default ACP client.
    """

    _acp_session_usage: Optional[Usage]
    """
    Cumulative token totals for the whole session from the latest prompt
    response: never decreases, and includes cache re-reads, so it grows
    past the context window size. Distinct from `acp_context_usage`, which measures
    current window occupancy. `None` until a prompt response carries
    usage. Set by the default ACP client.
    """

    _acp_context_percent: Optional[float]
    """
    Context-window fill as a bare percentage (0-100), for agents that report
    only a percentage over a vendor extension (e.g. Kiro) instead of a
    token-based `usage_update`. `None` until the agent sends one. Set by the
    default ACP client.
    """

    _acp_metering_total: Optional[float]
    """
    Session cost accumulated from per-turn metering reports, for agents that
    meter usage over a vendor extension (e.g. Kiro's `meteringUsage` credits)
    instead of a standard cost. Summed client-side per turn, so it resets when
    the server restarts even though the agent session itself is resumed.
    `None` until the agent reports metering. Set by the default ACP client.
    """

    _acp_metering_unit: Optional[str]
    """
    The unit of `_acp_metering_total` as the agent names it (e.g. "credits").
    `None` until the agent reports metering with a unit.
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
        self._acp_modes = []
        self._acp_current_mode_id = None
        self._acp_config_options = []
        self._acp_context_usage = None
        self._acp_session_usage = None
        self._acp_context_percent = None
        self._acp_metering_total = None
        self._acp_metering_unit = None
        self._acp_legacy_models = None

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
        self._set_acp_mode_state(response.modes)
        self.update_acp_config_options(response.config_options)
        self.update_acp_legacy_models(client.pop_legacy_models(existing_session_id))
        self._sync_awareness_config()
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
        self._set_acp_mode_state(response.modes)
        self.update_acp_config_options(response.config_options)
        self.update_acp_legacy_models(client.pop_legacy_models(response.session_id))

        # Reapply a previously selected mode so the choice survives session
        # recreation (e.g. a server restart that creates a fresh ACP session).
        # A model selection needs no special handling here: models are ordinary
        # config options now, so they ride the config-option reapply loop below.
        # The exception is a legacy-channel model choice (kiro-cli), which is
        # kept agent-side on the session: a resumed session keeps it, a fresh
        # session resets to the agent's default.
        stored_mode_id = self._get_stored_mode_choice()
        advertised_mode_ids = {m.id for m in self._acp_modes}
        if (
            stored_mode_id
            and stored_mode_id != self._acp_current_mode_id
            and stored_mode_id in advertised_mode_ids
        ):
            try:
                await client.set_session_mode(stored_mode_id, response.session_id)
                self._acp_current_mode_id = stored_mode_id
            except Exception:
                self.log.warning(
                    "Failed to reapply stored mode '%s' for '%s'.",
                    stored_mode_id,
                    self.__class__.__name__,
                    exc_info=True,
                )

        # Reapply previously selected config option values the same way.
        stored_config = self._get_stored_config_choices()
        advertised_options = {opt.id: opt for opt in self._acp_config_options}
        for config_id, value in stored_config.items():
            option = advertised_options.get(config_id)
            if option is None or option.current_value == value:
                continue
            try:
                await client.set_config_option(config_id, value, response.session_id)
                option.current_value = value
            except Exception:
                self.log.warning(
                    "Failed to reapply stored config option '%s' for '%s'.",
                    config_id,
                    self.__class__.__name__,
                    exc_info=True,
                )

        self._sync_awareness_config()
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
            # blocked on auth (e.g. Kiro), proactively resume their
            # original request now that the session is ready.
            if self._was_initially_unauthenticated:
                self._was_initially_unauthenticated = False
                await self._resume_after_auth(client, response.session_id)

            return response

    async def _resume_after_auth(
        self, client: JaiAcpClient, session_id: str
    ) -> None:
        """
        After the user signs in, send a hidden prompt with chat history and a
        prescribed message template for the agent to follow.
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
                "Display the following message to the user, filling in the "
                "bracketed section with a brief summary of their request:\n\n"
                "\"I'm logged in now. I see you asked about [brief summary of "
                "their request]. Would you like me to help you with this now?\""
            )
        else:
            prompt = (
                "Display the following message to the user exactly as written:\n\n"
                "\"I'm logged in now and ready to help. What can I do for you?\""
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

    async def cancel_response(self) -> None:
        """
        Interrupt this persona's in-progress ACP turn.

        Overrides the `BasePersona` no-op: cancels the agent's current prompt
        (via the ACP `session/cancel` notification), finalizes any messages
        streamed so far, rejects pending permissions, and clears the writing
        state.

        Only called for a persona that's actually processing (the cancel handler
        gates on `processing`), so ACP's `session/cancel` — defined only for an
        ongoing prompt turn — always has a turn to cancel. A not-yet-initialized
        session is still ignored defensively.
        """
        try:
            session_id = await self.get_session_id()
        except (AssertionError, KeyError):
            # No ACP session yet -> nothing to cancel.
            return
        client = await self.get_client()
        await client.stop_streaming(session_id)

    @property
    def acp_slash_commands(self) -> list[AvailableCommand]:
        """
        Returns the list of slash commands advertised by the ACP agent in the
        current session.

        This initializes to an empty list, and should be updated **only** by the
        ACP client upon receiving a `session/update` request containing an
        `AvailableCommandsUpdate` payload from the ACP agent. Commands arriving
        through the kiro vendor notification are published over awareness only
        and are not stored here.
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

    @property
    def acp_modes(self) -> list[SessionMode]:
        """
        Modes the ACP agent advertises for the current session. Empty when the
        agent does not advertise any. Set by the persona on session create/load.
        """
        return self._acp_modes

    @property
    def acp_current_mode_id(self) -> Optional[str]:
        """The mode ID currently selected for the ACP session, if any."""
        return self._acp_current_mode_id

    def _set_acp_mode_state(self, modes: Optional[SessionModeState]) -> None:
        """
        Store the mode state from a `session/new` or `session/load` response.
        `modes` is `None` when the agent does not advertise modes.
        """
        if modes is None:
            self._acp_modes = []
            self._acp_current_mode_id = None
            return
        self._acp_modes = modes.available_modes
        self._acp_current_mode_id = modes.current_mode_id

    def update_acp_current_mode(self, mode_id: str) -> None:
        """Record a mode the agent switched to itself (a `current_mode_update`)."""
        self._acp_current_mode_id = mode_id

    async def set_acp_mode(self, mode_id: str) -> None:
        """
        Select a mode for this persona's ACP session and persist the choice
        with the chat so it survives session recreation.
        """
        client = await self.get_client()
        session_id = await self.get_session_id()
        await client.set_session_mode(mode_id, session_id)
        self._acp_current_mode_id = mode_id
        self._record_mode_choice(mode_id)

    def _record_mode_choice(self, mode_id: str) -> None:
        """Persist the selected mode in chat metadata, keyed by persona ID."""
        existing = self.ychat.get_metadata().get("acp_modes", {})
        self.ychat.set_metadata("acp_modes", {**existing, self.id: mode_id})

    def _get_stored_mode_choice(self) -> Optional[str]:
        """Return the mode previously selected for this persona in this chat."""
        return self.ychat.get_metadata().get("acp_modes", {}).get(self.id)

    @property
    def acp_config_options(self) -> list[AcpConfigOption]:
        """
        Session config options the ACP agent advertises for the current session,
        each carrying its current value. Empty when the agent advertises none.
        Set by the persona on session create/load.
        """
        return self._acp_config_options

    def update_acp_config_options(
        self, config_options: Optional[list[AcpConfigOption]]
    ) -> None:
        """
        Store the config options from a `session/new`, `session/load`, or
        `config_option_update` payload. `config_options` is `None` when the agent
        does not advertise any.
        """
        self._acp_config_options = config_options or []

    async def set_acp_config_option(self, config_id: str, value: str | bool) -> None:
        """
        Set a session config option for this persona's ACP session and persist
        the choice with the chat so it survives session recreation.
        """
        client = await self.get_client()
        session_id = await self.get_session_id()
        await client.set_config_option(config_id, value, session_id)
        for option in self._acp_config_options:
            if option.id == config_id:
                option.current_value = value
                break
        self._record_config_choice(config_id, value)

    def _record_config_choice(self, config_id: str, value: str | bool) -> None:
        """Persist a config option value in chat metadata, keyed by persona ID."""
        all_choices = self.ychat.get_metadata().get("acp_config_options", {})
        persona_choices = {**all_choices.get(self.id, {}), config_id: value}
        self.ychat.set_metadata(
            "acp_config_options", {**all_choices, self.id: persona_choices}
        )

    def _get_stored_config_choices(self) -> dict[str, str | bool]:
        """Return config option values previously selected for this persona here."""
        return self.ychat.get_metadata().get("acp_config_options", {}).get(self.id, {})

    @property
    def acp_legacy_models(self) -> Optional[dict]:
        """
        The legacy `models` payload (`currentModelId` plus `availableModels`)
        for agents that advertise models the pre-v1 way, outside config options
        (e.g. kiro-cli). `None` when the agent advertises no such payload. Set
        by the persona on session create/load from what the client captured off
        the raw session response.
        """
        return self._acp_legacy_models

    def update_acp_legacy_models(self, models: Optional[dict]) -> None:
        """Store the legacy models payload captured for this persona's session."""
        self._acp_legacy_models = models

    ################################################
    # persona-manager awareness API
    #
    # Maps the raw ACP state above onto the persona-manager awareness schema
    # (`ModelConfiguration` + general `SettingConfiguration`s) and implements the
    # `update_*` methods from `BasePersona` over the ACP RPCs. `BasePersona`
    # records the new current values and rebroadcasts after an `update_*`; the
    # `_sync_awareness_config` here publishes the *full* configuration (options
    # included) when the session is (re)initialized or the agent changes it
    # itself. The awareness broadcast is the single source of truth for session
    # info; the frontend reads it directly rather than polling a REST endpoint.
    ################################################
    def _config_option_to_setting(
        self, opt: AcpConfigOption
    ) -> SettingConfiguration:
        """
        Convert an ACP config option into a `SettingConfiguration`.

        Selects map their choices to `SettingOption`s directly. Booleans are
        represented as a uniform two-option select ("true"/"false"); the ACP
        `current_value` bool is stringified to match.
        """
        if isinstance(opt, SessionConfigOptionSelect):
            return SettingConfiguration(
                id=opt.id,
                current=opt.current_value,
                name=opt.name,
                description=opt.description,
                options=[
                    SettingOption(id=c.value, name=c.name, description=c.description)
                    for c in _flatten_select_options(opt.options)
                ],
            )
        # SessionConfigOptionBoolean
        current = None
        if opt.current_value is not None:
            current = "true" if opt.current_value else "false"
        return SettingConfiguration(
            id=opt.id,
            current=current,
            name=opt.name,
            description=opt.description,
            options=[
                SettingOption(id="true", name="True"),
                SettingOption(id="false", name="False"),
            ],
        )

    def _model_config_option(self) -> Optional[SessionConfigOptionSelect]:
        """
        The config option backing the model picker: the first select with
        category `"model"` (or, as a fallback, id `"model"`).

        ACP models are ordinary config options now — the never-stabilized
        `session/set_model` API and its response fields were removed from the
        protocol. When several options share the `"model"` category, ACP breaks
        the tie by array order, so the earliest wins the prominent model slot;
        any later model-category options fall through to the general settings
        list. `None` when the agent advertises no model option.
        """
        for opt in self._acp_config_options:
            if isinstance(opt, SessionConfigOptionSelect) and (
                opt.category == _MODEL_CATEGORY or opt.id == "model"
            ):
                return opt
        return None

    def _mode_config_option(self) -> Optional[SessionConfigOptionSelect]:
        """
        The config option backing the mode selector: the first select with
        category `"mode"` (or, as a fallback, id `"mode"`).

        ACP v1 lets an agent expose its mode either through the dedicated
        `session/set_mode` state or as a config option; it tells clients to
        prefer config options and respect a `"mode"` category. So a mode config
        option wins over the dedicated mode state (see `_build_awareness_config`),
        and same-category ties resolve by array order. `None` when no config
        option advertises a mode.
        """
        for opt in self._acp_config_options:
            if isinstance(opt, SessionConfigOptionSelect) and (
                opt.category == _MODE_CATEGORY or opt.id == "mode"
            ):
                return opt
        return None

    def _build_awareness_config(
        self,
    ) -> tuple[ModelConfiguration, list[SettingConfiguration]]:
        """
        Build the awareness `ModelConfiguration` and general
        `SettingConfiguration` list from the current raw ACP state.

        Bucketing by ACP config-option category:

        - Model: the `"model"`-category config option (`_model_config_option`);
          when no config option advertises a model, the legacy `models` payload
          for agents on the pre-v1 channel (e.g. kiro-cli).
        - Mode: the `"mode"`-category config option if present, else the
          dedicated `session/set_mode` state — surfaced as a general setting
          keyed by `MODE_CONTROL_ID`. A mode config option is preferred, so a
          duplicate mode advertised through both channels appears only once.
        - Model settings (`ModelConfiguration.settings`): config options whose
          category is `"model_config"`.
        - General settings: every remaining config option.

        The one option consumed as the prominent model picker or mode selector
        is not also shown as a general setting; any *additional* same-category
        options are (ACP resolves such ties by array order — earliest wins the
        prominent slot).
        """
        model_opt = self._model_config_option()
        mode_opt = self._mode_config_option()

        model = ModelConfiguration()
        if model_opt is not None:
            model.current = model_opt.current_value
            model.options = [
                ModelOption(id=c.value, name=c.name, description=c.description)
                for c in _flatten_select_options(model_opt.options)
            ]
        elif self._acp_legacy_models:
            model.current = self._acp_legacy_models.get("currentModelId")
            model.options = [
                ModelOption(
                    id=m["modelId"],
                    name=m["name"] if isinstance(m.get("name"), str) else m["modelId"],
                    description=m["description"]
                    if isinstance(m.get("description"), str)
                    else None,
                )
                for m in (self._acp_legacy_models.get("availableModels") or [])
                if isinstance(m, dict) and isinstance(m.get("modelId"), str)
            ]

        model_settings: list[SettingConfiguration] = []
        general_settings: list[SettingConfiguration] = []

        # Mode is surfaced as a general setting with a stable pseudo-ID. Prefer a
        # mode config option; fall back to the dedicated set_mode state.
        if mode_opt is not None:
            general_settings.append(
                SettingConfiguration(
                    id=MODE_CONTROL_ID,
                    current=mode_opt.current_value,
                    name=mode_opt.name or "Mode",
                    description=mode_opt.description,
                    options=[
                        SettingOption(id=c.value, name=c.name, description=c.description)
                        for c in _flatten_select_options(mode_opt.options)
                    ],
                )
            )
        elif self._acp_modes:
            general_settings.append(
                SettingConfiguration(
                    id=MODE_CONTROL_ID,
                    current=self._acp_current_mode_id,
                    name="Mode",
                    options=[
                        SettingOption(id=m.id, name=m.name, description=m.description)
                        for m in self._acp_modes
                    ],
                )
            )

        for opt in self._acp_config_options:
            # The option consumed as the model picker / mode selector is not also
            # shown as a general setting.
            if opt is model_opt or opt is mode_opt:
                continue
            setting = self._config_option_to_setting(opt)
            if opt.category == _MODEL_CONFIG_CATEGORY:
                model_settings.append(setting)
            else:
                general_settings.append(setting)

        model.settings = model_settings
        return model, general_settings

    def _sync_awareness_config(self) -> None:
        """
        Rebuild the awareness model + settings configuration from current ACP
        state and broadcast it. Called whenever the ACP state is (re)initialized
        or changes (session create/load, a control is set, or the agent switches
        mode/config itself).
        """
        model, settings = self._build_awareness_config()
        self.report_model_configuration(model)
        self.report_settings_configuration(settings)

    def _sync_awareness_usage(self) -> None:
        """
        Map the raw ACP usage state onto the awareness `Usage` model and merge it
        into the broadcast usage. ACP reports cumulative counts and a live
        context snapshot, so the default replace semantics of `report_usage` are
        correct here.
        """
        usage = AwarenessUsage()
        context = self._acp_context_usage
        if context is not None:
            usage.context_tokens = context.used
            usage.context_size = context.size
            if context.cost is not None:
                usage.cost_amount = context.cost.amount
                usage.cost_currency = context.cost.currency
        if self._acp_context_percent is not None:
            usage.context_percent = self._acp_context_percent
        # Metered cost (e.g. kiro credits) fills the cost slots only when the
        # standard channel didn't provide a cost.
        if usage.cost_amount is None and self._acp_metering_total is not None:
            usage.cost_amount = self._acp_metering_total
            usage.cost_currency = self._acp_metering_unit or "credits"
        tokens = self._acp_session_usage
        if tokens is not None:
            usage.input_tokens = tokens.input_tokens
            usage.output_tokens = tokens.output_tokens
            usage.total_tokens = tokens.total_tokens
            usage.cached_read_tokens = tokens.cached_read_tokens
            usage.cached_write_tokens = tokens.cached_write_tokens
            usage.thought_tokens = tokens.thought_tokens
        self.report_usage(usage)

    def _coerce_config_value(self, config_id: str, value: str) -> str | bool:
        """
        Coerce an incoming string setting value to the type the ACP option
        expects. Booleans are advertised to clients as a "true"/"false" select,
        so convert those strings back to a bool before calling the ACP RPC.
        """
        for opt in self._acp_config_options:
            if opt.id == config_id and isinstance(opt, SessionConfigOptionBoolean):
                return value == "true"
        return value

    async def update_model(self, model_id: str) -> None:
        """
        Switch the ACP session's model. `BasePersona` rebroadcasts.

        Models are config options in ACP v1 (`session/set_model` was removed
        from the protocol), so the choice is applied through the backing model
        config option, matching how `_build_awareness_config` sourced it.
        Agents on the legacy channel (e.g. kiro-cli) advertise no model config
        option but still serve `session/set_model`, so when legacy models are
        stored the choice goes through that request instead. The legacy choice
        is not persisted with the chat: the agent keeps it on the session,
        which is resumed by ID.
        """
        model_opt = self._model_config_option()
        if model_opt is None and self._acp_legacy_models:
            client = await self.get_client()
            session_id = await self.get_session_id()
            await client.set_session_model(model_id, session_id)
            self._acp_legacy_models["currentModelId"] = model_id
            return
        config_id = model_opt.id if model_opt is not None else "model"
        await self.set_acp_config_option(
            config_id, self._coerce_config_value(config_id, model_id)
        )

    async def update_model_settings(self, settings: dict[str, str | None]) -> None:
        """
        Apply model settings — ACP `model_config` category config options — by
        setting each as a config option. `BasePersona` passes only the settings
        that changed and rebroadcasts afterward.
        """
        for config_id, value in settings.items():
            await self.set_acp_config_option(
                config_id, self._coerce_config_value(config_id, value)
            )

    async def update_settings(self, settings: dict[str, str | None]) -> None:
        """
        Apply general settings. The mode pseudo-setting (`MODE_CONTROL_ID`)
        routes to the mode config option if the agent advertised one, else to
        the dedicated `session/set_mode`; everything else is a config option.
        `BasePersona` passes only the settings that changed and rebroadcasts.
        """
        for setting_id, value in settings.items():
            if setting_id == MODE_CONTROL_ID:
                mode_opt = self._mode_config_option()
                if mode_opt is not None:
                    await self.set_acp_config_option(
                        mode_opt.id, self._coerce_config_value(mode_opt.id, value)
                    )
                else:
                    await self.set_acp_mode(value)
            else:
                await self.set_acp_config_option(
                    setting_id, self._coerce_config_value(setting_id, value)
                )

    @property
    def acp_context_usage(self) -> Optional[UsageUpdate]:
        """
        How full the agent's context window is right now: tokens currently in
        context, window size, and optional cumulative session cost, from the
        latest `usage_update`. A live snapshot that can decrease (e.g. after
        compaction). Distinct from `acp_session_usage`, which counts total
        session throughput. `None` when the agent has not reported any.
        """
        return self._acp_context_usage

    def update_acp_context_usage(self, usage: UsageUpdate) -> None:
        """Record a `usage_update` received from the ACP agent."""
        self._acp_context_usage = usage

    @property
    def acp_session_usage(self) -> Optional[Usage]:
        """
        Cumulative token totals for the whole session from the latest prompt
        response: never decreases, and includes cache re-reads, so it grows
        past the context window size. Distinct from `acp_context_usage`, which measures
        current window occupancy. `None` when no prompt response has carried
        usage.
        """
        return self._acp_session_usage

    def update_acp_session_usage(self, usage: Usage) -> None:
        """Record the token usage carried on a completed prompt response."""
        self._acp_session_usage = usage

    @property
    def acp_context_percent(self) -> Optional[float]:
        """
        Context-window fill as a bare percentage (0-100), for agents that
        report only a percentage instead of a token-based `usage_update`.
        `None` when the agent has not reported any.
        """
        return self._acp_context_percent

    def update_acp_context_percent(self, percent: float) -> None:
        """Record a percentage-only context fill report from the ACP agent."""
        self._acp_context_percent = percent

    @property
    def acp_metering_total(self) -> Optional[float]:
        """
        Session cost accumulated from the agent's per-turn metering reports
        (see `_acp_metering_total`). `None` when the agent has not metered.
        """
        return self._acp_metering_total

    def add_acp_metering(self, amount: float, unit: Optional[str] = None) -> None:
        """
        Add one turn's metered cost to the session total. The first unit the
        agent names sticks for the whole session, so a later report that omits
        the unit (or names another) never relabels the accumulated total.
        """
        self._acp_metering_total = (self._acp_metering_total or 0.0) + amount
        if unit and self._acp_metering_unit is None:
            self._acp_metering_unit = unit

    async def handle_uncaught_exception(self, exc: Exception) -> None:
        """Show structured error info for ACP RequestError inside the standard dropdown."""
        if not isinstance(exc, RequestError):
            await super().handle_uncaught_exception(exc)
            return

        import json
        import traceback

        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        error_msg = str(exc)
        if len(error_msg) > 120:
            error_msg = error_msg[:120] + "…"
        summary = f"Error {exc.code}: {html.escape(error_msg)}"

        # Build inner content
        sections = [f"**Error code:** {exc.code}\n\n**Message:** {html.escape(str(exc))}"]

        if exc.data is not None:
            try:
                data_str = json.dumps(exc.data, indent=2)
            except (TypeError, ValueError):
                data_str = str(exc.data)
            sections.append(f"**Data:**\n\n```json\n{data_str}\n```")

        sections.append(f"**Traceback:**\n\n```\n{tb}```")

        inner = "\n\n".join(sections)

        body = (
            f"An error occurred while processing your message.\n\n"
            f'<details class="jp-jai-error-details">\n'
            f"<summary>Error details ({summary})</summary>\n\n"
            f"{inner}\n"
            f"</details>"
        )
        self.send_message(body)

    async def shutdown(self):
        if getattr(self, "_shutting_down", False):
            return
        self._shutting_down = True
        await super().shutdown()
        await self._shutdown()

    async def _shutdown(self):
        self.log.info("[shutdown] Starting for '%s'.", self.__class__.__name__)

        # Cancel any pending startup futures to avoid hanging on auth-gated
        # personas (e.g. Kiro) that never finished startup.
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
