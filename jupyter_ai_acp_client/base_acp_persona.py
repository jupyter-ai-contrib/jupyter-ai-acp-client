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
    ModelInfo,
    SessionConfigOptionBoolean,
    SessionConfigOptionSelect,
    SessionMode,
    SessionModelState,
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

# ACP category marking a config option that belongs next to the model picker
# rather than in the general settings list.
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

    _acp_models: list[ModelInfo]
    """
    Models the ACP agent advertises for the current session. Set by the persona
    when a session is created or loaded.
    """

    _acp_current_model_id: Optional[str]
    """
    The model ID currently selected for the ACP session, if any.
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
        self._acp_models = []
        self._acp_current_model_id = None
        self._acp_modes = []
        self._acp_current_mode_id = None
        self._acp_config_options = []
        self._acp_context_usage = None
        self._acp_session_usage = None

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
        self._set_acp_model_state(response.models)
        self._set_acp_mode_state(response.modes)
        self.update_acp_config_options(response.config_options)
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
        self._set_acp_model_state(response.models)
        self._set_acp_mode_state(response.modes)
        self.update_acp_config_options(response.config_options)

        # Reapply a previously selected model so the choice survives session
        # recreation (e.g. a server restart that creates a fresh ACP session).
        stored_model_id = self._get_stored_model_choice()
        advertised_model_ids = {m.model_id for m in self._acp_models}
        if (
            stored_model_id
            and stored_model_id != self._acp_current_model_id
            and stored_model_id in advertised_model_ids
        ):
            try:
                await client.set_session_model(stored_model_id, response.session_id)
                self._acp_current_model_id = stored_model_id
            except Exception:
                self.log.warning(
                    "Failed to reapply stored model '%s' for '%s'.",
                    stored_model_id,
                    self.__class__.__name__,
                    exc_info=True,
                )

        # Reapply a previously selected mode the same way.
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

    @property
    def acp_models(self) -> list[ModelInfo]:
        """
        Models the ACP agent advertises for the current session. Empty when the
        agent does not advertise any. Set by the persona on session create/load.
        """
        return self._acp_models

    @property
    def acp_current_model_id(self) -> Optional[str]:
        """The model ID currently selected for the ACP session, if any."""
        return self._acp_current_model_id

    def _set_acp_model_state(self, models: Optional[SessionModelState]) -> None:
        """
        Store the model state from a `session/new` or `session/load` response.
        `models` is `None` when the agent does not advertise models.
        """
        if models is None:
            self._acp_models = []
            self._acp_current_model_id = None
            return
        self._acp_models = models.available_models
        self._acp_current_model_id = models.current_model_id

    async def set_acp_model(self, model_id: str) -> None:
        """
        Select a model for this persona's ACP session and persist the choice
        with the chat so it survives session recreation.
        """
        client = await self.get_client()
        session_id = await self.get_session_id()
        await client.set_session_model(model_id, session_id)
        self._acp_current_model_id = model_id
        self._record_model_choice(model_id)
        self._sync_awareness_config()

    def _record_model_choice(self, model_id: str) -> None:
        """Persist the selected model in chat metadata, keyed by persona ID."""
        existing = self.ychat.get_metadata().get("acp_models", {})
        self.ychat.set_metadata("acp_models", {**existing, self.id: model_id})

    def _get_stored_model_choice(self) -> Optional[str]:
        """Return the model previously selected for this persona in this chat."""
        return self.ychat.get_metadata().get("acp_models", {}).get(self.id)

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
        self._sync_awareness_config()

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
        self._sync_awareness_config()

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

    ################################################
    # persona-manager awareness API
    #
    # Maps the raw ACP state above onto the persona-manager awareness schema
    # (`ModelConfiguration` + general `SettingConfiguration`s) and implements the
    # three abstract `update_*` methods `BasePersona` requires. This mirrors the
    # normalization in `routes.py:build_controls`, keeping the REST toolbar and
    # the awareness broadcast in agreement.
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

    def _build_awareness_config(
        self,
    ) -> tuple[ModelConfiguration, list[SettingConfiguration]]:
        """
        Build the awareness `ModelConfiguration` and general
        `SettingConfiguration` list from the current raw ACP state.

        Bucketing (mirrors the issue FAQ and `build_controls` dedup):

        - Model: the dedicated `_acp_models`/`_acp_current_model_id` fields are
          the primary source. Only when no models are advertised there do we
          fall back to a `"model"` config option.
        - Model settings (`ModelConfiguration.settings`): config options whose
          ACP category is `"model_config"`.
        - General settings: the ACP mode (as a `SettingConfiguration` keyed by
          `MODE_CONTROL_ID`) followed by every remaining config option.
        """
        has_model = bool(self._acp_models)
        has_mode = bool(self._acp_modes)

        model = ModelConfiguration()
        if has_model:
            model.current = self._acp_current_model_id
            model.options = [
                ModelOption(id=m.model_id, name=m.name, description=m.description)
                for m in self._acp_models
            ]

        model_settings: list[SettingConfiguration] = []
        general_settings: list[SettingConfiguration] = []

        # Mode is surfaced as a general setting with a stable pseudo-ID.
        if has_mode:
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
            # Model advertised as a config option: fold into the dedicated model
            # field when models aren't already advertised there; otherwise drop
            # it as a duplicate of the dedicated field.
            if opt.id == "model":
                if not has_model and isinstance(opt, SessionConfigOptionSelect):
                    model.current = opt.current_value
                    model.options = [
                        ModelOption(id=c.value, name=c.name, description=c.description)
                        for c in _flatten_select_options(opt.options)
                    ]
                continue
            # Mode advertised as a config option: drop when the dedicated mode
            # field already covers it.
            if opt.id == "mode" and has_mode:
                continue
            setting = self._config_option_to_setting(opt)
            if getattr(opt, "category", None) == _MODEL_CONFIG_CATEGORY:
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
        self.set_model_configuration(model)
        self.set_setting_configurations(settings)

    def _sync_awareness_usage(self) -> None:
        """
        Map the raw ACP usage state onto the awareness `Usage` model and merge it
        into the broadcast usage. ACP reports cumulative counts and a live
        context snapshot, so the default replace semantics of `update_usage` are
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
        tokens = self._acp_session_usage
        if tokens is not None:
            usage.input_tokens = tokens.input_tokens
            usage.output_tokens = tokens.output_tokens
            usage.total_tokens = tokens.total_tokens
            usage.cached_read_tokens = tokens.cached_read_tokens
            usage.cached_write_tokens = tokens.cached_write_tokens
            usage.thought_tokens = tokens.thought_tokens
        self.update_usage(usage)

    def _coerce_config_value(self, config_id: str, value: str) -> str | bool:
        """
        Coerce an incoming string setting value to the type the ACP option
        expects. Booleans are advertised to clients as a "true"/"false" select,
        so convert those strings back to a bool before calling the ACP RPC.
        """
        for opt in self._acp_config_options:
            if opt.id == config_id and isinstance(opt, SessionConfigOptionBoolean):
                return value == "true" if isinstance(value, str) else value
        return value

    async def _apply_config_option(self, config_id: str, value: str) -> None:
        """
        Apply a single config option by ID, tolerating unknown IDs and RPC
        failures (mirroring the tolerance of the reapply-stored-config logic).
        """
        if not any(opt.id == config_id for opt in self._acp_config_options):
            self.log.warning(
                "Ignoring unknown config option '%s' for '%s'.",
                config_id,
                self.__class__.__name__,
            )
            return
        try:
            await self.set_acp_config_option(
                config_id, self._coerce_config_value(config_id, value)
            )
        except Exception:
            self.log.warning(
                "Failed to set config option '%s' for '%s'.",
                config_id,
                self.__class__.__name__,
                exc_info=True,
            )

    async def update_model(self, model_id: str) -> None:
        """Switch the ACP session's model, then rebroadcast the awareness config."""
        await self.set_acp_model(model_id)
        self._sync_awareness_config()

    async def update_model_settings(self, settings: dict[str, str]) -> None:
        """
        Apply model settings (ACP `model_config` category config options), keyed
        by option ID, then rebroadcast. Keys not present in the model_config
        category are ignored.
        """
        model_config_ids = {
            opt.id
            for opt in self._acp_config_options
            if getattr(opt, "category", None) == _MODEL_CONFIG_CATEGORY
        }
        for config_id, value in settings.items():
            if config_id in model_config_ids:
                await self._apply_config_option(config_id, value)
        self._sync_awareness_config()

    async def update_settings(self, settings: dict[str, str]) -> None:
        """
        Apply general settings, keyed by setting ID, then rebroadcast. The mode
        pseudo-setting (`MODE_CONTROL_ID`) routes to `set_acp_mode`; everything
        else is treated as a config option.
        """
        for setting_id, value in settings.items():
            if setting_id == MODE_CONTROL_ID:
                try:
                    await self.set_acp_mode(value)
                except Exception:
                    self.log.warning(
                        "Failed to set mode '%s' for '%s'.",
                        value,
                        self.__class__.__name__,
                        exc_info=True,
                    )
                continue
            await self._apply_config_option(setting_id, value)
        self._sync_awareness_config()

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
