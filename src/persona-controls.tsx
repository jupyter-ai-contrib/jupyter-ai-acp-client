import React, {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState
} from 'react';
import {
  Button,
  ListItemText,
  ListSubheader,
  Menu,
  MenuItem,
  Popover,
  Switch
} from '@mui/material';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import CheckIcon from '@mui/icons-material/Check';
import MoreHorizIcon from '@mui/icons-material/MoreHoriz';
import { PageConfig } from '@jupyterlab/coreutils';
import { IMessageMetadata, InputToolbarRegistry } from '@jupyter/chat';
import {
  AcpControl,
  AcpControlChoice,
  AcpUsage,
  EMPTY_USAGE,
  PersonaInfo,
  getPersonas,
  setAcpControl
} from './request';

const SELECTOR_CLASS = 'jp-jupyter-ai-acp-client-personaControls';
const MENU_CLASS = 'jp-jupyter-ai-acp-client-controlMenu';
const USAGE_CLASS = 'jp-jupyter-ai-acp-client-usage';
const NO_ONE_LABEL = 'No one';

// Context-fill fractions at which the chip starts demanding attention: the
// ring and percent turn warn, then error, colored.
const USAGE_WARN_AT = 0.7;
const USAGE_ERROR_AT = 0.9;

/**
 * The chat's default persona ID, advertised by the persona-manager server
 * extension via PageConfig. Used as the initial selection for a chat where the
 * user hasn't picked a persona yet. Empty string if none is configured.
 */
const DEFAULT_PERSONA_ID =
  PageConfig.getOption('jupyter_ai_default_persona') || null;

/**
 * Build the message metadata describing the current picker selection: the
 * target persona, its model, and its settings (mode + config options). This is
 * stamped onto the input model so every outgoing message is self-describing and
 * the PersonaManager can route it. A `null` persona means "no one".
 */
export function buildSelectionMetadata(
  selectedId: string | null,
  controls: AcpControl[]
): IMessageMetadata {
  const metadata: IMessageMetadata = { to_persona: selectedId };

  // With no persona selected ("No one"), there is nothing to configure — don't
  // stamp a model/settings from whatever controls happen to be loaded.
  if (!selectedId) {
    return metadata;
  }

  const modelControl = controls.find(c => c.source === 'model');
  if (modelControl && typeof modelControl.current_value === 'string') {
    metadata.model = modelControl.current_value;
  }

  const settings: { [id: string]: string | boolean } = {};
  for (const control of controls) {
    if (control.source === 'model') {
      continue;
    }
    if (control.current_value !== null) {
      settings[control.id] = control.current_value;
    }
  }
  if (Object.keys(settings).length) {
    metadata.settings = settings;
  }

  return metadata;
}

// Personas register a moment after a chat opens, and an ACP persona's controls
// load asynchronously while its agent session initializes, which can take 20s+
// for a slow agent (e.g. Kiro recovering a session). Poll a bounded number of
// times so the controls appear without needing a first message. The budget
// resets when the active persona changes, so each persona gets a full window.
const POLL_MS = 1500;
const MAX_POLLS = 24;

// Delay for the one trailing refresh after a burst of message updates, long
// enough for the turn's usage report to be stored server-side.
const TRAILING_REFRESH_MS = 1500;

// Width (px) reserved for the overflow ("...") button when not every control
// fits inline.
const OVERFLOW_BTN_WIDTH = 36;

const menuAnchorProps = {
  anchorOrigin: { vertical: 'top', horizontal: 'left' } as const,
  transformOrigin: { vertical: 'bottom', horizontal: 'left' } as const,
  PaperProps: { className: `${MENU_CLASS}-paper` }
};

/**
 * A small round avatar image, or a same-sized spacer to keep labels aligned.
 */
function Avatar(props: { url: string | null | undefined }): JSX.Element {
  if (!props.url) {
    return <span className={`${SELECTOR_CLASS}-avatar-spacer`} />;
  }
  return <img className={`${SELECTOR_CLASS}-avatar`} src={props.url} alt="" />;
}

/**
 * Resolve the label to show on a select control's button.
 */
function currentSelectLabel(control: AcpControl): string {
  return (
    control.choices.find(c => c.value === control.current_value)?.label ??
    (typeof control.current_value === 'string'
      ? control.current_value
      : null) ??
    control.label
  );
}

/**
 * One choice row in a select dropdown. Shows the choice name, and a secondary
 * description only when it adds information (some agents repeat the name as the
 * description, which is just noise). The full description is available on hover.
 */
function ChoiceMenuItem(props: {
  choice: AcpControlChoice;
  selected: boolean;
  onSelect: () => void;
}): JSX.Element {
  const { choice, selected, onSelect } = props;
  const description =
    choice.description &&
    choice.description.trim().toLowerCase() !==
      choice.label.trim().toLowerCase()
      ? choice.description
      : null;
  return (
    <MenuItem
      selected={selected}
      onClick={onSelect}
      title={description ?? undefined}
    >
      <ListItemText
        primary={choice.label}
        secondary={description}
        classes={{
          primary: `${MENU_CLASS}-name`,
          secondary: `${MENU_CLASS}-desc`
        }}
      />
      {selected ? (
        <CheckIcon className={`${MENU_CLASS}-check`} fontSize="small" />
      ) : null}
    </MenuItem>
  );
}

/**
 * A dropdown for a select control (model, mode, or a select config option).
 */
function SelectControl(props: {
  control: AcpControl;
  onSelect: (value: string) => void;
}): JSX.Element {
  const { control, onSelect } = props;
  const [anchor, setAnchor] = useState<HTMLElement | null>(null);
  return (
    <>
      <Button
        className={`${SELECTOR_CLASS} ${SELECTOR_CLASS}-control-btn`}
        size="small"
        variant="text"
        disableRipple
        endIcon={<ArrowDropDownIcon className={`${SELECTOR_CLASS}-arrow`} />}
        onClick={event => setAnchor(event.currentTarget)}
        title={control.label}
      >
        <span className={`${SELECTOR_CLASS}-control-value`}>
          {currentSelectLabel(control)}
        </span>
      </Button>
      <Menu
        anchorEl={anchor}
        open={!!anchor}
        onClose={() => setAnchor(null)}
        {...menuAnchorProps}
      >
        {control.choices.map(choice => (
          <ChoiceMenuItem
            key={choice.value}
            choice={choice}
            selected={choice.value === control.current_value}
            onSelect={() => {
              setAnchor(null);
              onSelect(choice.value);
            }}
          />
        ))}
      </Menu>
    </>
  );
}

/**
 * A compact toggle for a boolean config option.
 */
function BooleanControl(props: {
  control: AcpControl;
  onToggle: (value: boolean) => void;
}): JSX.Element {
  const { control, onToggle } = props;
  const on = control.current_value === true;
  return (
    <button
      type="button"
      className={`${SELECTOR_CLASS}-toggle${on ? ` ${SELECTOR_CLASS}-toggle-on` : ''}`}
      onClick={() => onToggle(!on)}
      title={control.label}
      aria-pressed={on}
    >
      <span className={`${SELECTOR_CLASS}-toggle-label`}>{control.label}</span>
      <span className={`${SELECTOR_CLASS}-toggle-state`}>
        {on ? 'On' : 'Off'}
      </span>
    </button>
  );
}

/**
 * Render one control inline as a dropdown (select) or toggle (boolean).
 */
function ControlItem(props: {
  control: AcpControl;
  onChange: (control: AcpControl, value: string | boolean) => void;
}): JSX.Element {
  const { control, onChange } = props;
  return control.kind === 'boolean' ? (
    <BooleanControl control={control} onToggle={v => onChange(control, v)} />
  ) : (
    <SelectControl control={control} onSelect={v => onChange(control, v)} />
  );
}

/**
 * The overflow popover: controls that did not fit inline, shown as a single flat
 * menu (no nested dropdowns). Each select renders as a `ListSubheader` group
 * label followed by its choices; each boolean is one toggle `MenuItem`. Using
 * MUI primitives keeps the menu keyboard-navigable: `ListSubheader` has no
 * tabindex so arrow-key focus skips it, and the toggle row is a focusable
 * `MenuItem`.
 */
function OverflowMenu(props: {
  controls: AcpControl[];
  anchor: HTMLElement | null;
  onClose: () => void;
  onChange: (control: AcpControl, value: string | boolean) => void;
}): JSX.Element {
  const { controls, anchor, onClose, onChange } = props;
  return (
    <Menu
      anchorEl={anchor}
      open={!!anchor}
      onClose={onClose}
      {...menuAnchorProps}
    >
      {controls.flatMap(control => {
        if (control.kind === 'boolean') {
          const on = control.current_value === true;
          return [
            <MenuItem
              key={control.id}
              className={`${SELECTOR_CLASS}-overflow-toggle`}
              onClick={() => onChange(control, !on)}
            >
              <ListItemText
                primary={control.label}
                classes={{ primary: `${MENU_CLASS}-name` }}
              />
              <Switch
                edge="end"
                size="small"
                checked={on}
                tabIndex={-1}
                disableRipple
              />
            </MenuItem>
          ];
        }
        return [
          <ListSubheader
            key={`${control.id}-label`}
            disableSticky
            className={`${SELECTOR_CLASS}-overflow-subheader`}
          >
            {control.label}
          </ListSubheader>,
          ...control.choices.map(choice => (
            <ChoiceMenuItem
              key={`${control.id}-${choice.value}`}
              choice={choice}
              selected={choice.value === control.current_value}
              onSelect={() => {
                onClose();
                onChange(control, choice.value);
              }}
            />
          ))
        ];
      })}
    </Menu>
  );
}

/**
 * A single-row, width-aware list of controls. Shows as many as fit inline and
 * collapses the rest into an overflow ("...") popover, recomputing on resize.
 */
function ControlsRow(props: {
  controls: AcpControl[];
  onChange: (control: AcpControl, value: string | boolean) => void;
}): JSX.Element {
  const { controls, onChange } = props;
  const rowRef = useRef<HTMLDivElement>(null);
  const measureRef = useRef<HTMLDivElement>(null);
  const overflowBtnRef = useRef<HTMLButtonElement>(null);
  const [visibleCount, setVisibleCount] = useState(controls.length);
  const [overflowAnchor, setOverflowAnchor] = useState<HTMLElement | null>(
    null
  );

  // Re-measure only when a control's displayed width could change (its set of
  // ids or current values), not on every refresh that returns a new array.
  const controlsKey = controls.map(c => `${c.id}:${c.current_value}`).join('|');

  useLayoutEffect(() => {
    const row = rowRef.current;
    const measure = measureRef.current;
    if (!row || !measure) {
      return;
    }
    // The measurement copy exists only to size controls; keep its buttons out of
    // the tab order and the accessibility tree.
    measure.inert = true;
    const GAP = 2;
    let frame = 0;
    const compute = () => {
      const avail = row.clientWidth;
      const widths = (Array.from(measure.children) as HTMLElement[]).map(
        el => el.offsetWidth
      );
      const total = widths.reduce((a, w, i) => a + w + (i ? GAP : 0), 0);
      if (total <= avail) {
        setVisibleCount(widths.length);
        return;
      }
      const reserve =
        (overflowBtnRef.current?.offsetWidth ?? OVERFLOW_BTN_WIDTH) + GAP;
      let used = 0;
      let count = 0;
      for (let i = 0; i < widths.length; i++) {
        const w = widths[i] + (i ? GAP : 0);
        if (used + w + reserve <= avail) {
          used += w;
          count++;
        } else {
          break;
        }
      }
      setVisibleCount(count);
    };
    // A ResizeObserver can fire many times during a drag; coalesce the work to
    // one measurement per animation frame.
    const schedule = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(compute);
    };
    compute();
    const observer = new ResizeObserver(schedule);
    observer.observe(row);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [controlsKey]);

  const visible = controls.slice(0, visibleCount);
  const overflow = controls.slice(visibleCount);

  return (
    <div className={`${SELECTOR_CLASS}-controls`} ref={rowRef}>
      {/* Hidden full-width copy used only to measure each control's width. */}
      <div
        className={`${SELECTOR_CLASS}-controls-measure`}
        ref={measureRef}
        aria-hidden="true"
      >
        {controls.map(control => (
          <ControlItem key={control.id} control={control} onChange={onChange} />
        ))}
      </div>

      {visible.map(control => (
        <ControlItem key={control.id} control={control} onChange={onChange} />
      ))}

      {overflow.length ? (
        <>
          <button
            type="button"
            ref={overflowBtnRef}
            className={`${SELECTOR_CLASS} ${SELECTOR_CLASS}-overflow-btn`}
            onClick={event => setOverflowAnchor(event.currentTarget)}
            title="More controls"
            aria-label="More controls"
          >
            <MoreHorizIcon fontSize="small" />
          </button>
          <OverflowMenu
            controls={overflow}
            anchor={overflowAnchor}
            onClose={() => setOverflowAnchor(null)}
            onChange={onChange}
          />
        </>
      ) : null}
    </div>
  );
}

// All formatters pin the `en` locale so numbers agree with each other and
// with the surrounding English labels.
const exactNumber = new Intl.NumberFormat('en');
const compactNumber = new Intl.NumberFormat('en', {
  notation: 'compact',
  maximumSignificantDigits: 3
});
const costNumber = new Intl.NumberFormat('en', {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2
});

/**
 * Format a token count compactly: 950 stays as-is, 41500 becomes "41.5k",
 * 1240000 becomes "1.24M". `Intl.NumberFormat` picks the tier after rounding,
 * so boundary values like 999500 become "1M" rather than an exponential form.
 * Token values render compactly everywhere (magnitude is what a status surface
 * communicates); the exact count rides on the element's hover title.
 */
export function formatTokens(n: number): string {
  return compactNumber.format(n).replace('K', 'k');
}

/**
 * Format a token count exactly, with thousands separators, for hover titles.
 */
export function formatTokensExact(n: number): string {
  return `${exactNumber.format(n)} tokens`;
}

/**
 * Format a cost amount with its ISO 4217 currency code.
 */
export function formatCost(amount: number, currency: string): string {
  const value = costNumber.format(amount);
  return currency === 'USD' ? `$${value}` : `${value} ${currency}`;
}

/**
 * A ring gauge showing how full the context window is. The track is a muted
 * full circle; the fill arc grows clockwise from 12 o'clock and takes the
 * chip's current color, so the warn/error classes color it via `currentColor`.
 */
function UsageRing(props: { fraction: number }): JSX.Element {
  const radius = 6;
  const circumference = 2 * Math.PI * radius;
  const clamped = Math.min(Math.max(props.fraction, 0), 1);
  return (
    <svg
      className={`${USAGE_CLASS}-ring`}
      viewBox="0 0 16 16"
      width="16"
      height="16"
      aria-hidden="true"
    >
      <circle
        className={`${USAGE_CLASS}-ring-track`}
        cx="8"
        cy="8"
        r={radius}
        fill="none"
        strokeWidth="2"
      />
      <circle
        className={`${USAGE_CLASS}-ring-fill`}
        cx="8"
        cy="8"
        r={radius}
        fill="none"
        strokeWidth="2"
        strokeDasharray={circumference}
        strokeDashoffset={circumference * (1 - clamped)}
        transform="rotate(-90 8 8)"
      />
    </svg>
  );
}

/**
 * A group header in the usage popover: an uppercase label with the group's
 * headline value. Detail rows, when the group has any, follow beneath.
 */
function UsageSection(props: {
  label: string;
  value: string;
  title?: string;
}): JSX.Element {
  return (
    <div className={`${USAGE_CLASS}-section`} title={props.title}>
      <span>{props.label}</span>
      <span className={`${USAGE_CLASS}-section-value`}>{props.value}</span>
    </div>
  );
}

/**
 * One "label: value" detail row in the usage popover. `title` carries the
 * exact value behind a compact one.
 */
function UsageRow(props: {
  label: string;
  value: string;
  title?: string;
}): JSX.Element {
  return (
    <div className={`${USAGE_CLASS}-row`} title={props.title}>
      <span className={`${USAGE_CLASS}-row-label`}>{props.label}</span>
      <span className={`${USAGE_CLASS}-row-value`}>{props.value}</span>
    </div>
  );
}

/**
 * The usage chip for the input toolbar: a ring gauge and percent of the active
 * persona's context-window fill, shown next to the persona it describes, and
 * colored once fill crosses the warn threshold. Hover shows a one-line
 * summary; click opens a popover with the full breakdown (context, session
 * token totals, cost). Renders nothing when the agent has reported no usage
 * at all, so absence reads as unknown rather than empty.
 */
function UsageChip(props: { usage: AcpUsage }): JSX.Element | null {
  const { context, tokens, cost } = props.usage;
  const [anchor, setAnchor] = useState<HTMLElement | null>(null);

  if (!context && !tokens && !cost) {
    return null;
  }

  const fraction =
    context && context.size > 0 ? context.used / context.size : 0;
  const percent = Math.round(fraction * 100);
  const level =
    fraction >= USAGE_ERROR_AT
      ? 'error'
      : fraction >= USAGE_WARN_AT
        ? 'warn'
        : 'ok';

  const summary = [
    context &&
      `Context: ${formatTokens(context.used)} of ${formatTokens(context.size)} tokens (${percent}%)`,
    tokens && `Session tokens: ${formatTokens(tokens.total_tokens)}`,
    cost && `Cost: ${formatCost(cost.amount, cost.currency)}`
  ]
    .filter(Boolean)
    .join('\n');

  return (
    <>
      <button
        type="button"
        className={`${USAGE_CLASS}-chip ${USAGE_CLASS}-${level}`}
        onClick={event => setAnchor(event.currentTarget)}
        title={summary}
        aria-label={context ? `Context ${percent}% used` : 'Usage'}
      >
        {context ? (
          <>
            <UsageRing fraction={fraction} />
            <span className={`${USAGE_CLASS}-pct`}>{percent}%</span>
          </>
        ) : null}
        {!context && tokens ? (
          <span className={`${USAGE_CLASS}-pct`}>
            {formatTokens(tokens.total_tokens)}
          </span>
        ) : null}
      </button>
      <Popover
        anchorEl={anchor}
        open={!!anchor}
        onClose={() => setAnchor(null)}
        {...menuAnchorProps}
      >
        <div className={`${USAGE_CLASS}-card`}>
          {context ? (
            <UsageSection
              label="Context"
              value={`${formatTokens(context.used)} of ${formatTokens(context.size)} (${percent}%)`}
              title={`${exactNumber.format(context.used)} of ${exactNumber.format(context.size)} tokens`}
            />
          ) : null}
          {tokens ? (
            <>
              <UsageSection
                label="Session tokens"
                value={formatTokens(tokens.total_tokens)}
                title={formatTokensExact(tokens.total_tokens)}
              />
              <UsageRow
                label="Input"
                value={formatTokens(tokens.input_tokens)}
                title={formatTokensExact(tokens.input_tokens)}
              />
              <UsageRow
                label="Output"
                value={formatTokens(tokens.output_tokens)}
                title={formatTokensExact(tokens.output_tokens)}
              />
              {tokens.cached_read_tokens !== null ? (
                <UsageRow
                  label="Cache read"
                  value={formatTokens(tokens.cached_read_tokens)}
                  title={formatTokensExact(tokens.cached_read_tokens)}
                />
              ) : null}
              {tokens.cached_write_tokens !== null ? (
                <UsageRow
                  label="Cache write"
                  value={formatTokens(tokens.cached_write_tokens)}
                  title={formatTokensExact(tokens.cached_write_tokens)}
                />
              ) : null}
              {tokens.thought_tokens !== null ? (
                <UsageRow
                  label="Thinking"
                  value={formatTokens(tokens.thought_tokens)}
                  title={formatTokensExact(tokens.thought_tokens)}
                />
              ) : null}
            </>
          ) : null}
          {cost ? (
            <UsageSection
              label="Session cost (est.)"
              value={formatCost(cost.amount, cost.currency)}
              title="Estimated at API list prices"
            />
          ) : null}
        </div>
      </Popover>
    </>
  );
}

/**
 * The persona control for the chat input toolbar. Shows which persona a message
 * will be directed to (with its avatar), lets the user switch it, and, when the
 * selected persona is an ACP persona, renders its session controls (model,
 * mode, and any config options) next to it. Hides itself when the chat has no
 * personas.
 *
 * The selection is owned by the frontend and stamped onto each message's
 * metadata (there is no server-side "active persona"). It's seeded from the
 * default persona advertised over PageConfig.
 */
export function AcpPersonaControls(
  props: InputToolbarRegistry.IToolbarItemProps
): JSX.Element | null {
  const { chatModel, model } = props;
  const [personas, setPersonas] = useState<PersonaInfo[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(
    DEFAULT_PERSONA_ID
  );
  const [controls, setControls] = useState<AcpControl[]>([]);
  const [usage, setUsage] = useState<AcpUsage>(EMPTY_USAGE);
  const [personaAnchor, setPersonaAnchor] = useState<HTMLElement | null>(null);
  const [polls, setPolls] = useState(0);
  const refreshSeq = useRef(0);

  const chatPath = chatModel?.name ?? null;

  // Fetch the persona list and the selected persona's controls. Keyed on the
  // selection so switching personas pulls the right controls.
  const refresh = useCallback(async () => {
    if (!chatPath) {
      return;
    }
    // Refreshes overlap (message events, polling, the trailing refresh, and
    // persona switches), and a slow earlier response must not overwrite a newer
    // one's state.
    const seq = ++refreshSeq.current;
    const response = await getPersonas(chatPath, selectedId);
    if (seq !== refreshSeq.current) {
      return;
    }
    setPersonas(response.personas);
    setControls(response.controls);
    setUsage(response.usage ?? EMPTY_USAGE);
  }, [chatPath, selectedId]);

  useEffect(() => {
    let trailing: number | undefined;
    const onMessages = () => {
      refresh();
      // The agent's usage report is stored when the prompt response resolves,
      // a beat after the turn's final message update, so a refresh driven only
      // by message events always misses it. One trailing refresh after the
      // burst settles picks it up.
      window.clearTimeout(trailing);
      trailing = window.setTimeout(refresh, TRAILING_REFRESH_MS);
    };
    refresh();
    chatModel?.messagesUpdated?.connect(onMessages);
    return () => {
      window.clearTimeout(trailing);
      chatModel?.messagesUpdated?.disconnect(onMessages);
    };
  }, [chatModel, refresh]);

  // Once personas load, reconcile the selection: if the seeded default isn't
  // present in this chat, fall back to the sole persona (if there's exactly
  // one) or to no one, so we never point at a persona the chat doesn't have.
  useEffect(() => {
    if (!personas.length) {
      return;
    }
    if (selectedId && personas.some(p => p.id === selectedId)) {
      return;
    }
    setSelectedId(personas.length === 1 ? personas[0].id : null);
  }, [personas, selectedId]);

  // Reset the poll counter when the chat or the selection changes.
  useEffect(() => {
    setPolls(0);
  }, [chatPath, selectedId]);

  // Poll until the personas register and the selected ACP persona's controls
  // load, so nothing waits on a first message. Depend on primitive flags, not
  // the array references (which change on every refresh), to avoid restarting
  // the timer on unrelated updates.
  const selectedIsAcp =
    personas.find(p => p.id === selectedId)?.is_acp ?? false;
  const needPersonas = personas.length === 0;
  const needControls = selectedIsAcp && controls.length === 0;
  const shouldPoll = needPersonas || needControls;
  useEffect(() => {
    if (!shouldPoll || polls >= MAX_POLLS) {
      return;
    }
    const timer = window.setTimeout(() => {
      setPolls(p => p + 1);
      refresh();
    }, POLL_MS);
    return () => window.clearTimeout(timer);
  }, [shouldPoll, polls, refresh]);

  // Stamp the current selection onto the input model's metadata, so it rides
  // out with the next message and the PersonaManager routes on it. Runs on
  // initial load and whenever the persona or a control value changes. Keyed on
  // a signature of the values (not the array reference, which changes on every
  // refresh) to avoid redundant writes.
  const controlsSignature = controls
    .map(c => `${c.id}:${c.current_value}`)
    .join('|');
  useEffect(() => {
    model.clearMetadata();
    model.updateMetadata(buildSelectionMetadata(selectedId, controls));
  }, [model, selectedId, controlsSignature]);

  // No personas in the chat: nothing to show.
  if (!personas.length) {
    return null;
  }

  const selectedPersona = personas.find(p => p.id === selectedId) ?? null;
  const activeName = selectedPersona?.name ?? null;
  const activeAvatar = selectedPersona?.avatar_url ?? null;
  const personaLabel = activeName ?? NO_ONE_LABEL;

  const handlePersona = (personaId: string | null) => {
    setPersonaAnchor(null);
    // Update the local selection; the effect above refetches this persona's
    // controls and the metadata sync effect stamps it onto the input model.
    setSelectedId(personaId);
    // Usage is per persona; clearing it now keeps the previous persona's
    // numbers from showing next to the new persona's name while the refresh
    // is in flight.
    setUsage(EMPTY_USAGE);
  };

  const handleControl = async (
    control: AcpControl,
    value: string | boolean
  ) => {
    // Optimistic update for immediate feedback; the metadata sync effect picks
    // up the new value and stamps it onto the input model.
    setControls(prev =>
      prev.map(c => (c.id === control.id ? { ...c, current_value: value } : c))
    );
    if (chatPath && selectedId) {
      // POST so the change reaches the persona's live ACP session (which the
      // metadata does not yet drive).
      await setAcpControl(
        chatPath,
        selectedId,
        control.id,
        control.source,
        value
      );
      // Refetch in case the change cascades (e.g. a mode switch alters options).
      await refresh();
    }
  };

  return (
    <div className={`${SELECTOR_CLASS}-group`}>
      <Button
        className={`${SELECTOR_CLASS} ${SELECTOR_CLASS}-persona-btn`}
        size="small"
        variant="text"
        disableRipple
        startIcon={<Avatar url={activeAvatar} />}
        endIcon={<ArrowDropDownIcon className={`${SELECTOR_CLASS}-arrow`} />}
        onClick={event => {
          setPersonaAnchor(event.currentTarget);
          refresh();
        }}
        title="Choose which persona to message"
      >
        <span className={`${SELECTOR_CLASS}-persona`}>{personaLabel}</span>
      </Button>
      <Menu
        anchorEl={personaAnchor}
        open={!!personaAnchor}
        onClose={() => setPersonaAnchor(null)}
        {...menuAnchorProps}
      >
        {personas.map(p => (
          <MenuItem
            key={p.id}
            selected={p.id === selectedId}
            onClick={() => handlePersona(p.id)}
          >
            <Avatar url={p.avatar_url} />
            <ListItemText
              primary={p.name}
              classes={{ primary: `${MENU_CLASS}-name` }}
            />
            {p.id === selectedId ? (
              <CheckIcon className={`${MENU_CLASS}-check`} fontSize="small" />
            ) : null}
          </MenuItem>
        ))}
        <MenuItem
          selected={selectedId === null}
          onClick={() => handlePersona(null)}
        >
          <Avatar url={null} />
          <ListItemText
            primary={NO_ONE_LABEL}
            classes={{ primary: `${MENU_CLASS}-name` }}
          />
          {selectedId === null ? (
            <CheckIcon className={`${MENU_CLASS}-check`} fontSize="small" />
          ) : null}
        </MenuItem>
      </Menu>

      <UsageChip usage={usage} />

      {controls.length ? (
        <>
          <span className={`${SELECTOR_CLASS}-divider`} />
          <ControlsRow controls={controls} onChange={handleControl} />
        </>
      ) : null}
    </div>
  );
}
