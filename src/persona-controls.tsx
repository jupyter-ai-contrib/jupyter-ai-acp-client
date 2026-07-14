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
  Popover
} from '@mui/material';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import CheckIcon from '@mui/icons-material/Check';
import MoreHorizIcon from '@mui/icons-material/MoreHoriz';
import { PageConfig } from '@jupyterlab/coreutils';
import { InputToolbarRegistry } from '@jupyter/chat';
import { IAwareness } from '@jupyter/ydoc';
import {
  EMPTY_USAGE,
  PersonaAwareness,
  PersonaManagerAwareness,
  PersonaOption,
  SettingConfiguration,
  Usage
} from '@jupyter-ai/persona-manager';
import {
  PersonaSettings,
  buildMessageMetadata,
  emptyPersonaSettings
} from './metadata';

const SELECTOR_CLASS = 'jp-jupyter-ai-acp-client-personaControls';
const MENU_CLASS = 'jp-jupyter-ai-acp-client-controlMenu';
const USAGE_CLASS = 'jp-jupyter-ai-acp-client-usage';
const NO_ONE_LABEL = 'No one';

// Stable control ID for the model selector (setting IDs are used verbatim).
const MODEL_CONTROL_ID = '__model__';

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

// Width (px) reserved for the overflow ("...") button when not every control
// fits inline.
const OVERFLOW_BTN_WIDTH = 36;

const menuAnchorProps = {
  anchorOrigin: { vertical: 'top', horizontal: 'left' } as const,
  transformOrigin: { vertical: 'bottom', horizontal: 'left' } as const,
  PaperProps: { className: `${MENU_CLASS}-paper` }
};

/**
 * A UI control for one control (the model, a model setting, or a general
 * setting). It carries the persona's current value (from awareness) and the
 * user's per-message selection (null = use the persona's default).
 */
export type Control = {
  id: string;
  kind: 'model' | 'model_setting' | 'setting';
  label: string;
  /** The persona's current value, from awareness. Null when on its default. */
  current: string | null;
  /** The user's selection for this message. Null means "use the default". */
  selection: string | null;
  options: { id: string; name: string; description: string | null }[];
};

/**
 * Convert a persona's awareness `SettingConfiguration` into a `Control` of the
 * given kind, seeding the user selection from the current per-persona
 * selection (defaulting to null = default).
 */
function settingToControl(
  setting: SettingConfiguration,
  kind: 'model_setting' | 'setting',
  selection: string | null
): Control {
  return {
    id: setting.id,
    kind,
    label: setting.name ?? setting.id,
    current: setting.current,
    selection,
    options: setting.options.map(o => ({
      id: o.id,
      name: o.name ?? o.id,
      description: o.description
    }))
  };
}

/**
 * Build the list of controls to render for a persona: the model control (when the
 * persona advertises models), its model settings, then its general settings.
 * The user's current selection seeds each control's `selection`.
 */
export function buildControls(
  persona: PersonaAwareness | null,
  settings: PersonaSettings
): Control[] {
  if (!persona) {
    return [];
  }
  const controls: Control[] = [];
  if (persona.model.options.length) {
    controls.push({
      id: MODEL_CONTROL_ID,
      kind: 'model',
      label: 'Model',
      current: persona.model.current,
      selection: settings.modelId,
      options: persona.model.options.map(o => ({
        id: o.id,
        name: o.name ?? o.id,
        description: o.description
      }))
    });
  }
  for (const setting of persona.model.settings) {
    controls.push(
      settingToControl(
        setting,
        'model_setting',
        settings.modelSettings[setting.id] ?? null
      )
    );
  }
  for (const setting of persona.settings) {
    controls.push(
      settingToControl(
        setting,
        'setting',
        settings.settings[setting.id] ?? null
      )
    );
  }
  return controls;
}

/**
 * Fold a changed control value into the user's `PersonaSettings`, keyed by the
 * control's kind. A null value resets that control to the persona's default.
 */
export function applyControlChange(
  settings: PersonaSettings,
  control: Control,
  value: string | null
): PersonaSettings {
  const next: PersonaSettings = {
    modelId: settings.modelId,
    modelSettings: { ...settings.modelSettings },
    settings: { ...settings.settings }
  };
  if (control.kind === 'model') {
    next.modelId = value;
  } else if (control.kind === 'model_setting') {
    next.modelSettings[control.id] = value;
  } else {
    next.settings[control.id] = value;
  }
  return next;
}

/**
 * The value a control currently reflects: the user's selection if they picked
 * one, otherwise the persona's current value (the default).
 */
function effectiveValue(control: Control): string | null {
  return control.selection ?? control.current;
}

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
 * The label shown on a control's button: the name of its effective value, or the
 * control's own label when nothing resolves (no options, no current value).
 */
function currentControlLabel(control: Control): string {
  const value = effectiveValue(control);
  const option = control.options.find(o => o.id === value);
  return option?.name ?? value ?? control.label;
}

/**
 * One choice row in a control dropdown. Shows the choice name, and a secondary
 * description only when it adds information (some agents repeat the name as the
 * description, which is just noise). The full description is available on hover.
 */
function ChoiceMenuItem(props: {
  primary: string;
  description: string | null;
  selected: boolean;
  onSelect: () => void;
}): JSX.Element {
  const { primary, selected, onSelect } = props;
  const description =
    props.description &&
    props.description.trim().toLowerCase() !== primary.trim().toLowerCase()
      ? props.description
      : null;
  return (
    <MenuItem
      selected={selected}
      onClick={onSelect}
      title={description ?? undefined}
    >
      <ListItemText
        primary={primary}
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
 * The "Default" row shown at the top of every control. Selecting it sets the
 * user's value to null, i.e. "use the persona's current value". Its label shows
 * that current value so the user sees what the default points to.
 */
function defaultChoiceLabel(control: Control): string {
  const current = control.options.find(o => o.id === control.current);
  const name = current?.name ?? control.current;
  return name ? `Default (${name})` : 'Default';
}

/**
 * A dropdown for a control. The first row is "Default" (selection = null); the
 * rest are the persona's advertised options (selection = that option's id).
 */
function ControlItem(props: {
  control: Control;
  onSelect: (value: string | null) => void;
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
          {currentControlLabel(control)}
        </span>
      </Button>
      <Menu
        anchorEl={anchor}
        open={!!anchor}
        onClose={() => setAnchor(null)}
        {...menuAnchorProps}
      >
        <ChoiceMenuItem
          primary={defaultChoiceLabel(control)}
          description={null}
          selected={control.selection === null}
          onSelect={() => {
            setAnchor(null);
            onSelect(null);
          }}
        />
        {control.options.map(option => (
          <ChoiceMenuItem
            key={option.id}
            primary={option.name}
            description={option.description}
            selected={control.selection === option.id}
            onSelect={() => {
              setAnchor(null);
              onSelect(option.id);
            }}
          />
        ))}
      </Menu>
    </>
  );
}

/**
 * The overflow popover: controls that did not fit inline, shown as a single flat
 * menu (no nested dropdowns). Each control renders as a `ListSubheader` group
 * label followed by its Default row and choices. Using MUI primitives keeps the
 * menu keyboard-navigable: `ListSubheader` has no tabindex so arrow-key focus
 * skips it.
 */
function OverflowMenu(props: {
  controls: Control[];
  anchor: HTMLElement | null;
  onClose: () => void;
  onChange: (control: Control, value: string | null) => void;
}): JSX.Element {
  const { controls, anchor, onClose, onChange } = props;
  return (
    <Menu
      anchorEl={anchor}
      open={!!anchor}
      onClose={onClose}
      {...menuAnchorProps}
    >
      {controls.flatMap(control => [
        <ListSubheader
          key={`${control.id}-label`}
          disableSticky
          className={`${SELECTOR_CLASS}-overflow-subheader`}
        >
          {control.label}
        </ListSubheader>,
        <ChoiceMenuItem
          key={`${control.id}-default`}
          primary={defaultChoiceLabel(control)}
          description={null}
          selected={control.selection === null}
          onSelect={() => {
            onClose();
            onChange(control, null);
          }}
        />,
        ...control.options.map(option => (
          <ChoiceMenuItem
            key={`${control.id}-${option.id}`}
            primary={option.name}
            description={option.description}
            selected={control.selection === option.id}
            onSelect={() => {
              onClose();
              onChange(control, option.id);
            }}
          />
        ))
      ])}
    </Menu>
  );
}

/**
 * A single-row, width-aware list of controls. Shows as many as fit inline and
 * collapses the rest into an overflow ("...") popover, recomputing on resize.
 */
function ControlsRow(props: {
  controls: Control[];
  onChange: (control: Control, value: string | null) => void;
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
  // ids or effective values), not on every re-render.
  const controlsKey = controls
    .map(p => `${p.id}:${effectiveValue(p)}`)
    .join('|');

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
          <ControlItem
            key={control.id}
            control={control}
            onSelect={v => onChange(control, v)}
          />
        ))}
      </div>

      {visible.map(control => (
        <ControlItem
          key={control.id}
          control={control}
          onSelect={v => onChange(control, v)}
        />
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
 * The usage chip for the input toolbar: a ring gauge and percent of the
 * persona's context-window fill, colored once fill crosses the warn threshold.
 * Hover shows a one-line summary; click opens a popover with the full breakdown
 * (context, session token totals, cost). Renders nothing when the persona has
 * reported no usage at all, so absence reads as unknown rather than empty.
 */
export function UsageChip(props: { usage: Usage }): JSX.Element | null {
  const usage = props.usage;
  const [anchor, setAnchor] = useState<HTMLElement | null>(null);

  const hasContext =
    usage.context_tokens !== null && usage.context_size !== null;
  const hasTokens = usage.total_tokens !== null;
  const hasCost = usage.cost_amount !== null && usage.cost_currency !== null;

  if (!hasContext && !hasTokens && !hasCost) {
    return null;
  }

  const fraction =
    hasContext && (usage.context_size as number) > 0
      ? (usage.context_tokens as number) / (usage.context_size as number)
      : 0;
  const percent = Math.round(fraction * 100);
  const level =
    fraction >= USAGE_ERROR_AT
      ? 'error'
      : fraction >= USAGE_WARN_AT
        ? 'warn'
        : 'ok';

  const summary = [
    hasContext &&
      `Context: ${formatTokens(usage.context_tokens as number)} of ${formatTokens(usage.context_size as number)} tokens (${percent}%)`,
    hasTokens &&
      `Session tokens: ${formatTokens(usage.total_tokens as number)}`,
    hasCost &&
      `Cost: ${formatCost(usage.cost_amount as number, usage.cost_currency as string)}`
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
        aria-label={hasContext ? `Context ${percent}% used` : 'Usage'}
      >
        {hasContext ? (
          <>
            <UsageRing fraction={fraction} />
            <span className={`${USAGE_CLASS}-pct`}>{percent}%</span>
          </>
        ) : null}
        {!hasContext && hasTokens ? (
          <span className={`${USAGE_CLASS}-pct`}>
            {formatTokens(usage.total_tokens as number)}
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
          {hasContext ? (
            <UsageSection
              label="Context"
              value={`${formatTokens(usage.context_tokens as number)} of ${formatTokens(usage.context_size as number)} (${percent}%)`}
              title={`${exactNumber.format(usage.context_tokens as number)} of ${exactNumber.format(usage.context_size as number)} tokens`}
            />
          ) : null}
          {hasTokens ? (
            <>
              <UsageSection
                label="Session tokens"
                value={formatTokens(usage.total_tokens as number)}
                title={formatTokensExact(usage.total_tokens as number)}
              />
              {usage.input_tokens !== null ? (
                <UsageRow
                  label="Input"
                  value={formatTokens(usage.input_tokens)}
                  title={formatTokensExact(usage.input_tokens)}
                />
              ) : null}
              {usage.output_tokens !== null ? (
                <UsageRow
                  label="Output"
                  value={formatTokens(usage.output_tokens)}
                  title={formatTokensExact(usage.output_tokens)}
                />
              ) : null}
              {usage.cached_read_tokens !== null ? (
                <UsageRow
                  label="Cache read"
                  value={formatTokens(usage.cached_read_tokens)}
                  title={formatTokensExact(usage.cached_read_tokens)}
                />
              ) : null}
              {usage.cached_write_tokens !== null ? (
                <UsageRow
                  label="Cache write"
                  value={formatTokens(usage.cached_write_tokens)}
                  title={formatTokensExact(usage.cached_write_tokens)}
                />
              ) : null}
              {usage.thought_tokens !== null ? (
                <UsageRow
                  label="Thinking"
                  value={formatTokens(usage.thought_tokens)}
                  title={formatTokensExact(usage.thought_tokens)}
                />
              ) : null}
            </>
          ) : null}
          {hasCost ? (
            <UsageSection
              label="Session cost (est.)"
              value={formatCost(
                usage.cost_amount as number,
                usage.cost_currency as string
              )}
              title="Estimated at API list prices"
            />
          ) : null}
        </div>
      </Popover>
    </>
  );
}

/**
 * The concrete chat model exposes the Yjs shared model, whose `awareness` is
 * the channel personas broadcast their session state over. The generic
 * `IChatModel` type does not yet surface this, so we read it structurally.
 */
function getAwareness(chatModel: unknown): IAwareness | null {
  const shared = (chatModel as { sharedModel?: { awareness?: IAwareness } })
    ?.sharedModel;
  return shared?.awareness ?? null;
}

/**
 * The persona control for the chat input toolbar. Shows which persona a message
 * will be directed to (with its avatar), lets the user switch it, and, when the
 * selected persona advertises model/settings, renders those controls next to it.
 * Hides itself when the chat has no personas.
 *
 * All session information (the persona list, each persona's model/settings
 * configuration, usage, and slash commands) is read from the chat's Yjs
 * awareness channel. The selection is owned by the frontend and stamped onto
 * each message's metadata (there is no server-side "active persona" and no REST
 * polling). It's seeded from the default persona advertised over PageConfig.
 */
export function AcpPersonaControls(
  props: InputToolbarRegistry.IToolbarItemProps
): JSX.Element | null {
  const { chatModel, model } = props;
  const awareness = getAwareness(chatModel);

  // The manager's awareness view, resolved once its slot appears. Null until
  // then. `PersonaManagerAwareness.from()` polls internally, so nothing here
  // polls; once resolved, awareness `change` events drive all updates.
  const [manager, setManager] = useState<PersonaManagerAwareness | null>(null);
  const [personas, setPersonas] = useState<PersonaOption[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(
    DEFAULT_PERSONA_ID
  );
  const [personaState, setPersonaState] = useState<PersonaAwareness | null>(
    null
  );
  // Per-persona settings the user has chosen, indexed by persona ID. Remembers
  // each persona's picks so switching away and back restores them, rather than
  // resetting to defaults. Component-lifetime only (not persisted across
  // reloads). A persona absent from the cache has made no changes yet.
  const [settingsCache, setSettingsCache] = useState<
    Record<string, PersonaSettings>
  >({});
  const [personaAnchor, setPersonaAnchor] = useState<HTMLElement | null>(null);

  // The selected persona's settings: its cache entry, or empty (all defaults)
  // when it has none yet.
  const settings = selectedId
    ? (settingsCache[selectedId] ?? emptyPersonaSettings())
    : emptyPersonaSettings();

  // Resolve the manager's awareness view once the manager registers its slot.
  useEffect(() => {
    if (!awareness) {
      return;
    }
    let cancelled = false;
    PersonaManagerAwareness.from(awareness)
      .then(pm => {
        if (!cancelled) {
          setManager(pm);
        }
      })
      .catch(() => {
        // Manager never registered (e.g. extension disabled); the toolbar stays
        // hidden. `.from()` already logged the timeout reason.
      });
    return () => {
      cancelled = true;
    };
  }, [awareness]);

  // Re-read the persona list from the manager view and reconcile the selection:
  // if the current selection isn't present in this chat, fall back to the sole
  // persona (if there's exactly one) or to no one. This is the reactive
  // plumbing that replaces polling: a persona publishing or updating its state
  // fires an awareness `change` event.
  const readManager = useCallback(() => {
    if (!manager) {
      return;
    }
    const list = manager.personas;
    setPersonas(list);
    setSelectedId(current => {
      if (current && list.some(p => p.id === current)) {
        return current;
      }
      return list.length === 1 ? list[0].id : null;
    });
  }, [manager]);

  useEffect(() => {
    if (!awareness || !manager) {
      return;
    }
    readManager();
    const onChange = () => readManager();
    awareness.on('change', onChange);
    return () => {
      awareness.off('change', onChange);
    };
  }, [awareness, manager, readManager]);

  // Build a view of the selected persona's slot from the manager's list, or
  // null when nothing is selected / the persona isn't present yet.
  const readSelectedPersona = (): PersonaAwareness | null => {
    if (!awareness || !manager || !selectedId) {
      return null;
    }
    const option = manager.personas.find(p => p.id === selectedId);
    return option ? PersonaAwareness.from(awareness, option) : null;
  };

  // Track the selected persona's view in state, re-reading on every awareness
  // change (a persona updating usage, model, or commands) so the toolbar
  // reflects the latest published state.
  useEffect(() => {
    if (!awareness || !manager || !selectedId) {
      setPersonaState(null);
      return;
    }
    const read = () => setPersonaState(readSelectedPersona());
    read();
    awareness.on('change', read);
    return () => {
      awareness.off('change', read);
    };
  }, [awareness, manager, selectedId]);

  // Stamp the current persona + its settings onto the input model's metadata,
  // so it rides out with the next message and the PersonaManager routes and
  // applies it. Keyed on a signature so we only write when it changes.
  const metadataSignature = JSON.stringify({ selectedId, settings });
  useEffect(() => {
    model.clearMetadata();
    model.updateMetadata(buildMessageMetadata(selectedId, settings));
  }, [model, metadataSignature]);

  // No personas in the chat: nothing to show.
  if (!personas.length) {
    return null;
  }

  const selectedPersona = personas.find(p => p.id === selectedId) ?? null;
  const personaLabel = selectedPersona?.name ?? NO_ONE_LABEL;
  const activeAvatar = selectedPersona?.avatar_url ?? null;
  const usage = personaState?.usage ?? EMPTY_USAGE;
  const controls = buildControls(personaState, settings);

  const handlePersona = (personaId: string | null) => {
    setPersonaAnchor(null);
    setSelectedId(personaId);
  };

  const handleControl = (control: Control, value: string | null) => {
    if (!selectedId) {
      return;
    }
    // Fold the change into this persona's cached settings, remembering it for
    // when the user switches away and back.
    setSettingsCache(prev => ({
      ...prev,
      [selectedId]: applyControlChange(
        prev[selectedId] ?? emptyPersonaSettings(),
        control,
        value
      )
    }));
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
        onClick={event => setPersonaAnchor(event.currentTarget)}
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
