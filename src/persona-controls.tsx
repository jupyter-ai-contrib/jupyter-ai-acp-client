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
  Switch
} from '@mui/material';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import CheckIcon from '@mui/icons-material/Check';
import MoreHorizIcon from '@mui/icons-material/MoreHoriz';
import { InputToolbarRegistry } from '@jupyter/chat';
import {
  AcpControl,
  AcpControlChoice,
  ActivePersonaInfo,
  getActivePersona,
  setAcpControl,
  setActivePersona
} from './request';

const SELECTOR_CLASS = 'jp-jupyter-ai-acp-client-personaControls';
const MENU_CLASS = 'jp-jupyter-ai-acp-client-controlMenu';
const NO_ONE_LABEL = 'No one';

// Personas register a moment after a chat opens, and an ACP persona's controls
// load asynchronously while its agent session initializes, which can take 20s+
// for a slow agent (e.g. Kiro recovering a session). Poll a bounded number of
// times so the controls appear without needing a first message. The budget
// resets when the active persona changes, so each persona gets a full window.
const POLL_MS = 1500;
const MAX_POLLS = 24;

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

/**
 * The active-persona control for the chat input toolbar. Shows which persona
 * replies (with its avatar), lets the user switch it, and, when the active
 * persona is an ACP persona, renders its session controls (model, mode, and any
 * config options) next to it. Hides itself when the chat has no personas.
 */
export function AcpPersonaControls(
  props: InputToolbarRegistry.IToolbarItemProps
): JSX.Element | null {
  const { chatModel } = props;
  const [personas, setPersonas] = useState<ActivePersonaInfo[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [activeName, setActiveName] = useState<string | null>(null);
  const [controls, setControls] = useState<AcpControl[]>([]);
  const [personaAnchor, setPersonaAnchor] = useState<HTMLElement | null>(null);
  const [polls, setPolls] = useState(0);

  const chatPath = chatModel?.name ?? null;

  const refresh = useCallback(async () => {
    if (!chatPath) {
      return;
    }
    const response = await getActivePersona(chatPath);
    setPersonas(response.personas);
    setActiveId(response.active_id);
    setActiveName(response.active_name);
    setControls(response.controls);
  }, [chatPath]);

  useEffect(() => {
    refresh();
    chatModel?.messagesUpdated?.connect(refresh);
    return () => {
      chatModel?.messagesUpdated?.disconnect(refresh);
    };
  }, [chatModel, refresh]);

  // Reset the poll counter when the chat or the active persona changes.
  useEffect(() => {
    setPolls(0);
  }, [chatPath, activeId]);

  // Poll until the personas register and the active ACP persona's controls
  // load, so nothing waits on a first message. Depend on primitive flags, not
  // the array references (which change on every refresh), to avoid restarting
  // the timer on unrelated updates.
  const activeIsAcp = personas.find(p => p.id === activeId)?.is_acp ?? false;
  const needPersonas = personas.length === 0;
  const needControls = activeIsAcp && controls.length === 0;
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

  // No personas in the chat: nothing to show.
  if (!personas.length) {
    return null;
  }

  const activeAvatar =
    personas.find(p => p.id === activeId)?.avatar_url ?? null;
  const personaLabel = activeName ?? NO_ONE_LABEL;

  const handlePersona = async (personaId: string | null) => {
    setPersonaAnchor(null);
    setActiveId(personaId);
    if (chatPath) {
      await setActivePersona(chatPath, personaId);
      // Refetch so the controls follow the new active persona.
      await refresh();
    }
  };

  const handleControl = async (
    control: AcpControl,
    value: string | boolean
  ) => {
    // Optimistic update for immediate feedback.
    setControls(prev =>
      prev.map(c => (c.id === control.id ? { ...c, current_value: value } : c))
    );
    if (chatPath) {
      await setAcpControl(chatPath, control.id, control.source, value);
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
        title="Choose which persona replies"
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
            selected={p.id === activeId}
            onClick={() => handlePersona(p.id)}
          >
            <Avatar url={p.avatar_url} />
            <ListItemText
              primary={p.name}
              classes={{ primary: `${MENU_CLASS}-name` }}
            />
            {p.id === activeId ? (
              <CheckIcon className={`${MENU_CLASS}-check`} fontSize="small" />
            ) : null}
          </MenuItem>
        ))}
        <MenuItem
          selected={activeId === null}
          onClick={() => handlePersona(null)}
        >
          <Avatar url={null} />
          <ListItemText
            primary={NO_ONE_LABEL}
            classes={{ primary: `${MENU_CLASS}-name` }}
          />
          {activeId === null ? (
            <CheckIcon className={`${MENU_CLASS}-check`} fontSize="small" />
          ) : null}
        </MenuItem>
      </Menu>

      {controls.length ? (
        <>
          <span className={`${SELECTOR_CLASS}-divider`} />
          <ControlsRow controls={controls} onChange={handleControl} />
        </>
      ) : null}
    </div>
  );
}
