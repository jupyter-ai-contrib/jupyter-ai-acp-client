import React, { useCallback, useEffect, useState } from 'react';
import { Button, ListItemText, Menu, MenuItem } from '@mui/material';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import CheckIcon from '@mui/icons-material/Check';
import { InputToolbarRegistry } from '@jupyter/chat';
import {
  AcpControl,
  ActivePersonaInfo,
  getActivePersona,
  setAcpControl,
  setActivePersona
} from './request';

const SELECTOR_CLASS = 'jp-jupyter-ai-acp-client-modelSelector';
const MENU_CLASS = 'jp-jupyter-ai-acp-client-modelMenu';
const NO_ONE_LABEL = 'No one';

// Personas register a moment after a chat opens, and an ACP persona's controls
// load asynchronously while its agent session initializes. Poll a bounded
// number of times so the controls appear without needing a first message.
const POLL_MS = 1500;
const MAX_POLLS = 10;

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
 * A dropdown for a select control (model, mode, or a select config option).
 */
function SelectControl(props: {
  control: AcpControl;
  onSelect: (value: string) => void;
}): JSX.Element {
  const { control, onSelect } = props;
  const [anchor, setAnchor] = useState<HTMLElement | null>(null);
  const currentLabel =
    control.choices.find(c => c.value === control.current_value)?.label ??
    (typeof control.current_value === 'string'
      ? control.current_value
      : null) ??
    control.label;
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
          {currentLabel}
        </span>
      </Button>
      <Menu
        anchorEl={anchor}
        open={!!anchor}
        onClose={() => setAnchor(null)}
        anchorOrigin={{ vertical: 'top', horizontal: 'left' }}
        transformOrigin={{ vertical: 'bottom', horizontal: 'left' }}
        PaperProps={{ className: `${MENU_CLASS}-paper` }}
      >
        {control.choices.map(choice => (
          <MenuItem
            key={choice.value}
            selected={choice.value === control.current_value}
            onClick={() => {
              setAnchor(null);
              onSelect(choice.value);
            }}
          >
            <ListItemText
              primary={choice.label}
              secondary={choice.description ?? null}
              classes={{
                primary: `${MENU_CLASS}-name`,
                secondary: `${MENU_CLASS}-desc`
              }}
            />
            {choice.value === control.current_value ? (
              <CheckIcon className={`${MENU_CLASS}-check`} fontSize="small" />
            ) : null}
          </MenuItem>
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
        anchorOrigin={{ vertical: 'top', horizontal: 'left' }}
        transformOrigin={{ vertical: 'bottom', horizontal: 'left' }}
        PaperProps={{ className: `${MENU_CLASS}-paper` }}
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
          {controls.map(control =>
            control.kind === 'boolean' ? (
              <BooleanControl
                key={control.id}
                control={control}
                onToggle={value => handleControl(control, value)}
              />
            ) : (
              <SelectControl
                key={control.id}
                control={control}
                onSelect={value => handleControl(control, value)}
              />
            )
          )}
        </>
      ) : null}
    </div>
  );
}
