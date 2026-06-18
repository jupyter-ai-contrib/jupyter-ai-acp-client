import React, { useCallback, useEffect, useState } from 'react';
import { Button, ListItemText, Menu, MenuItem } from '@mui/material';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import CheckIcon from '@mui/icons-material/Check';
import { InputToolbarRegistry } from '@jupyter/chat';
import {
  AcpModel,
  ActivePersonaInfo,
  getActivePersona,
  setAcpModel,
  setActivePersona
} from './request';

const SELECTOR_CLASS = 'jp-jupyter-ai-acp-client-modelSelector';
const MENU_CLASS = 'jp-jupyter-ai-acp-client-modelMenu';
const NO_ONE_LABEL = 'No one';

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
 * The active-persona control for the chat input toolbar. Shows which persona
 * replies (with its avatar), lets the user switch it, and, when the active
 * persona is an ACP persona, shows its model selector next to it. Hides itself
 * when the chat has no personas.
 */
export function AcpPersonaControls(
  props: InputToolbarRegistry.IToolbarItemProps
): JSX.Element | null {
  const { chatModel } = props;
  const [personas, setPersonas] = useState<ActivePersonaInfo[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [activeName, setActiveName] = useState<string | null>(null);
  const [models, setModels] = useState<AcpModel[]>([]);
  const [currentModelId, setCurrentModelId] = useState<string | null>(null);
  const [personaAnchor, setPersonaAnchor] = useState<HTMLElement | null>(null);
  const [modelAnchor, setModelAnchor] = useState<HTMLElement | null>(null);
  const [modelRetries, setModelRetries] = useState(0);

  const chatPath = chatModel?.name ?? null;

  const refresh = useCallback(async () => {
    if (!chatPath) {
      return;
    }
    const response = await getActivePersona(chatPath);
    setPersonas(response.personas);
    setActiveId(response.active_id);
    setActiveName(response.active_name);
    setModels(response.models);
    setCurrentModelId(response.current_model_id);
  }, [chatPath]);

  useEffect(() => {
    refresh();
    chatModel?.messagesUpdated?.connect(refresh);
    return () => {
      chatModel?.messagesUpdated?.disconnect(refresh);
    };
  }, [chatModel, refresh]);

  // Reset the model retry counter whenever the active persona changes.
  useEffect(() => {
    setModelRetries(0);
  }, [activeId]);

  // An ACP persona's models load asynchronously while its agent session
  // initializes. If the active persona is an ACP persona but no models have
  // arrived yet, retry a few times until they do. Depend on primitive flags,
  // not the personas/models array references (which change on every refresh),
  // so unrelated refreshes do not restart the retry timer.
  const activeIsAcp = personas.find(p => p.id === activeId)?.is_acp ?? false;
  const hasModels = models.length > 0;
  useEffect(() => {
    if (!activeIsAcp || hasModels || modelRetries >= 6) {
      return;
    }
    const timer = window.setTimeout(() => {
      setModelRetries(r => r + 1);
      refresh();
    }, 1500);
    return () => window.clearTimeout(timer);
  }, [activeIsAcp, hasModels, modelRetries, refresh]);

  // No personas in the chat: nothing to show.
  if (!personas.length) {
    return null;
  }

  const activeAvatar = personas.find(p => p.id === activeId)?.avatar_url ?? null;
  const personaLabel = activeName ?? NO_ONE_LABEL;
  const currentModelName =
    models.find(m => m.model_id === currentModelId)?.name ??
    currentModelId ??
    'model';

  const handlePersona = async (personaId: string | null) => {
    setPersonaAnchor(null);
    setActiveId(personaId);
    if (chatPath) {
      await setActivePersona(chatPath, personaId);
      // Refetch so the model control follows the new active persona.
      await refresh();
    }
  };

  const handleModel = async (modelId: string) => {
    setModelAnchor(null);
    setCurrentModelId(modelId);
    if (chatPath) {
      await setAcpModel(chatPath, null, modelId);
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

      {models.length ? (
        <>
          <span className={`${SELECTOR_CLASS}-divider`} />
          <Button
            className={`${SELECTOR_CLASS} ${SELECTOR_CLASS}-model-btn`}
            size="small"
            variant="text"
            disableRipple
            endIcon={<ArrowDropDownIcon className={`${SELECTOR_CLASS}-arrow`} />}
            onClick={event => {
              setModelAnchor(event.currentTarget);
              refresh();
            }}
            title="Select the model for this persona"
          >
            <span className={`${SELECTOR_CLASS}-model`}>{currentModelName}</span>
          </Button>
          <Menu
            anchorEl={modelAnchor}
            open={!!modelAnchor}
            onClose={() => setModelAnchor(null)}
            anchorOrigin={{ vertical: 'top', horizontal: 'left' }}
            transformOrigin={{ vertical: 'bottom', horizontal: 'left' }}
            PaperProps={{ className: `${MENU_CLASS}-paper` }}
          >
            {models.map(m => (
              <MenuItem
                key={m.model_id}
                selected={m.model_id === currentModelId}
                onClick={() => handleModel(m.model_id)}
              >
                <ListItemText
                  primary={m.name}
                  secondary={m.description ?? null}
                  classes={{
                    primary: `${MENU_CLASS}-name`,
                    secondary: `${MENU_CLASS}-desc`
                  }}
                />
                {m.model_id === currentModelId ? (
                  <CheckIcon
                    className={`${MENU_CLASS}-check`}
                    fontSize="small"
                  />
                ) : null}
              </MenuItem>
            ))}
          </Menu>
        </>
      ) : null}
    </div>
  );
}
