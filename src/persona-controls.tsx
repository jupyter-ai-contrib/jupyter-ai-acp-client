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
 * The active-persona control for the chat input toolbar. Shows which persona
 * replies (or "No one"), lets the user switch it, and, when the active persona
 * is an ACP persona, shows its model selector next to it. Hides itself when the
 * chat has no personas.
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

  // No personas in the chat: nothing to show.
  if (!personas.length) {
    return null;
  }

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
    <>
      <Button
        className={SELECTOR_CLASS}
        size="small"
        variant="text"
        disableRipple
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
          <Button
            className={SELECTOR_CLASS}
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
    </>
  );
}
