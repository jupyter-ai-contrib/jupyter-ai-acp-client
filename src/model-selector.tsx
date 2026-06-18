import React, { useCallback, useEffect, useState } from 'react';
import { Button, ListItemText, Menu, MenuItem } from '@mui/material';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import CheckIcon from '@mui/icons-material/Check';
import { InputToolbarRegistry } from '@jupyter/chat';
import { AcpModel, getAcpModels, setAcpModel } from './request';

const SELECTOR_CLASS = 'jp-jupyter-ai-acp-client-modelSelector';
const MENU_CLASS = 'jp-jupyter-ai-acp-client-modelMenu';

/**
 * Returns the single persona mention name in the draft, or `null` when the
 * draft mentions zero or more than one persona. When `null`, the backend
 * resolves the addressed persona as the last-mentioned or default one.
 */
function draftMention(value: string | undefined): string | null {
  const mentions = new Set<string>();
  const matches = value?.matchAll(/@([\w-]*)/g);
  if (matches) {
    for (const match of matches) {
      if (match[1]) {
        mentions.add(match[1]);
      }
    }
  }
  return mentions.size === 1 ? (mentions.values().next().value ?? null) : null;
}

/**
 * A model selector for the chat input toolbar. Shows the addressed ACP
 * persona's current model and lets the user pick another. Hides itself when the
 * addressed persona is not an ACP persona or advertises no models.
 */
export function AcpModelSelector(
  props: InputToolbarRegistry.IToolbarItemProps
): JSX.Element | null {
  const { model, chatModel } = props;
  const [models, setModels] = useState<AcpModel[]>([]);
  const [currentModelId, setCurrentModelId] = useState<string | null>(null);
  const [persona, setPersona] = useState<string | null>(null);
  const [anchorEl, setAnchorEl] = useState<HTMLElement | null>(null);

  const chatPath = chatModel?.name ?? null;

  const refresh = useCallback(async () => {
    if (!chatPath) {
      return;
    }
    const response = await getAcpModels(chatPath, draftMention(model.value));
    setModels(response.models);
    setCurrentModelId(response.current_model_id);
    setPersona(response.persona);
  }, [chatPath, model]);

  useEffect(() => {
    refresh();
    chatModel?.messagesUpdated?.connect(refresh);
    return () => {
      chatModel?.messagesUpdated?.disconnect(refresh);
    };
  }, [chatModel, refresh]);

  // Nothing to choose: hide the control entirely.
  if (!models.length) {
    return null;
  }

  const currentName =
    models.find(m => m.model_id === currentModelId)?.name ??
    currentModelId ??
    'model';

  const handleSelect = async (modelId: string) => {
    setAnchorEl(null);
    setCurrentModelId(modelId);
    if (chatPath) {
      await setAcpModel(chatPath, draftMention(model.value), modelId);
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
          setAnchorEl(event.currentTarget);
          refresh();
        }}
        title="Select the model for this persona"
      >
        {persona ? (
          <>
            <span className={`${SELECTOR_CLASS}-persona`}>{persona}</span>
            <span className={`${SELECTOR_CLASS}-sep`}>·</span>
          </>
        ) : null}
        <span className={`${SELECTOR_CLASS}-model`}>{currentName}</span>
      </Button>
      <Menu
        anchorEl={anchorEl}
        open={!!anchorEl}
        onClose={() => setAnchorEl(null)}
        anchorOrigin={{ vertical: 'top', horizontal: 'left' }}
        transformOrigin={{ vertical: 'bottom', horizontal: 'left' }}
        PaperProps={{ className: `${MENU_CLASS}-paper` }}
      >
        {models.map(m => {
          const isCurrent = m.model_id === currentModelId;
          return (
            <MenuItem
              key={m.model_id}
              selected={isCurrent}
              onClick={() => handleSelect(m.model_id)}
            >
              <ListItemText
                primary={m.name}
                secondary={m.description ?? null}
                classes={{
                  primary: `${MENU_CLASS}-name`,
                  secondary: `${MENU_CLASS}-desc`
                }}
              />
              {isCurrent ? (
                <CheckIcon className={`${MENU_CLASS}-check`} fontSize="small" />
              ) : null}
            </MenuItem>
          );
        })}
      </Menu>
    </>
  );
}
