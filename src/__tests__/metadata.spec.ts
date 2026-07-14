/**
 * Tests that the user's persona/model/settings selection is written to message
 * metadata in the shape the persona-manager expects — the single mechanism by
 * which selections reach the server (no REST calls). Covers building the
 * metadata, seeding a selection from a persona's awareness state, and folding
 * picker changes back into the selection.
 */

import {
  buildSelectionMetadata,
  emptySelection,
  PersonaSelection,
  selectionForPersona
} from '../metadata';
import { PersonaAwarenessState } from '../awareness';
import { applyPickerChange, Picker } from '../persona-controls';

function state(
  partial: Partial<PersonaAwarenessState> = {}
): PersonaAwarenessState {
  return {
    id: 'kiro',
    model: { current: null, options: [], settings: [] },
    settings: [],
    usage: {
      context_tokens: null,
      context_size: null,
      input_tokens: null,
      output_tokens: null,
      cached_read_tokens: null,
      cached_write_tokens: null,
      thought_tokens: null,
      total_tokens: null,
      cost_amount: null,
      cost_currency: null
    },
    slash_commands: [],
    ...partial
  };
}

function picker(
  partial: Partial<Picker> & { id: string; kind: Picker['kind'] }
): Picker {
  return {
    label: partial.id,
    current: null,
    selection: null,
    options: [],
    ...partial
  };
}

describe('buildSelectionMetadata', () => {
  it('stamps the target persona id', () => {
    const metadata = buildSelectionMetadata(emptySelection('kiro'));
    expect(metadata.to_persona).toBe('kiro');
  });

  it('carries a null persona (no one) with nothing to configure', () => {
    expect(buildSelectionMetadata(emptySelection(null))).toEqual({
      to_persona: null
    });
  });

  it('always includes a model spec and settings for a real persona', () => {
    // Even at all-defaults, the message fully describes its selection: a model
    // spec with a null id and empty settings, and empty general settings.
    expect(buildSelectionMetadata(emptySelection('kiro'))).toEqual({
      to_persona: 'kiro',
      model: { id: null, settings: {} },
      settings: {}
    });
  });

  it('writes the chosen model id and model settings into the model spec', () => {
    const selection: PersonaSelection = {
      personaId: 'kiro',
      modelId: 'opus-48',
      modelSettings: { context_size: '200k' },
      settings: {}
    };
    expect(buildSelectionMetadata(selection)).toEqual({
      to_persona: 'kiro',
      model: { id: 'opus-48', settings: { context_size: '200k' } },
      settings: {}
    });
  });

  it('writes general settings (mode and config options) keyed by id', () => {
    const selection: PersonaSelection = {
      personaId: 'kiro',
      modelId: null,
      modelSettings: {},
      settings: { __mode__: 'code', reasoning: 'true' }
    };
    expect(buildSelectionMetadata(selection)).toEqual({
      to_persona: 'kiro',
      model: { id: null, settings: {} },
      settings: { __mode__: 'code', reasoning: 'true' }
    });
  });

  it('keeps null values (a null means "use the persona default")', () => {
    const selection: PersonaSelection = {
      personaId: 'kiro',
      modelId: null,
      modelSettings: { context_size: null },
      settings: { __mode__: null }
    };
    const metadata = buildSelectionMetadata(selection);
    expect(metadata.model).toEqual({
      id: null,
      settings: { context_size: null }
    });
    expect(metadata.settings).toEqual({ __mode__: null });
  });
});

describe('selectionForPersona', () => {
  it('is empty for no persona', () => {
    expect(selectionForPersona(null, null)).toEqual({
      personaId: null,
      modelId: null,
      modelSettings: {},
      settings: {}
    });
  });

  it('seeds every control at the default (null), keyed from awareness', () => {
    const s = selectionForPersona(
      'kiro',
      state({
        model: {
          current: 'opus-48',
          options: [{ id: 'opus-48', name: 'Opus', description: null }],
          settings: [
            {
              id: 'context_size',
              current: '200k',
              name: 'Context',
              description: null,
              options: []
            }
          ]
        },
        settings: [
          {
            id: '__mode__',
            current: 'ask',
            name: 'Mode',
            description: null,
            options: []
          }
        ]
      })
    );
    // All controls present, all at default (null) — the user hasn't diverged yet.
    expect(s).toEqual({
      personaId: 'kiro',
      modelId: null,
      modelSettings: { context_size: null },
      settings: { __mode__: null }
    });
  });
});

describe('applyPickerChange', () => {
  const base: PersonaSelection = {
    personaId: 'kiro',
    modelId: null,
    modelSettings: { context_size: null },
    settings: { __mode__: null }
  };

  it('routes a model picker change to modelId', () => {
    const next = applyPickerChange(
      base,
      picker({ id: '__model__', kind: 'model' }),
      'opus-48'
    );
    expect(next.modelId).toBe('opus-48');
    expect(next.modelSettings).toEqual({ context_size: null });
  });

  it('routes a model-setting change to modelSettings by id', () => {
    const next = applyPickerChange(
      base,
      picker({ id: 'context_size', kind: 'model_setting' }),
      '200k'
    );
    expect(next.modelSettings).toEqual({ context_size: '200k' });
  });

  it('routes a general-setting change to settings by id', () => {
    const next = applyPickerChange(
      base,
      picker({ id: '__mode__', kind: 'setting' }),
      'code'
    );
    expect(next.settings).toEqual({ __mode__: 'code' });
  });

  it('resets a control to default when the value is null', () => {
    const chosen = applyPickerChange(
      base,
      picker({ id: '__mode__', kind: 'setting' }),
      'code'
    );
    const reset = applyPickerChange(
      chosen,
      picker({ id: '__mode__', kind: 'setting' }),
      null
    );
    expect(reset.settings).toEqual({ __mode__: null });
  });

  it('does not mutate the input selection', () => {
    applyPickerChange(base, picker({ id: '__model__', kind: 'model' }), 'x');
    expect(base.modelId).toBeNull();
  });
});
