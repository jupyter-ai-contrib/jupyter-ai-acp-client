/**
 * Tests that the toolbar's pickers are built from a persona's awareness state
 * (model, model settings, general settings) and reflect the user's current
 * per-message selection.
 */

import { buildPickers } from '../persona-controls';
import { PersonaAwarenessState } from '../awareness';
import { emptySelection, PersonaSelection } from '../metadata';

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

const withControls = state({
  model: {
    current: 'opus-48',
    options: [
      { id: 'opus-48', name: 'Opus 4.8', description: null },
      { id: 'fable-5', name: 'Fable 5', description: null }
    ],
    settings: [
      {
        id: 'context_size',
        current: '200k',
        name: 'Context size',
        description: null,
        options: [{ id: '200k', name: '200K', description: null }]
      }
    ]
  },
  settings: [
    {
      id: '__mode__',
      current: 'ask',
      name: 'Mode',
      description: null,
      options: [
        { id: 'ask', name: 'Ask', description: null },
        { id: 'code', name: 'Code', description: null }
      ]
    }
  ]
});

describe('buildPickers', () => {
  it('returns nothing when there is no persona state', () => {
    expect(buildPickers(null, emptySelection('kiro'))).toEqual([]);
  });

  it('builds a model picker, its model settings, then general settings, in order', () => {
    const pickers = buildPickers(withControls, emptySelection('kiro'));
    expect(pickers.map(p => [p.id, p.kind])).toEqual([
      ['__model__', 'model'],
      ['context_size', 'model_setting'],
      ['__mode__', 'setting']
    ]);
  });

  it('omits the model picker when the persona advertises no models', () => {
    const pickers = buildPickers(
      state({
        settings: [
          {
            id: '__mode__',
            current: 'ask',
            name: 'Mode',
            description: null,
            options: []
          }
        ]
      }),
      emptySelection('kiro')
    );
    expect(pickers.map(p => p.id)).toEqual(['__mode__']);
  });

  it('carries the persona current value from awareness onto each picker', () => {
    const pickers = buildPickers(withControls, emptySelection('kiro'));
    const model = pickers.find(p => p.id === '__model__')!;
    expect(model.current).toBe('opus-48');
    const mode = pickers.find(p => p.id === '__mode__')!;
    expect(mode.current).toBe('ask');
  });

  it('reflects the user selection on each picker', () => {
    const selection: PersonaSelection = {
      personaId: 'kiro',
      modelId: 'fable-5',
      modelSettings: { context_size: null },
      settings: { __mode__: 'code' }
    };
    const pickers = buildPickers(withControls, selection);
    expect(pickers.find(p => p.id === '__model__')!.selection).toBe('fable-5');
    // Left at default → null selection (renders as the persona's current value).
    expect(pickers.find(p => p.id === 'context_size')!.selection).toBeNull();
    expect(pickers.find(p => p.id === '__mode__')!.selection).toBe('code');
  });

  it('maps each model option into the picker choices', () => {
    const pickers = buildPickers(withControls, emptySelection('kiro'));
    const model = pickers.find(p => p.id === '__model__')!;
    expect(model.options).toEqual([
      { id: 'opus-48', name: 'Opus 4.8', description: null },
      { id: 'fable-5', name: 'Fable 5', description: null }
    ]);
  });
});
