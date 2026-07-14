/**
 * Tests that the toolbar's controls are built from a persona's awareness state
 * (model, model settings, general settings) and reflect the user's current
 * per-message selection.
 */

import { buildControls } from '../persona-controls';
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
    isWriting: false,
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

describe('buildControls', () => {
  it('returns nothing when there is no persona state', () => {
    expect(buildControls(null, emptySelection('kiro'))).toEqual([]);
  });

  it('builds a model control, its model settings, then general settings, in order', () => {
    const controls = buildControls(withControls, emptySelection('kiro'));
    expect(controls.map(p => [p.id, p.kind])).toEqual([
      ['__model__', 'model'],
      ['context_size', 'model_setting'],
      ['__mode__', 'setting']
    ]);
  });

  it('omits the model control when the persona advertises no models', () => {
    const controls = buildControls(
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
    expect(controls.map(p => p.id)).toEqual(['__mode__']);
  });

  it('carries the persona current value from awareness onto each control', () => {
    const controls = buildControls(withControls, emptySelection('kiro'));
    const model = controls.find(p => p.id === '__model__')!;
    expect(model.current).toBe('opus-48');
    const mode = controls.find(p => p.id === '__mode__')!;
    expect(mode.current).toBe('ask');
  });

  it('reflects the user selection on each control', () => {
    const selection: PersonaSelection = {
      personaId: 'kiro',
      modelId: 'fable-5',
      modelSettings: { context_size: null },
      settings: { __mode__: 'code' }
    };
    const controls = buildControls(withControls, selection);
    expect(controls.find(p => p.id === '__model__')!.selection).toBe('fable-5');
    // Left at default → null selection (renders as the persona's current value).
    expect(controls.find(p => p.id === 'context_size')!.selection).toBeNull();
    expect(controls.find(p => p.id === '__mode__')!.selection).toBe('code');
  });

  it('maps each model option into the control choices', () => {
    const controls = buildControls(withControls, emptySelection('kiro'));
    const model = controls.find(p => p.id === '__model__')!;
    expect(model.options).toEqual([
      { id: 'opus-48', name: 'Opus 4.8', description: null },
      { id: 'fable-5', name: 'Fable 5', description: null }
    ]);
  });
});
