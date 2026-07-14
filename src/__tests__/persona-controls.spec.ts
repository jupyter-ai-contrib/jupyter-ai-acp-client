/**
 * Tests that the toolbar's controls are built from a persona's awareness state
 * (model, model settings, general settings) and reflect the user's current
 * per-message selection.
 */

import { buildControls } from '../persona-controls';
import { emptyPersonaSettings, PersonaSettings } from '../metadata';
import { personaAwareness } from './awareness-fixture';

const withControls = personaAwareness({
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
    expect(buildControls(null, emptyPersonaSettings())).toEqual([]);
  });

  it('builds a model control, its model settings, then general settings, in order', () => {
    const controls = buildControls(withControls, emptyPersonaSettings());
    expect(controls.map(p => [p.id, p.kind])).toEqual([
      ['__model__', 'model'],
      ['context_size', 'model_setting'],
      ['__mode__', 'setting']
    ]);
  });

  it('omits the model control when the persona advertises no models', () => {
    const controls = buildControls(
      personaAwareness({
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
      emptyPersonaSettings()
    );
    expect(controls.map(p => p.id)).toEqual(['__mode__']);
  });

  it('carries the persona current value from awareness onto each control', () => {
    const controls = buildControls(withControls, emptyPersonaSettings());
    const model = controls.find(p => p.id === '__model__')!;
    expect(model.current).toBe('opus-48');
    const mode = controls.find(p => p.id === '__mode__')!;
    expect(mode.current).toBe('ask');
  });

  it('reflects the user selection on each control', () => {
    const settings: PersonaSettings = {
      modelId: 'fable-5',
      modelSettings: { context_size: null },
      settings: { __mode__: 'code' }
    };
    const controls = buildControls(withControls, settings);
    expect(controls.find(p => p.id === '__model__')!.selection).toBe('fable-5');
    // Left at default → null selection (renders as the persona's current value).
    expect(controls.find(p => p.id === 'context_size')!.selection).toBeNull();
    expect(controls.find(p => p.id === '__mode__')!.selection).toBe('code');
  });

  it('maps each model option into the control choices', () => {
    const controls = buildControls(withControls, emptyPersonaSettings());
    const model = controls.find(p => p.id === '__model__')!;
    expect(model.options).toEqual([
      { id: 'opus-48', name: 'Opus 4.8', description: null },
      { id: 'fable-5', name: 'Fable 5', description: null }
    ]);
  });
});
