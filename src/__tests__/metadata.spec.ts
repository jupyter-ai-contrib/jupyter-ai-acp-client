/**
 * Tests that the user's persona + settings selection is written to message
 * metadata in the shape the persona-manager expects — the single mechanism by
 * which selections reach the server (no REST calls) — and that folding control
 * changes into a persona's settings behaves.
 */

import {
  buildMessageMetadata,
  emptyPersonaSettings,
  PersonaSettings
} from '../metadata';
import { applyControlChange, Control } from '../persona-controls';

function control(
  partial: Partial<Control> & { id: string; kind: Control['kind'] }
): Control {
  return {
    label: partial.id,
    current: null,
    selection: null,
    options: [],
    ...partial
  };
}

describe('buildMessageMetadata', () => {
  it('stamps the target persona id', () => {
    const metadata = buildMessageMetadata('kiro', emptyPersonaSettings());
    expect(metadata.to_persona).toBe('kiro');
  });

  it('carries a null persona (no one) with nothing to configure', () => {
    expect(buildMessageMetadata(null, emptyPersonaSettings())).toEqual({
      to_persona: null
    });
  });

  it('always includes a model spec and settings for a real persona', () => {
    // Even at all-defaults, the message fully describes its selection: a model
    // spec with a null id and empty settings, and empty general settings.
    expect(buildMessageMetadata('kiro', emptyPersonaSettings())).toEqual({
      to_persona: 'kiro',
      model: { id: null, settings: {} },
      settings: {}
    });
  });

  it('writes the chosen model id and model settings into the model spec', () => {
    const settings: PersonaSettings = {
      modelId: 'opus-48',
      modelSettings: { context_size: '200k' },
      settings: {}
    };
    expect(buildMessageMetadata('kiro', settings)).toEqual({
      to_persona: 'kiro',
      model: { id: 'opus-48', settings: { context_size: '200k' } },
      settings: {}
    });
  });

  it('writes general settings (mode and config options) keyed by id', () => {
    const settings: PersonaSettings = {
      modelId: null,
      modelSettings: {},
      settings: { __mode__: 'code', reasoning: 'true' }
    };
    expect(buildMessageMetadata('kiro', settings)).toEqual({
      to_persona: 'kiro',
      model: { id: null, settings: {} },
      settings: { __mode__: 'code', reasoning: 'true' }
    });
  });

  it('keeps null values (a null means "use the persona default")', () => {
    const settings: PersonaSettings = {
      modelId: null,
      modelSettings: { context_size: null },
      settings: { __mode__: null }
    };
    const metadata = buildMessageMetadata('kiro', settings);
    expect(metadata.model).toEqual({
      id: null,
      settings: { context_size: null }
    });
    expect(metadata.settings).toEqual({ __mode__: null });
  });
});

describe('emptyPersonaSettings', () => {
  it('changes nothing (all defaults)', () => {
    expect(emptyPersonaSettings()).toEqual({
      modelId: null,
      modelSettings: {},
      settings: {}
    });
  });
});

describe('applyControlChange', () => {
  const base: PersonaSettings = {
    modelId: null,
    modelSettings: { context_size: null },
    settings: { __mode__: null }
  };

  it('routes a model control change to modelId', () => {
    const next = applyControlChange(
      base,
      control({ id: '__model__', kind: 'model' }),
      'opus-48'
    );
    expect(next.modelId).toBe('opus-48');
    expect(next.modelSettings).toEqual({ context_size: null });
  });

  it('routes a model-setting change to modelSettings by id', () => {
    const next = applyControlChange(
      base,
      control({ id: 'context_size', kind: 'model_setting' }),
      '200k'
    );
    expect(next.modelSettings).toEqual({ context_size: '200k' });
  });

  it('routes a general-setting change to settings by id', () => {
    const next = applyControlChange(
      base,
      control({ id: '__mode__', kind: 'setting' }),
      'code'
    );
    expect(next.settings).toEqual({ __mode__: 'code' });
  });

  it('resets a control to default when the value is null', () => {
    const chosen = applyControlChange(
      base,
      control({ id: '__mode__', kind: 'setting' }),
      'code'
    );
    const reset = applyControlChange(
      chosen,
      control({ id: '__mode__', kind: 'setting' }),
      null
    );
    expect(reset.settings).toEqual({ __mode__: null });
  });

  it('does not mutate the input settings', () => {
    applyControlChange(base, control({ id: '__model__', kind: 'model' }), 'x');
    expect(base.modelId).toBeNull();
  });
});
