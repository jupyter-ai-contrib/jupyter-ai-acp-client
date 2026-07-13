import { buildSelectionMetadata } from '../persona-controls';
import { AcpControl } from '../request';

function control(partial: Partial<AcpControl> & { id: string }): AcpControl {
  return {
    source: 'config_option',
    kind: 'select',
    label: partial.id,
    current_value: null,
    choices: [],
    ...partial
  };
}

describe('buildSelectionMetadata', () => {
  it('stamps the active persona id', () => {
    expect(buildSelectionMetadata('kiro', [])).toEqual({ to_persona: 'kiro' });
  });

  it('carries a null persona (no one) verbatim', () => {
    expect(buildSelectionMetadata(null, [])).toEqual({ to_persona: null });
  });

  it('omits model/settings when no persona is selected', () => {
    const controls = [
      control({ id: '__model__', source: 'model', current_value: 'opus-48' }),
      control({ id: '__mode__', source: 'mode', current_value: 'auto' })
    ];
    // Even with controls loaded, "No one" carries no model/settings.
    expect(buildSelectionMetadata(null, controls)).toEqual({
      to_persona: null
    });
  });

  it('picks up the model from the model control', () => {
    const controls = [
      control({ id: '__model__', source: 'model', current_value: 'opus-48' })
    ];
    expect(buildSelectionMetadata('kiro', controls)).toEqual({
      to_persona: 'kiro',
      model: 'opus-48'
    });
  });

  it('collects non-model controls into settings, keyed by control id', () => {
    const controls = [
      control({ id: '__model__', source: 'model', current_value: 'opus-48' }),
      control({ id: '__mode__', source: 'mode', current_value: 'auto' }),
      control({
        id: 'reasoning',
        source: 'config_option',
        kind: 'boolean',
        current_value: true
      })
    ];
    expect(buildSelectionMetadata('kiro', controls)).toEqual({
      to_persona: 'kiro',
      model: 'opus-48',
      settings: { __mode__: 'auto', reasoning: true }
    });
  });

  it('omits settings when there are no non-model controls', () => {
    const controls = [
      control({ id: '__model__', source: 'model', current_value: 'opus-48' })
    ];
    const metadata = buildSelectionMetadata('kiro', controls);
    expect(metadata).not.toHaveProperty('settings');
  });

  it('omits the model when its value is not set', () => {
    const controls = [
      control({ id: '__model__', source: 'model', current_value: null })
    ];
    const metadata = buildSelectionMetadata('kiro', controls);
    expect(metadata).not.toHaveProperty('model');
  });

  it('skips controls whose value is null', () => {
    const controls = [
      control({ id: '__mode__', source: 'mode', current_value: null }),
      control({ id: 'verbose', kind: 'boolean', current_value: false })
    ];
    expect(buildSelectionMetadata('kiro', controls)).toEqual({
      to_persona: 'kiro',
      settings: { verbose: false }
    });
  });
});
