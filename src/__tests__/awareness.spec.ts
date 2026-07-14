/**
 * Tests that all session information is read from the chat awareness channel:
 * the persona list from the manager's fixed slot, and each persona's model
 * configuration, settings, usage, and slash commands from its own slot. This is
 * the single source of truth the toolbar and slash-command provider read from.
 */

import { Awareness } from 'y-protocols/awareness';
import {
  findPersonaList,
  readPersonaList,
  readPersonaState,
  readPersonaStateById,
  PersonaAwarenessState
} from '../awareness';

const MANAGER_CLIENT_ID = 7133713371337;

/**
 * A minimal stand-in for a y-protocols Awareness that only implements the
 * `getStates()` map the readers use. Keyed by client ID, as the real one is.
 */
function fakeAwareness(
  states: Record<number, Record<string, unknown>>
): Awareness {
  const map = new Map<number, Record<string, unknown>>(
    Object.entries(states).map(([k, v]) => [Number(k), v])
  );
  return { getStates: () => map } as unknown as Awareness;
}

function personaState(
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

describe('readPersonaList', () => {
  it('reads the persona list from the manager slot', () => {
    const awareness = fakeAwareness({
      [MANAGER_CLIENT_ID]: {
        personas: [
          { id: 'kiro', name: 'Kiro', avatar_url: '/k', yjs_client_id: 42 }
        ]
      }
    });
    expect(readPersonaList(awareness, MANAGER_CLIENT_ID)).toEqual([
      { id: 'kiro', name: 'Kiro', avatar_url: '/k', yjs_client_id: 42 }
    ]);
  });

  it('is empty until the manager slot appears', () => {
    expect(readPersonaList(fakeAwareness({}), MANAGER_CLIENT_ID)).toEqual([]);
  });
});

describe('findPersonaList', () => {
  it('finds the manager slot by scanning, without a known client ID', () => {
    const awareness = fakeAwareness({
      42: { persona: personaState() },
      [MANAGER_CLIENT_ID]: {
        personas: [
          { id: 'kiro', name: 'Kiro', avatar_url: null, yjs_client_id: 42 }
        ]
      }
    });
    expect(findPersonaList(awareness).map(p => p.id)).toEqual(['kiro']);
  });
});

describe('readPersonaState', () => {
  it('reads a persona state from its own client-id slot', () => {
    const state = personaState({
      model: {
        current: 'opus-48',
        options: [{ id: 'opus-48', name: 'Opus', description: null }],
        settings: [
          {
            id: 'context_size',
            current: '200k',
            name: 'Context',
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
      ],
      usage: {
        ...personaState().usage,
        context_tokens: 1000,
        context_size: 200000,
        total_tokens: 4200
      },
      slash_commands: [{ name: '/compact', description: 'Compact context' }]
    });
    const awareness = fakeAwareness({ 42: { persona: state } });

    const read = readPersonaState(awareness, 42);
    expect(read?.model.current).toBe('opus-48');
    expect(read?.model.settings[0].id).toBe('context_size');
    expect(read?.settings[0].id).toBe('__mode__');
    expect(read?.usage.context_tokens).toBe(1000);
    expect(read?.slash_commands).toEqual([
      { name: '/compact', description: 'Compact context' }
    ]);
  });

  it('is null when the slot is absent', () => {
    expect(readPersonaState(fakeAwareness({}), 42)).toBeNull();
  });
});

describe('readPersonaStateById', () => {
  it('resolves the client id through the persona list, then reads the state', () => {
    const awareness = fakeAwareness({
      [MANAGER_CLIENT_ID]: {
        personas: [
          { id: 'kiro', name: 'Kiro', avatar_url: null, yjs_client_id: 42 }
        ]
      },
      42: {
        persona: personaState({
          slash_commands: [{ name: '/login', description: null }]
        })
      }
    });
    const state = readPersonaStateById(awareness, 'kiro');
    expect(state?.slash_commands).toEqual([
      { name: '/login', description: null }
    ]);
  });

  it('is null for an unknown persona id', () => {
    const awareness = fakeAwareness({
      [MANAGER_CLIENT_ID]: { personas: [] }
    });
    expect(readPersonaStateById(awareness, 'ghost')).toBeNull();
  });
});
