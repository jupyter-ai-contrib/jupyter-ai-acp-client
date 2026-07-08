/**
 * Tests for the usage chip's number formatters.
 */

import { formatCost, formatTokens } from '../persona-controls';

describe('formatTokens', () => {
  it('keeps small counts as-is', () => {
    expect(formatTokens(950)).toEqual('950');
  });

  it('formats thousands with a lowercase k', () => {
    expect(formatTokens(41500)).toEqual('41.5k');
  });

  it('rounds up to the next tier at the boundary instead of exponential form', () => {
    expect(formatTokens(999500)).toEqual('1M');
  });

  it('formats millions', () => {
    expect(formatTokens(1240000)).toEqual('1.24M');
  });

  it('formats billions', () => {
    expect(formatTokens(2500000000)).toEqual('2.5B');
  });
});

describe('formatCost', () => {
  it('renders USD with a dollar sign', () => {
    expect(formatCost(0.41, 'USD')).toEqual('$0.41');
  });

  it('renders other currencies with their code', () => {
    expect(formatCost(1.5, 'EUR')).toEqual('1.50 EUR');
  });
});
