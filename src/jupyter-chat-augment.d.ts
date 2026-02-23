// Required: makes this a module file so `declare module` below augments
// @jupyter/chat rather than replacing it with an ambient module declaration.
export {};

declare module '@jupyter/chat' {
  export interface IToolCall {
    /**
     * Unique identifier for this tool call, used to correlate events
     * across the tool call lifecycle.
     */
    tool_call_id: string;
    /**
     * Human-readable label displayed in the message preamble.
     */
    title: string;
    /**
     * The category of tool operation.
     */
    kind?:
      | 'read'
      | 'edit'
      | 'delete'
      | 'move'
      | 'search'
      | 'execute'
      | 'think'
      | 'fetch'
      | 'switch_mode'
      | (string & {});
    /**
     * Current execution status.
     */
    status?: 'in_progress' | 'completed' | 'failed' | (string & {});
    /**
     * Raw return value from tool execution.
     */
    raw_output?: unknown;
    /**
     * File paths or resource URIs involved in this tool call.
     */
    locations?: string[];
  }

  export interface IMessageMetadata {
    tool_calls?: IToolCall[];
  }
}
