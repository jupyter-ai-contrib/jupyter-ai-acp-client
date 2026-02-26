// Required: makes this a module file so `declare module` below augments
// @jupyter/chat rather than replacing it with an ambient module declaration.
export {};

declare module '@jupyter/chat' {
  export interface IToolCallDiff {
    path: string;
    new_text: string;
    old_text?: string;
  }

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
    /**
     * Permission options
     */
    permission_options?: IPermissionOption[];
    /**
     * Whether the permission request is waiting for user.
     */
    permission_status?: 'pending' | 'resolved';
    /**
     * The option_id the user selected.
     */
    selected_option_id?: string;
    /**
     * The ACP session ID this tool call belongs to.
     */
    session_id?: string;
    /**
     * File diffs from ACP FileEditToolCallContent.
     */
    diffs?: IToolCallDiff[];
  }

  export interface IPermissionOption {
    option_id: string;
    title: string;
    description?: string;
  }

  export interface IMessageMetadata {
    tool_calls?: IToolCall[];
  }
}
