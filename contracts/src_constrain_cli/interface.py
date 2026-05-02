# === Constrain CLI (src_constrain_cli) v1 ===
#  Dependencies: os, sys, pathlib, click, constrain.engine, constrain.models, constrain.session, constrain.synthesizer
# Click-based command-line interface for the Constrain tool. Provides commands to create, resume, list, and display sessions. Handles session management, artifact generation, and user interaction for the conversation engine that finds boundaries between specification and intent.

# Module invariants:
#   - API key must be non-empty when ensure_api_key() succeeds
#   - understand_min <= understand_max in resolved EngineConfig
#   - challenge_min <= challenge_max in resolved EngineConfig
#   - Default understand_min=2, understand_max=10, challenge_min=2, challenge_max=10
#   - Environment variables: ANTHROPIC_API_KEY, CONSTRAIN_MIN_UNDERSTAND, CONSTRAIN_MAX_UNDERSTAND, CONSTRAIN_MIN_CHALLENGE, CONSTRAIN_MAX_CHALLENGE
#   - Artifacts written are prompt.md and constraints.yaml
#   - SafeGroup exits with code 1 for Abort and general errors, 130 for KeyboardInterrupt

class SafeGroup:
    """Click group with centralized exception handling. Catches and handles ClickException, Abort, KeyboardInterrupt, SystemExit, and general exceptions."""
    pass

def ensure_api_key() -> str:
    """
    Retrieves and validates the ANTHROPIC_API_KEY from environment variables. Strips whitespace and ensures the key is non-empty.

    Postconditions:
      - Returns a non-empty stripped API key string

    Errors:
      - missing_api_key (click.ClickException): ANTHROPIC_API_KEY environment variable is not set or is empty/whitespace
          message: ANTHROPIC_API_KEY not set. Export it or add it to your shell profile.

    Side effects: Reads ANTHROPIC_API_KEY from environment variables
    Idempotent: no
    """
    ...

def _round_options(
    f: function,
) -> function:
    """
    Decorator function that adds four Click options (--min-understand, --max-understand, --min-challenge, --max-challenge) to a command function. Each option is an optional integer with a default of None.

    Postconditions:
      - Returns the decorated function with four additional Click option parameters

    Side effects: none
    Idempotent: no
    """
    ...

def _resolve_int(
    cli_val: int | None,
    env_var: str,
    default: int,
) -> int:
    """
    Resolves an integer value from CLI argument, environment variable, or default. Validates that environment variable values are positive integers if present.

    Postconditions:
      - Returns cli_val if not None
      - Otherwise returns parsed environment variable if set and valid
      - Otherwise returns default value

    Errors:
      - invalid_env_value (click.ClickException): Environment variable is set but cannot be parsed as a positive integer
          message: Invalid value for {env_var}: expected a positive integer, got '{env}'.

    Side effects: Reads from os.environ
    Idempotent: no
    """
    ...

def resolve_config(
    min_understand: int | None,
    max_understand: int | None,
    min_challenge: int | None,
    max_challenge: int | None,
) -> EngineConfig:
    """
    Resolves EngineConfig from optional CLI arguments and environment variables. Validates that min values do not exceed max values for both understand and challenge phases.

    Postconditions:
      - Returns EngineConfig with understand_min <= understand_max
      - Returns EngineConfig with challenge_min <= challenge_max

    Errors:
      - understand_min_exceeds_max (click.ClickException): Resolved understand_min > understand_max
          message: min-understand ({cfg.understand_min}) cannot exceed max-understand ({cfg.understand_max}).
      - challenge_min_exceeds_max (click.ClickException): Resolved challenge_min > challenge_max
          message: min-challenge ({cfg.challenge_min}) cannot exceed max-challenge ({cfg.challenge_max}).

    Side effects: Calls _resolve_int which reads environment variables
    Idempotent: no
    """
    ...

def _run_engine(
    session: Session,
    mgr: SessionManager,
    config: EngineConfig,
) -> None:
    """
    Creates a ConversationEngine with the given session, manager, and config, runs the session, and writes artifacts (prompt.md, constraints.yaml) to current directory if session completes successfully.

    Postconditions:
      - Session has been executed via engine.run_session()
      - If session.phase is Phase.complete and session.prompt_md exists, artifacts are written to current directory
      - User is notified of artifact paths via click.echo

    Errors:
      - user_aborts_overwrite (click.Abort): Artifacts exist and user declines to overwrite in _confirm_overwrite

    Side effects: Executes ConversationEngine which may interact with external APIs, May write prompt.md and constraints.yaml files to current directory, Outputs to stdout via click.echo, May prompt user for confirmation via _confirm_overwrite
    Idempotent: no
    """
    ...

def _confirm_overwrite(
    cwd: Path,
) -> bool:
    """
    Checks if prompt.md or constraints.yaml exist in the given directory. If they do, prompts user for confirmation to overwrite. Returns True if user confirms, raises Abort if declined.

    Postconditions:
      - Returns False if no artifacts exist (no conflict)
      - Returns True if artifacts exist and user confirms overwrite

    Errors:
      - user_declines_overwrite (click.Abort): Artifacts exist and user does not confirm overwrite

    Side effects: Checks filesystem for existence of prompt.md and constraints.yaml, Prompts user via click.confirm if files exist
    Idempotent: no
    """
    ...

def cli(
    ctx: click.Context,
    min_understand: int | None,
    max_understand: int | None,
    min_challenge: int | None,
    max_challenge: int | None,
) -> None:
    """
    Main Click group command. When invoked without subcommand, checks for incomplete sessions and prompts to resume, or creates a new session. Decorated with @click.group, @_round_options, and @click.pass_context.

    Postconditions:
      - If subcommand is invoked, returns immediately without action
      - Otherwise ensures API key is set
      - If incomplete session exists, prompts user to resume/decline/list
      - Creates and runs new session if no incomplete session or user declines resume

    Errors:
      - missing_api_key (click.ClickException): ANTHROPIC_API_KEY not set
      - invalid_config (click.ClickException): Config validation fails in resolve_config

    Side effects: Reads environment variables, Creates SessionManager with current directory, May create new session, May save session, May run engine, Outputs to stdout via click.echo, Prompts user for input via click.prompt
    Idempotent: no
    """
    ...

def cmd_new(
    min_understand: int | None,
    max_understand: int | None,
    min_challenge: int | None,
    max_challenge: int | None,
) -> None:
    """
    Click command to start a new session, ignoring any incomplete sessions. Decorated with @cli.command('new') and @_round_options.

    Postconditions:
      - Creates a new session unconditionally
      - Saves and runs the new session

    Errors:
      - missing_api_key (click.ClickException): ANTHROPIC_API_KEY not set
      - invalid_config (click.ClickException): Config validation fails

    Side effects: Creates SessionManager, Creates and saves new session, Runs engine
    Idempotent: no
    """
    ...

def cmd_resume(
    session_id: str | None,
) -> None:
    """
    Click command to resume an incomplete session by ID or latest. Decorated with @cli.command('resume') and @click.argument('session_id', required=False).

    Postconditions:
      - Loads specified or latest incomplete session
      - Runs the session with default config

    Errors:
      - missing_api_key (click.ClickException): ANTHROPIC_API_KEY not set
      - session_not_found (click.ClickException): Specified session_id does not exist (FileNotFoundError)
          message: Session '{session_id}' not found.
      - session_load_error (click.ClickException): Session load raises ValueError
          message: Could not load session '{session_id}': {e}
      - session_already_complete (click.ClickException): Loaded session has phase == Phase.complete
          message: Session '{session_id}' is already completed. Use 'constrain show' to view its artifacts.
      - no_incomplete_sessions (click.ClickException): No session_id provided and no incomplete sessions found
          message: No incomplete sessions found. Use 'constrain new' to start one.

    Side effects: Loads session from filesystem, Runs engine
    Idempotent: no
    """
    ...

def cmd_show() -> None:
    """
    Click command to display artifacts (prompt.md and constraints.yaml) from the most recent completed session. Decorated with @cli.command('show').

    Postconditions:
      - Outputs prompt.md and constraints.yaml content to stdout

    Errors:
      - no_completed_sessions (click.ClickException): No completed sessions found in session list
          message: No completed sessions found. Use 'constrain' to start one.
      - artifacts_not_found (click.ClickException): Latest completed session has no prompt_md
          message: Artifacts not found in session {session.id[:8]}. They may not have been generated.

    Side effects: Reads session data from filesystem, Outputs to stdout
    Idempotent: no
    """
    ...

def cmd_list() -> None:
    """
    Click command to list all sessions. Decorated with @cli.command('list'). Delegates to _do_list.

    Postconditions:
      - Displays table of all sessions

    Errors:
      - no_sessions (click.ClickException): No sessions found (raised by _do_list)

    Side effects: Reads session data, Outputs to stdout
    Idempotent: no
    """
    ...

def _do_list(
    mgr: SessionManager,
) -> None:
    """
    Displays a formatted table of all sessions with ID, phase, rounds, posture, and updated timestamp.

    Postconditions:
      - Outputs formatted session table to stdout

    Errors:
      - no_sessions (click.ClickException): mgr.list_all() returns empty list
          message: No sessions found. Use 'constrain' to start one.

    Side effects: Calls mgr.list_all(), Outputs to stdout via click.echo
    Idempotent: no
    """
    ...

def main() -> None:
    """
    Entry point for the constrain CLI. Simply invokes the cli() Click group.

    Postconditions:
      - Invokes cli() Click group which handles command dispatch

    Side effects: Delegates to cli() which has extensive side effects
    Idempotent: no
    """
    ...

def invoke(
    self: SafeGroup,
    ctx: click.Context,
) -> None:
    """
    SafeGroup.invoke override that wraps Click group invocation with centralized exception handling. Catches ClickException, Abort, KeyboardInterrupt, SystemExit, and general exceptions, providing appropriate error messages and exit codes.

    Postconditions:
      - Invokes parent Click.Group.invoke(ctx)
      - Handles exceptions and exits with appropriate codes: 1 for Abort/general errors, 130 for KeyboardInterrupt

    Errors:
      - click_exception (click.ClickException): click.ClickException raised during invocation
          message: Re-raised as-is
      - abort (SystemExit): click.Abort raised
          exit_code: 1
          message: Aborted.
      - keyboard_interrupt (SystemExit): KeyboardInterrupt (Ctrl+C)
          exit_code: 130
          message: \nInterrupted.
      - system_exit (SystemExit): SystemExit raised
          message: Re-raised as-is
      - general_exception (SystemExit): Any other exception
          exit_code: 1
          message: Error: {e}

    Side effects: May call sys.exit() with codes 1 or 130, Outputs to stderr via click.echo(..., err=True)
    Idempotent: no
    """
    ...

# ── REQUIRED EXPORTS ──────────────────────────────────
# Your implementation module MUST export ALL of these names
# with EXACTLY these spellings. Tests import them by name.
# __all__ = ['SafeGroup', 'ensure_api_key', '_round_options', '_resolve_int', 'resolve_config', '_run_engine', '_confirm_overwrite', 'cli', 'cmd_new', 'cmd_resume', 'cmd_show', 'cmd_list', '_do_list', 'main', 'invoke', 'SystemExit']
