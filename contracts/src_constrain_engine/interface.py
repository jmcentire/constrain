# === Conversation Engine - Three-Phase Interview Orchestration (src_constrain_engine) v1 ===
#  Dependencies: anthropic, json, re, sys, time, dataclasses, pathlib, typing
# Orchestrates a three-phase conversation interview (understand, challenge, synthesize) using Claude API. Manages conversation state, handles user I/O, performs API calls with retry logic, parses structured responses, and generates final artifacts (prompt.md and constraints.yaml).

# Module invariants:
#   - MODEL constant is 'claude-sonnet-4-20250514'
#   - Phase order is [understand, challenge, synthesize, complete]
#   - Retry delays are [1, 2, 4] seconds for API calls
#   - Max API tokens is 4096
#   - Synthesis phase allows exactly one revision cycle
#   - Messages must alternate user/assistant roles for API calls

class TerminalIO:
    """Protocol defining terminal I/O interface"""
    pass

class DefaultIO:
    """Production terminal I/O implementation using print/input"""
    pass

class EngineConfig:
    """Configuration for conversation phase round limits"""
    understand_min: int                      # required, Minimum rounds for understand phase
    understand_max: int                      # required, Maximum rounds for understand phase
    challenge_min: int                       # required, Minimum rounds for challenge phase
    challenge_max: int                       # required, Maximum rounds for challenge phase

class ConversationEngine:
    """Main orchestrator for the three-phase constrain interview"""
    session: Session                         # required, Current conversation session state
    session_mgr: SessionManager              # required, Persistence manager for sessions
    client: anthropic.Anthropic              # required, Anthropic API client
    io: TerminalIO                           # required, Terminal I/O interface
    config: EngineConfig                     # required, Round limit configuration

def DefaultIO.display(
    self: DefaultIO,
    text: str,
) -> None:
    """
    Display text to terminal using print

    Side effects: Prints to stdout
    Idempotent: no
    """
    ...

def DefaultIO.prompt(
    self: DefaultIO,
    prefix: str,
) -> str:
    """
    Prompt user for input from terminal

    Errors:
      - EOFError (EOFError): User sends EOF (Ctrl+D)

    Side effects: Reads from stdin
    Idempotent: no
    """
    ...

def ConversationEngine.__init__(
    self: ConversationEngine,
    session: Session,
    session_mgr: SessionManager,
    client: anthropic.Anthropic | None = None,
    io: TerminalIO | None = None,
    config: EngineConfig | None = None,
) -> None:
    """
    Initialize ConversationEngine with session, manager, optional client, I/O, and config

    Postconditions:
      - self.session is set to session parameter
      - self.session_mgr is set to session_mgr parameter
      - self.client is initialized (creates new Anthropic() if None passed)
      - self.io is initialized (creates DefaultIO() if None passed)
      - self.config is initialized (creates EngineConfig() if None passed)

    Side effects: none
    Idempotent: no
    """
    ...

def ConversationEngine.run_session(
    self: ConversationEngine,
) -> Session:
    """
    Run all remaining phases of the interview. Saves session on KeyboardInterrupt and exits.

    Preconditions:
      - self.session is valid Session object

    Postconditions:
      - All incomplete phases are executed sequentially
      - Session is saved after each phase transition
      - Returns the updated session object

    Errors:
      - KeyboardInterrupt (SystemExit): User sends interrupt signal (Ctrl+C)
          exit_code: 0

    Side effects: Calls _display_resume_summary if session.round > 0, Runs phase loops (_run_phase or _run_synthesis), Transitions phases via session_mgr.transition_phase, Saves session via session_mgr.save, Displays phase transition messages via io.display, Exits with sys.exit(0) on KeyboardInterrupt
    Idempotent: no
    """
    ...

def ConversationEngine._run_phase(
    self: ConversationEngine,
    phase: Phase,
) -> None:
    """
    Run understand or challenge phase conversation loop with round limits and user interaction

    Preconditions:
      - phase is Phase.understand or Phase.challenge

    Postconditions:
      - Conversation proceeds for min_rounds to max_rounds
      - Session is updated with messages and round counts
      - Session is saved after each exchange

    Errors:
      - EOFError (None): User sends EOF during prompt
          behavior: breaks loop and advances to next phase

    Side effects: Calls API via _call_api, Prompts user via io.prompt, Displays responses via io.display, Updates session.conversation with messages, Updates session.problem_model if model updates received, Saves session via session_mgr.save
    Idempotent: no
    """
    ...

def ConversationEngine._run_synthesis(
    self: ConversationEngine,
) -> None:
    """
    Run synthesis phase: generate prompt.md and constraints.yaml artifacts with optional revision cycle

    Postconditions:
      - session.prompt_md is populated
      - session.constraints_yaml is populated
      - Session transitions to Phase.complete
      - Session is saved

    Errors:
      - ValueError (None): parse_synthesis_output fails to parse response
          behavior: displays warning, stores raw output, transitions to complete
      - EOFError (None): User sends EOF when prompted for feedback
          behavior: treats as empty feedback, accepts artifacts

    Side effects: Calls API twice (initial generation + optional revision), Parses synthesis output via parse_synthesis_output, Displays artifacts to user, Prompts for optional feedback, Updates session.prompt_md and session.constraints_yaml, Transitions phase to Phase.complete, Saves session
    Idempotent: no
    """
    ...

def ConversationEngine._call_api(
    self: ConversationEngine,
    system: str,
    messages: list[dict],
) -> str:
    """
    Call Claude API with exponential backoff on transient errors (rate limit, timeout, connection)

    Postconditions:
      - Returns text content from Claude API response on success

    Errors:
      - RuntimeError (RuntimeError): API returns empty response (no content)
      - RuntimeError (RuntimeError): API unavailable after 3 retry attempts
      - AuthenticationError (anthropic.AuthenticationError): API authentication fails

    Side effects: Makes HTTP request to Anthropic API, Sleeps on transient errors (1s, 2s, 4s)
    Idempotent: no
    """
    ...

def ConversationEngine._parse_response(
    self: ConversationEngine,
    raw: str,
) -> tuple[str, dict, bool]:
    """
    Extract display text, problem model update dict, and ready flag from LLM response containing JSON code block

    Postconditions:
      - Returns tuple of (display_text, model_update_dict, ready_flag)
      - If no JSON block found, returns (raw, {}, False)
      - If JSON parse fails, returns (raw, {}, False)
      - Extracts last ```json...``` block if multiple present

    Side effects: none
    Idempotent: no
    """
    ...

def ConversationEngine._api_messages(
    self: ConversationEngine,
) -> list[dict]:
    """
    Build API message list from session conversation history

    Postconditions:
      - Returns list of dicts with 'role' and 'content' keys from session.conversation

    Side effects: none
    Idempotent: no
    """
    ...

def ConversationEngine._add_message(
    self: ConversationEngine,
    role: str,
    content: str,
) -> None:
    """
    Append a message to session conversation and touch session timestamp

    Postconditions:
      - Message(role, content) appended to session.conversation
      - session.touch() called to update timestamp

    Side effects: none
    Idempotent: no
    """
    ...

def ConversationEngine._increment_round(
    self: ConversationEngine,
    phase: Phase,
) -> None:
    """
    Increment session round counter and phase-specific round counter

    Postconditions:
      - session.round incremented by 1
      - session.understand_rounds incremented if phase is understand
      - session.challenge_rounds incremented if phase is challenge

    Side effects: none
    Idempotent: no
    """
    ...

def ConversationEngine._current_rounds(
    self: ConversationEngine,
    phase: Phase,
) -> int:
    """
    Get current round count for the given phase

    Postconditions:
      - Returns session.understand_rounds if phase is understand
      - Returns session.challenge_rounds if phase is challenge
      - Returns 0 for other phases

    Side effects: none
    Idempotent: no
    """
    ...

def ConversationEngine._round_limits(
    self: ConversationEngine,
    phase: Phase,
) -> tuple[int, int]:
    """
    Get min and max round limits for the given phase from config

    Postconditions:
      - Returns (understand_min, understand_max) if phase is understand
      - Returns (challenge_min, challenge_max) if phase is challenge
      - Returns (1, 1) for other phases

    Side effects: none
    Idempotent: no
    """
    ...

def ConversationEngine._phase_done(
    self: ConversationEngine,
    phase: Phase,
) -> bool:
    """
    Check if a phase is already completed based on session state by comparing phase order indices

    Postconditions:
      - Returns True if session.phase is later than the target phase in the phase order
      - Returns False otherwise

    Side effects: none
    Idempotent: no
    """
    ...

def ConversationEngine._display_resume_summary(
    self: ConversationEngine,
) -> None:
    """
    Display resume summary showing session ID, phase, round counts, and partial problem model info

    Side effects: Displays session resume information via io.display
    Idempotent: no
    """
    ...

# ── REQUIRED EXPORTS ──────────────────────────────────
# Your implementation module MUST export ALL of these names
# with EXACTLY these spellings. Tests import them by name.
# __all__ = ['TerminalIO', 'DefaultIO', 'EngineConfig', 'ConversationEngine', 'EOFError', 'SystemExit']
