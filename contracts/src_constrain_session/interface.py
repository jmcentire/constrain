# === Session Lifecycle and Persistence Manager (src_constrain_session) v1 ===
#  Dependencies: pathlib, json, sys, models, posture
# Manages session creation, persistence, and retrieval for the constrain system. Provides lifecycle management including phase transitions, file-based persistence in JSON format, and session discovery utilities.

# Module invariants:
#   - ALLOWED_TRANSITIONS defines valid phase transitions: understand→challenge, challenge→synthesize, synthesize→complete
#   - _sessions_dir is always base_path/.constrain/sessions
#   - Session files are stored as {session_id}.json
#   - Atomic writes use .json.tmp temporary files
#   - Phase.complete sessions are considered complete and filtered by find_latest_incomplete

class SessionManager:
    """Class that manages session creation, persistence, and retrieval operations"""
    base_path: Path                          # required, Base directory path for session storage
    _sessions_dir: Path                      # required, Directory path where session JSON files are stored (.constrain/sessions)

class SessionDict:
    """Dictionary structure returned by list_all for session summaries"""
    id: str                                  # required
    phase: str                               # required
    posture: str                             # required
    created_at: Any                          # required
    updated_at: Any                          # required
    understand_rounds: Any                   # required
    challenge_rounds: Any                    # required
    is_complete: bool                        # required

def __init__(
    self: SessionManager,
    base_path: str | Path,
) -> None:
    """
    Initialize SessionManager with a base path for session storage. Sets up the sessions directory structure under .constrain/sessions.

    Postconditions:
      - self.base_path is set to Path(base_path)
      - self._sessions_dir is set to base_path/.constrain/sessions

    Side effects: mutates_state
    Idempotent: no
    """
    ...

def create(
    self: SessionManager,
    posture_override: Posture | None = None,
) -> Session:
    """
    Create a new session with the specified or auto-selected posture. Delegates posture selection to select_posture function.

    Postconditions:
      - Returns a new Session instance with selected posture

    Side effects: none
    Idempotent: no
    """
    ...

def save(
    self: SessionManager,
    session: Session,
) -> None:
    """
    Persist a session to disk as JSON. Updates session timestamp, creates directory if needed, writes atomically via temporary file, and optionally checks .gitignore on first save.

    Preconditions:
      - session has valid id attribute
      - session.model_dump_json() is callable

    Postconditions:
      - session.touch() has been called
      - _sessions_dir exists
      - Session JSON file exists at _sessions_dir/{session.id}.json
      - _check_gitignore() called if this was the first save

    Errors:
      - OSError during file write (RuntimeError): File system operation fails (write_text or replace)
          message: Failed to save session: {e}

    Side effects: mutates_state, writes_file, logging
    Idempotent: no
    """
    ...

def load(
    self: SessionManager,
    session_id: str,
) -> Session:
    """
    Load a session from disk by its ID. Reads JSON file and validates it into a Session object.

    Preconditions:
      - Session file {session_id}.json exists in _sessions_dir

    Postconditions:
      - Returns a valid Session object loaded from JSON

    Errors:
      - Session file not found (FileNotFoundError): Session file does not exist
          message: No session found with ID '{session_id}'.
      - OSError during file read (RuntimeError): File system read operation fails
          message: Failed to read session file: {e}
      - JSON validation error (ValueError): Session file cannot be parsed or validated
          message: Session file is corrupted: {e}

    Side effects: reads_file
    Idempotent: no
    """
    ...

def find_latest_incomplete(
    self: SessionManager,
) -> Session | None:
    """
    Find the most recently updated incomplete session. Scans all session files and returns the latest non-complete session by updated_at timestamp. Silently skips corrupted files.

    Postconditions:
      - Returns None if no sessions exist or all are complete
      - Returns Session with latest updated_at among incomplete sessions

    Side effects: reads_file
    Idempotent: no
    """
    ...

def list_all(
    self: SessionManager,
) -> list[dict]:
    """
    List all sessions with summary information. Returns list of dictionaries sorted by updated_at in descending order. Silently skips corrupted files.

    Postconditions:
      - Returns empty list if no sessions exist
      - Returns list of session dictionaries sorted by updated_at (newest first)
      - Each dict contains: id, phase, posture, created_at, updated_at, understand_rounds, challenge_rounds, is_complete

    Side effects: reads_file
    Idempotent: no
    """
    ...

def transition_phase(
    self: SessionManager,
    session: Session,
    to_phase: Phase,
) -> None:
    """
    Transition a session to a new phase according to allowed transitions. Updates session phase and timestamp if transition is valid.

    Preconditions:
      - to_phase must be in ALLOWED_TRANSITIONS[session.phase]

    Postconditions:
      - session.phase is set to to_phase
      - session.touch() has been called

    Errors:
      - Invalid phase transition (ValueError): to_phase is not in allowed transitions for current phase
          message: Cannot transition from '{session.phase.value}' to '{to_phase.value}'.

    Side effects: mutates_state
    Idempotent: no
    """
    ...

def _check_gitignore(
    self: SessionManager,
) -> None:
    """
    Check if .gitignore exists and contains .constrain entry. Prints tip to stderr if .gitignore is missing or doesn't contain .constrain patterns.

    Postconditions:
      - May print tip to stderr if .constrain not in .gitignore

    Side effects: reads_file, logging
    Idempotent: no
    """
    ...

# ── REQUIRED EXPORTS ──────────────────────────────────
# Your implementation module MUST export ALL of these names
# with EXACTLY these spellings. Tests import them by name.
# __all__ = ['SessionManager', 'SessionDict', 'create', 'save', 'load', 'find_latest_incomplete', 'list_all', 'transition_phase', '_check_gitignore']
