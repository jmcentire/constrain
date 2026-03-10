"""
Contract tests for SessionManager component.

Tests cover:
- Initialization and configuration
- Session creation
- Persistence (save/load)
- Query operations (find_latest_incomplete, list_all)
- Phase transitions
- Error handling
- Invariants
"""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime
from io import StringIO

# Import the component under test
from constrain.session import SessionManager, Session, Phase, Posture, select_posture


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tmp_session_manager(tmp_path):
    """Create a SessionManager with a temporary base path."""
    return SessionManager(tmp_path)


@pytest.fixture
def sample_session():
    """Create a sample session for testing."""
    session = Mock(spec=Session)
    session.id = "test-session-123"
    session.phase = Phase.understand
    session.posture = Posture.collaborator
    session.created_at = "2024-01-01T12:00:00+00:00"
    session.updated_at = "2024-01-01T12:30:00+00:00"
    session.understand_rounds = 0
    session.challenge_rounds = 0
    session.model_dump_json = Mock(return_value=json.dumps({
        "id": session.id,
        "phase": "understand",
        "posture": "collaborator",
        "created_at": "2024-01-01T12:00:00+00:00",
        "updated_at": "2024-01-01T12:30:00+00:00",
        "understand_rounds": 0,
        "challenge_rounds": 0,
        "schema_version": 1,
        "round": 0,
        "conversation": [],
        "problem_model": {"system_description": "", "stakeholders": [], "failure_modes": [], "dependencies": [], "assumptions": [], "boundaries": [], "history": [], "success_shape": [], "acceptance_criteria": []},
        "prompt_md": "",
        "constraints_yaml": ""
    }))
    session.touch = Mock()
    return session


@pytest.fixture
def sample_complete_session():
    """Create a complete session for testing."""
    session = Mock(spec=Session)
    session.id = "complete-session-456"
    session.phase = Phase.complete
    session.posture = Posture.skeptic
    session.created_at = "2024-01-01T10:00:00+00:00"
    session.updated_at = "2024-01-01T11:00:00+00:00"
    session.understand_rounds = 3
    session.challenge_rounds = 2
    session.model_dump_json = Mock(return_value=json.dumps({
        "id": session.id,
        "phase": "complete",
        "posture": "skeptic",
        "created_at": "2024-01-01T10:00:00+00:00",
        "updated_at": "2024-01-01T11:00:00+00:00",
        "understand_rounds": 3,
        "challenge_rounds": 2,
        "schema_version": 1,
        "round": 0,
        "conversation": [],
        "problem_model": {"system_description": "", "stakeholders": [], "failure_modes": [], "dependencies": [], "assumptions": [], "boundaries": [], "history": [], "success_shape": [], "acceptance_criteria": []},
        "prompt_md": "",
        "constraints_yaml": ""
    }))
    session.touch = Mock()
    return session


def _rewrite_session(manager, session):
    """Helper: rewrite a session file with the session's current state (bypasses touch)."""
    path = manager._sessions_dir / f"{session.id}.json"
    path.write_text(session.model_dump_json(indent=2), encoding="utf-8")


# ============================================================================
# Test: __init__
# ============================================================================

def test_init_happy_path(tmp_path):
    """Initialize SessionManager with a base path, verify base_path and _sessions_dir are set correctly."""
    manager = SessionManager(tmp_path)
    
    assert manager.base_path == Path(tmp_path)
    assert manager._sessions_dir == Path(tmp_path) / ".constrain" / "sessions"


def test_init_with_path_object(tmp_path):
    """Initialize SessionManager with a Path object instead of string."""
    path_obj = Path(tmp_path)
    manager = SessionManager(path_obj)
    
    assert isinstance(manager.base_path, Path)
    assert manager.base_path == path_obj
    assert manager._sessions_dir == path_obj / ".constrain" / "sessions"


def test_init_with_string_path(tmp_path):
    """Initialize SessionManager with a string path."""
    path_str = str(tmp_path)
    manager = SessionManager(path_str)
    
    assert isinstance(manager.base_path, Path)
    assert manager.base_path == Path(path_str)
    assert manager._sessions_dir == Path(path_str) / ".constrain" / "sessions"


def test_path_with_special_chars(tmp_path):
    """Handle paths with spaces and special characters."""
    special_path = tmp_path / "test path with spaces & special!"
    special_path.mkdir(parents=True, exist_ok=True)
    
    manager = SessionManager(special_path)
    
    assert manager.base_path == special_path
    assert manager._sessions_dir == special_path / ".constrain" / "sessions"


def test_invariant_sessions_dir_path(tmp_path):
    """Verify _sessions_dir is always base_path/.constrain/sessions."""
    # Test with various base paths
    test_paths = [
        tmp_path,
        tmp_path / "subdir",
        tmp_path / "a" / "b" / "c"
    ]
    
    for base_path in test_paths:
        manager = SessionManager(base_path)
        expected = Path(base_path) / ".constrain" / "sessions"
        assert manager._sessions_dir == expected, f"Failed for base_path={base_path}"


# ============================================================================
# Test: create
# ============================================================================

@patch('constrain.session.select_posture')
@patch('constrain.session.Session')
def test_create_happy_path(mock_session_class, mock_select_posture, tmp_session_manager):
    """Create a new session with default posture selection."""
    mock_posture = Posture.collaborator
    mock_select_posture.return_value = mock_posture
    mock_session_instance = Mock(spec=Session)
    mock_session_class.return_value = mock_session_instance
    
    result = tmp_session_manager.create(posture_override=None)
    
    mock_select_posture.assert_called_once_with(None)
    assert result == mock_session_instance


@patch('constrain.session.select_posture')
@patch('constrain.session.Session')
def test_create_with_posture_override(mock_session_class, mock_select_posture, tmp_session_manager):
    """Create a new session with explicit posture override."""
    posture_override = Posture.skeptic
    mock_select_posture.return_value = posture_override
    mock_session_instance = Mock(spec=Session)
    mock_session_class.return_value = mock_session_instance
    
    result = tmp_session_manager.create(posture_override=posture_override)
    
    mock_select_posture.assert_called_once_with(posture_override)
    assert result == mock_session_instance


# ============================================================================
# Test: save
# ============================================================================

def test_save_happy_path(tmp_session_manager, sample_session):
    """Save a session to disk, verify file creation and session.touch() called."""
    tmp_session_manager.save(sample_session)
    
    # Verify touch was called
    sample_session.touch.assert_called_once()
    
    # Verify directory exists
    assert tmp_session_manager._sessions_dir.exists()
    
    # Verify session file exists
    session_file = tmp_session_manager._sessions_dir / f"{sample_session.id}.json"
    assert session_file.exists()
    
    # Verify content
    content = json.loads(session_file.read_text())
    assert content["id"] == sample_session.id


def test_save_creates_directory(tmp_session_manager, sample_session):
    """Save creates sessions directory if it doesn't exist."""
    # Ensure directory doesn't exist
    assert not tmp_session_manager._sessions_dir.exists()
    
    tmp_session_manager.save(sample_session)
    
    # Verify directory was created
    assert tmp_session_manager._sessions_dir.exists()
    assert tmp_session_manager._sessions_dir.is_dir()
    
    # Verify session file exists
    session_file = tmp_session_manager._sessions_dir / f"{sample_session.id}.json"
    assert session_file.exists()


def test_save_atomic_write(tmp_session_manager, sample_session):
    """Save uses atomic write with temporary file."""
    tmp_session_manager.save(sample_session)
    
    # Verify final file exists
    final_file = tmp_session_manager._sessions_dir / f"{sample_session.id}.json"
    assert final_file.exists()
    
    # Verify temporary file does not exist after save
    tmp_file = tmp_session_manager._sessions_dir / f"{sample_session.id}.json.tmp"
    assert not tmp_file.exists()


def test_save_first_save_checks_gitignore(tmp_session_manager, sample_session):
    """First save triggers _check_gitignore."""
    with patch.object(tmp_session_manager, '_check_gitignore') as mock_check:
        tmp_session_manager.save(sample_session)
        mock_check.assert_called_once()


def test_save_file_write_error(tmp_session_manager, sample_session):
    """Save raises RuntimeError when file write fails."""
    # Create the directory first
    tmp_session_manager._sessions_dir.mkdir(parents=True, exist_ok=True)

    with patch('pathlib.Path.write_text', side_effect=OSError("Write failed")):
        with pytest.raises(RuntimeError):
            tmp_session_manager.save(sample_session)


def test_invariant_session_file_naming(tmp_session_manager, sample_session):
    """Verify session files are named {session_id}.json."""
    tmp_session_manager.save(sample_session)
    
    expected_path = tmp_session_manager._sessions_dir / f"{sample_session.id}.json"
    assert expected_path.exists()


def test_save_load_round_trip(tmp_session_manager):
    """Verify session data integrity through save/load cycle."""
    # Create a real session (not mocked) for round-trip test
    with patch('constrain.session.select_posture', return_value=Posture.collaborator):
        original_session = tmp_session_manager.create(None)
    
    # Save the session
    tmp_session_manager.save(original_session)
    
    # Load the session
    loaded_session = tmp_session_manager.load(original_session.id)
    
    # Verify data matches
    assert loaded_session.id == original_session.id
    assert loaded_session.phase == original_session.phase
    assert loaded_session.posture == original_session.posture


def test_json_with_unicode(tmp_session_manager):
    """Handle session data with Unicode characters."""
    session = Mock(spec=Session)
    session.id = "unicode-session-测试"
    session.model_dump_json = Mock(return_value=json.dumps({
        "id": session.id,
        "phase": "understand",
        "posture": "advocate",
        "data": "Unicode: 你好世界 🌍"
    }))
    session.touch = Mock()
    
    tmp_session_manager.save(session)
    
    session_file = tmp_session_manager._sessions_dir / f"{session.id}.json"
    assert session_file.exists()
    
    content = json.loads(session_file.read_text(encoding='utf-8'))
    assert content["data"] == "Unicode: 你好世界 🌍"


# ============================================================================
# Test: load
# ============================================================================

def test_load_happy_path(tmp_session_manager, sample_session):
    """Load an existing session by ID, verify Session object is returned."""
    # Save a session first
    tmp_session_manager.save(sample_session)
    
    # Load it back
    loaded_session = tmp_session_manager.load(sample_session.id)
    
    assert isinstance(loaded_session, Session)
    assert loaded_session.id == sample_session.id


def test_load_file_not_found(tmp_session_manager):
    """Load raises error when session file does not exist."""
    with pytest.raises(Exception) as exc_info:
        tmp_session_manager.load("nonexistent-session")
    
    assert "not found" in str(exc_info.value).lower() or isinstance(exc_info.value, FileNotFoundError)


def test_load_os_error(tmp_session_manager, sample_session):
    """Load raises RuntimeError when file read fails."""
    # Save a session first
    tmp_session_manager.save(sample_session)

    with patch('pathlib.Path.read_text', side_effect=OSError("Read failed")):
        with pytest.raises(RuntimeError):
            tmp_session_manager.load(sample_session.id)


def test_load_json_validation_error(tmp_session_manager):
    """Load raises validation error when JSON is invalid."""
    # Create a malformed JSON file
    tmp_session_manager._sessions_dir.mkdir(parents=True, exist_ok=True)
    bad_file = tmp_session_manager._sessions_dir / "bad-session.json"
    bad_file.write_text("invalid json content {{{")
    
    with pytest.raises(Exception):  # Could be JSONDecodeError or ValidationError
        tmp_session_manager.load("bad-session")


# ============================================================================
# Test: find_latest_incomplete
# ============================================================================

def test_find_latest_incomplete_happy_path(tmp_session_manager):
    """Find the most recently updated incomplete session."""
    # Create multiple incomplete sessions with different timestamps
    with patch('constrain.session.select_posture', return_value=Posture.collaborator):
        session1 = tmp_session_manager.create(None)
        tmp_session_manager.save(session1)
        # Overwrite updated_at after save (save calls touch which resets it)
        session1.updated_at = "2024-01-01T10:00:00+00:00"
        _rewrite_session(tmp_session_manager, session1)

        session2 = tmp_session_manager.create(None)
        tmp_session_manager.save(session2)
        session2.updated_at = "2024-01-01T12:00:00+00:00"  # Latest
        _rewrite_session(tmp_session_manager, session2)

        session3 = tmp_session_manager.create(None)
        tmp_session_manager.save(session3)
        session3.updated_at = "2024-01-01T11:00:00+00:00"
        _rewrite_session(tmp_session_manager, session3)

    latest = tmp_session_manager.find_latest_incomplete()

    assert latest is not None
    assert latest.id == session2.id
    assert latest.phase != Phase.complete


def test_find_latest_incomplete_no_sessions(tmp_session_manager):
    """Return None when no sessions exist."""
    result = tmp_session_manager.find_latest_incomplete()
    assert result is None


def test_find_latest_incomplete_all_complete(tmp_session_manager):
    """Return None when all sessions are complete."""
    with patch('constrain.session.select_posture', return_value=Posture.collaborator):
        session = tmp_session_manager.create(None)
        session.phase = Phase.complete
        tmp_session_manager.save(session)

    result = tmp_session_manager.find_latest_incomplete()
    assert result is None


def test_find_latest_incomplete_skips_corrupted(tmp_session_manager):
    """Silently skip corrupted session files."""
    # Create a valid session
    with patch('constrain.session.select_posture', return_value=Posture.collaborator):
        valid_session = tmp_session_manager.create(None)
        tmp_session_manager.save(valid_session)
    
    # Create a corrupted file
    tmp_session_manager._sessions_dir.mkdir(parents=True, exist_ok=True)
    corrupted_file = tmp_session_manager._sessions_dir / "corrupted.json"
    corrupted_file.write_text("invalid json {{{")
    
    # Should return the valid session without raising error
    result = tmp_session_manager.find_latest_incomplete()
    assert result is not None
    assert result.id == valid_session.id


# ============================================================================
# Test: list_all
# ============================================================================

def test_list_all_happy_path(tmp_session_manager):
    """List all sessions with summary information, sorted by updated_at."""
    # Create multiple sessions
    with patch('constrain.session.select_posture', return_value=Posture.collaborator):
        session1 = tmp_session_manager.create(None)
        tmp_session_manager.save(session1)
        session1.updated_at = "2024-01-01T10:00:00+00:00"
        _rewrite_session(tmp_session_manager, session1)

        session2 = tmp_session_manager.create(None)
        tmp_session_manager.save(session2)
        session2.updated_at = "2024-01-01T12:00:00+00:00"
        _rewrite_session(tmp_session_manager, session2)

        session3 = tmp_session_manager.create(None)
        tmp_session_manager.save(session3)
        session3.updated_at = "2024-01-01T11:00:00+00:00"
        _rewrite_session(tmp_session_manager, session3)

    sessions = tmp_session_manager.list_all()

    assert isinstance(sessions, list)
    assert len(sessions) == 3

    # Verify sorted by updated_at descending
    assert sessions[0]["id"] == session2.id  # Latest
    assert sessions[1]["id"] == session3.id
    assert sessions[2]["id"] == session1.id  # Oldest


def test_list_all_empty(tmp_session_manager):
    """Return empty list when no sessions exist."""
    result = tmp_session_manager.list_all()
    assert result == []


def test_list_all_skips_corrupted(tmp_session_manager):
    """Silently skip corrupted session files."""
    # Create a valid session
    with patch('constrain.session.select_posture', return_value=Posture.collaborator):
        valid_session = tmp_session_manager.create(None)
        tmp_session_manager.save(valid_session)
    
    # Create a corrupted file
    corrupted_file = tmp_session_manager._sessions_dir / "corrupted.json"
    corrupted_file.write_text("invalid json {{{")
    
    # Should return only valid sessions
    sessions = tmp_session_manager.list_all()
    assert len(sessions) == 1
    assert sessions[0]["id"] == valid_session.id


def test_list_all_dict_structure(tmp_session_manager):
    """Verify each returned dict has correct structure."""
    with patch('constrain.session.select_posture', return_value=Posture.collaborator):
        session = tmp_session_manager.create(None)
        tmp_session_manager.save(session)
    
    sessions = tmp_session_manager.list_all()
    
    assert len(sessions) == 1
    session_dict = sessions[0]
    
    required_fields = [
        "id", "phase", "posture", "created_at", "updated_at",
        "understand_rounds", "challenge_rounds", "is_complete"
    ]
    
    for field in required_fields:
        assert field in session_dict, f"Missing field: {field}"


# ============================================================================
# Test: transition_phase
# ============================================================================

def test_transition_phase_understand_to_challenge(tmp_session_manager):
    """Valid transition from understand to challenge phase."""
    with patch('constrain.session.select_posture', return_value=Posture.collaborator):
        session = tmp_session_manager.create(None)
        session.phase = Phase.understand
        old_updated = session.updated_at

        tmp_session_manager.transition_phase(session, Phase.challenge)

        assert session.phase == Phase.challenge
        assert session.updated_at >= old_updated


def test_transition_phase_challenge_to_synthesize(tmp_session_manager):
    """Valid transition from challenge to synthesize phase."""
    with patch('constrain.session.select_posture', return_value=Posture.collaborator):
        session = tmp_session_manager.create(None)
        session.phase = Phase.challenge
        old_updated = session.updated_at

        tmp_session_manager.transition_phase(session, Phase.synthesize)

        assert session.phase == Phase.synthesize
        assert session.updated_at >= old_updated


def test_transition_phase_synthesize_to_complete(tmp_session_manager):
    """Valid transition from synthesize to complete phase."""
    with patch('constrain.session.select_posture', return_value=Posture.collaborator):
        session = tmp_session_manager.create(None)
        session.phase = Phase.synthesize
        old_updated = session.updated_at

        tmp_session_manager.transition_phase(session, Phase.complete)

        assert session.phase == Phase.complete
        assert session.updated_at >= old_updated


def test_transition_phase_invalid_transition(tmp_session_manager):
    """Invalid transition raises error."""
    with patch('constrain.session.select_posture', return_value=Posture.collaborator):
        session = tmp_session_manager.create(None)
        original_phase = session.phase = Phase.understand
        
        with pytest.raises(Exception) as exc_info:
            tmp_session_manager.transition_phase(session, Phase.synthesize)
        
        assert "transition" in str(exc_info.value).lower()
        # Phase should be unchanged
        assert session.phase == original_phase


def test_invariant_phase_transitions(tmp_session_manager):
    """Verify ALLOWED_TRANSITIONS defines correct transition matrix."""
    # Test all valid transitions
    valid_transitions = [
        (Phase.understand, Phase.challenge),
        (Phase.challenge, Phase.synthesize),
        (Phase.synthesize, Phase.complete),
    ]
    
    for from_phase, to_phase in valid_transitions:
        with patch('constrain.session.select_posture', return_value=Posture.collaborator):
            session = tmp_session_manager.create(None)
            session.phase = from_phase
            
            # Should not raise
            tmp_session_manager.transition_phase(session, to_phase)
            assert session.phase == to_phase
    
    # Test some invalid transitions
    invalid_transitions = [
        (Phase.understand, Phase.synthesize),
        (Phase.understand, Phase.complete),
        (Phase.challenge, Phase.complete),
    ]
    
    for from_phase, to_phase in invalid_transitions:
        with patch('constrain.session.select_posture', return_value=Posture.collaborator):
            session = tmp_session_manager.create(None)
            session.phase = from_phase
            
            with pytest.raises(Exception):
                tmp_session_manager.transition_phase(session, to_phase)


# ============================================================================
# Test: _check_gitignore
# ============================================================================

def test_check_gitignore_missing(tmp_session_manager):
    """Print tip when .gitignore is missing."""
    # Ensure .gitignore doesn't exist
    gitignore_path = tmp_session_manager.base_path / ".gitignore"
    assert not gitignore_path.exists()
    
    # Capture stderr
    with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
        tmp_session_manager._check_gitignore()
        output = mock_stderr.getvalue()
        
        # Should contain tip about .constrain
        assert ".constrain" in output or "gitignore" in output.lower()


def test_check_gitignore_missing_entry(tmp_session_manager):
    """Print tip when .gitignore exists but doesn't contain .constrain."""
    # Create .gitignore without .constrain
    gitignore_path = tmp_session_manager.base_path / ".gitignore"
    gitignore_path.write_text("*.pyc\n__pycache__/\n")
    
    with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
        tmp_session_manager._check_gitignore()
        output = mock_stderr.getvalue()
        
        # Should contain tip about .constrain
        assert ".constrain" in output or "gitignore" in output.lower()


def test_check_gitignore_exists_with_entry(tmp_session_manager):
    """No tip when .gitignore contains .constrain."""
    # Create .gitignore with .constrain
    gitignore_path = tmp_session_manager.base_path / ".gitignore"
    gitignore_path.write_text("*.pyc\n.constrain/\n__pycache__/\n")
    
    with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
        tmp_session_manager._check_gitignore()
        output = mock_stderr.getvalue()
        
        # Should not print tip
        assert output == "" or len(output.strip()) == 0


# ============================================================================
# Additional Edge Cases and Integration Tests
# ============================================================================

def test_multiple_saves_idempotent(tmp_session_manager, sample_session):
    """Multiple saves of the same session should be idempotent."""
    tmp_session_manager.save(sample_session)
    first_save_content = (tmp_session_manager._sessions_dir / f"{sample_session.id}.json").read_text()
    
    tmp_session_manager.save(sample_session)
    second_save_content = (tmp_session_manager._sessions_dir / f"{sample_session.id}.json").read_text()
    
    assert first_save_content == second_save_content


def test_session_persistence_across_managers(tmp_path):
    """Sessions saved by one manager can be loaded by another."""
    manager1 = SessionManager(tmp_path)
    
    with patch('constrain.session.select_posture', return_value=Posture.collaborator):
        session = manager1.create(None)
        manager1.save(session)
    
    # Create a new manager with the same path
    manager2 = SessionManager(tmp_path)
    loaded_session = manager2.load(session.id)
    
    assert loaded_session.id == session.id
    assert loaded_session.phase == session.phase


def test_concurrent_session_saves(tmp_session_manager):
    """Multiple sessions can be saved concurrently."""
    with patch('constrain.session.select_posture', return_value=Posture.collaborator):
        session1 = tmp_session_manager.create(None)
        session2 = tmp_session_manager.create(None)
        session3 = tmp_session_manager.create(None)
    
    tmp_session_manager.save(session1)
    tmp_session_manager.save(session2)
    tmp_session_manager.save(session3)
    
    # All should exist
    assert (tmp_session_manager._sessions_dir / f"{session1.id}.json").exists()
    assert (tmp_session_manager._sessions_dir / f"{session2.id}.json").exists()
    assert (tmp_session_manager._sessions_dir / f"{session3.id}.json").exists()


def test_list_all_with_mixed_completion_states(tmp_session_manager):
    """List all returns both complete and incomplete sessions."""
    with patch('constrain.session.select_posture', return_value=Posture.collaborator):
        incomplete = tmp_session_manager.create(None)
        incomplete.updated_at = "2024-01-01T12:00:00+00:00"
        tmp_session_manager.save(incomplete)

        complete = tmp_session_manager.create(None)
        complete.phase = Phase.complete
        complete.updated_at = "2024-01-01T10:00:00+00:00"
        tmp_session_manager.save(complete)

    sessions = tmp_session_manager.list_all()

    assert len(sessions) == 2
    assert any(s["is_complete"] for s in sessions)
    assert any(not s["is_complete"] for s in sessions)


def test_find_latest_incomplete_with_mixed_states(tmp_session_manager):
    """Find latest incomplete ignores complete sessions."""
    with patch('constrain.session.select_posture', return_value=Posture.collaborator):
        incomplete = tmp_session_manager.create(None)
        incomplete.updated_at = "2024-01-01T10:00:00+00:00"
        tmp_session_manager.save(incomplete)

        complete = tmp_session_manager.create(None)
        complete.phase = Phase.complete
        complete.updated_at = "2024-01-01T12:00:00+00:00"  # Later but complete
        tmp_session_manager.save(complete)

    latest = tmp_session_manager.find_latest_incomplete()

    assert latest is not None
    assert latest.id == incomplete.id
    assert latest.phase != Phase.complete
