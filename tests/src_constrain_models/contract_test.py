"""
Contract test suite for constrain.models
Tests the Posture, Phase, Severity enums, Message, ProblemModel, Constraint, Session structs,
and the apply_update() and touch() methods.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch
import uuid
import re

# Import the component under test
from constrain.models import (
    Posture,
    Phase,
    Severity,
    Message,
    ProblemModel,
    Constraint,
    Session
)


# ============================================================================
# Fixtures - Test Data Builders
# ============================================================================

@pytest.fixture
def sample_message():
    """Create a sample Message instance."""
    return Message(
        role="user",
        content="Test message",
        timestamp="2024-01-01T12:00:00Z"
    )


@pytest.fixture
def sample_problem_model():
    """Create a sample ProblemModel instance."""
    return ProblemModel(
        system_description="Test system",
        stakeholders=["user", "admin"],
        failure_modes=[{"mode": "crash", "impact": "high"}],
        dependencies=["database", "api"],
        assumptions=["users are authenticated"],
        boundaries=["web interface only"],
        history=["version 1.0"],
        success_shape=["99% uptime"],
        acceptance_criteria=["all tests pass"]
    )


@pytest.fixture
def sample_constraint():
    """Create a sample Constraint instance."""
    return Constraint(
        id="c1",
        boundary="API rate limiting",
        condition="max 100 requests per minute",
        violation="exceeding rate limit",
        severity=Severity.must,
        rationale="prevent abuse"
    )


@pytest.fixture
def sample_session():
    """Create a sample Session instance."""
    return Session(
        id=str(uuid.uuid4()),
        schema_version=1,
        posture=Posture.collaborator,
        phase=Phase.understand,
        round=1,
        understand_rounds=2,
        challenge_rounds=2,
        conversation=[],
        problem_model=ProblemModel(
            system_description="",
            stakeholders=[],
            failure_modes=[],
            dependencies=[],
            assumptions=[],
            boundaries=[],
            history=[],
            success_shape=[],
            acceptance_criteria=[]
        ),
        prompt_md="",
        constraints_yaml="",
        created_at="2024-01-01T12:00:00Z",
        updated_at="2024-01-01T12:00:00Z"
    )


# ============================================================================
# Happy Path Tests - apply_update
# ============================================================================

def test_apply_update_happy_path_string_replacement(sample_problem_model):
    """Test apply_update replaces string fields with new values."""
    initial_description = sample_problem_model.system_description
    update = {"system_description": "Updated system description"}
    
    sample_problem_model.apply_update(update)
    
    assert sample_problem_model.system_description == "Updated system description"
    assert sample_problem_model.system_description != initial_description


def test_apply_update_happy_path_list_append(sample_problem_model):
    """Test apply_update appends to list fields correctly."""
    initial_stakeholders = sample_problem_model.stakeholders.copy()
    update = {"stakeholders": ["new_stakeholder"]}
    
    sample_problem_model.apply_update(update)
    
    assert "new_stakeholder" in sample_problem_model.stakeholders
    assert all(s in sample_problem_model.stakeholders for s in initial_stakeholders)
    assert len(sample_problem_model.stakeholders) == len(initial_stakeholders) + 1


def test_apply_update_happy_path_dict_list_append(sample_problem_model):
    """Test apply_update appends dict items to list fields."""
    initial_failure_modes = sample_problem_model.failure_modes.copy()
    new_failure = {"mode": "timeout", "impact": "medium"}
    update = {"failure_modes": [new_failure]}
    
    sample_problem_model.apply_update(update)
    
    assert new_failure in sample_problem_model.failure_modes
    assert len(sample_problem_model.failure_modes) == len(initial_failure_modes) + 1


def test_apply_update_mutation_in_place(sample_problem_model):
    """Test apply_update mutates the instance in place."""
    original_id = id(sample_problem_model)
    update = {"system_description": "New description"}
    
    result = sample_problem_model.apply_update(update)
    
    assert result is None  # apply_update returns None
    assert id(sample_problem_model) == original_id  # Same object
    assert sample_problem_model.system_description == "New description"


# ============================================================================
# Edge Case Tests - apply_update
# ============================================================================

def test_apply_update_edge_case_duplicate_strings(sample_problem_model):
    """Test apply_update does not append duplicate strings to list fields."""
    initial_count = len(sample_problem_model.stakeholders)
    existing_stakeholder = sample_problem_model.stakeholders[0]
    update = {"stakeholders": [existing_stakeholder]}
    
    sample_problem_model.apply_update(update)
    
    assert len(sample_problem_model.stakeholders) == initial_count
    assert sample_problem_model.stakeholders.count(existing_stakeholder) == 1


def test_apply_update_edge_case_unknown_fields_ignored(sample_problem_model):
    """Test apply_update ignores keys not in model_fields."""
    update = {
        "unknown_field": "unknown_value",
        "another_unknown": ["list", "of", "values"],
        "system_description": "Valid update"
    }
    
    # Should not raise an error
    sample_problem_model.apply_update(update)
    
    assert sample_problem_model.system_description == "Valid update"
    assert not hasattr(sample_problem_model, "unknown_field")
    assert not hasattr(sample_problem_model, "another_unknown")


def test_apply_update_edge_case_empty_update(sample_problem_model):
    """Test apply_update with empty dictionary."""
    original_description = sample_problem_model.system_description
    original_stakeholders = sample_problem_model.stakeholders.copy()
    update = {}
    
    sample_problem_model.apply_update(update)
    
    assert sample_problem_model.system_description == original_description
    assert sample_problem_model.stakeholders == original_stakeholders


def test_apply_update_edge_case_partial_update(sample_problem_model):
    """Test apply_update with only some fields updated."""
    original_description = sample_problem_model.system_description
    original_dependencies = sample_problem_model.dependencies.copy()
    update = {"stakeholders": ["new_stakeholder"]}
    
    sample_problem_model.apply_update(update)
    
    assert sample_problem_model.system_description == original_description
    assert sample_problem_model.dependencies == original_dependencies
    assert "new_stakeholder" in sample_problem_model.stakeholders


# ============================================================================
# Error Case Tests - apply_update
# ============================================================================

def test_apply_update_precondition_dict_required(sample_problem_model):
    """Test apply_update precondition that update must be a dict."""
    with pytest.raises((TypeError, AttributeError)):
        sample_problem_model.apply_update("not a dict")
    
    with pytest.raises((TypeError, AttributeError)):
        sample_problem_model.apply_update(["list", "of", "items"])
    
    with pytest.raises((TypeError, AttributeError)):
        sample_problem_model.apply_update(None)


# ============================================================================
# Invariant Tests - apply_update
# ============================================================================

def test_invariant_no_duplicate_strings_in_lists(sample_problem_model):
    """Test that ProblemModel list fields contain no duplicate strings after apply_update."""
    # Apply multiple updates with duplicate strings
    sample_problem_model.apply_update({"stakeholders": ["stakeholder1", "stakeholder2"]})
    sample_problem_model.apply_update({"stakeholders": ["stakeholder1"]})  # Duplicate
    sample_problem_model.apply_update({"stakeholders": ["stakeholder2"]})  # Duplicate
    sample_problem_model.apply_update({"stakeholders": ["stakeholder3"]})
    
    # Check no duplicates
    assert len(sample_problem_model.stakeholders) == len(set(sample_problem_model.stakeholders))
    # Verify all unique values are present
    expected = {"user", "admin", "stakeholder1", "stakeholder2", "stakeholder3"}
    assert set(sample_problem_model.stakeholders) == expected


# ============================================================================
# Happy Path Tests - touch
# ============================================================================

def test_touch_happy_path(sample_session):
    """Test touch updates updated_at timestamp."""
    original_updated_at = sample_session.updated_at
    
    # Mock datetime to control timestamp
    fixed_time = datetime(2024, 6, 15, 10, 30, 45, tzinfo=timezone.utc)
    with patch('constrain.models.datetime') as mock_datetime:
        mock_datetime.now.return_value = fixed_time
        mock_datetime.timezone = timezone
        
        sample_session.touch()
    
    # The updated_at should be different from original
    # In real implementation, it should be the current UTC time
    assert sample_session.updated_at != original_updated_at


def test_touch_mutation_in_place(sample_session):
    """Test touch mutates the instance in place."""
    original_id = id(sample_session)
    original_updated_at = sample_session.updated_at
    
    result = sample_session.touch()
    
    assert result is None  # touch returns None
    assert id(sample_session) == original_id  # Same object
    assert sample_session.updated_at != original_updated_at


def test_touch_timestamp_format_iso8601(sample_session):
    """Test touch sets timestamp in ISO 8601 format."""
    sample_session.touch()
    
    # Validate ISO 8601 format - should be parseable by datetime.fromisoformat
    # or match ISO 8601 pattern
    iso8601_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?$'
    assert re.match(iso8601_pattern, sample_session.updated_at)
    
    # Should be parseable
    try:
        # Try parsing with fromisoformat (handles most ISO formats)
        parsed = datetime.fromisoformat(sample_session.updated_at.replace('Z', '+00:00'))
        assert parsed is not None
    except ValueError:
        pytest.fail(f"updated_at '{sample_session.updated_at}' is not valid ISO 8601 format")


# ============================================================================
# Struct Creation Tests
# ============================================================================

def test_message_creation():
    """Test Message struct creation with all fields."""
    msg = Message(
        role="assistant",
        content="Hello, world!",
        timestamp="2024-01-01T12:00:00Z"
    )
    
    assert msg.role == "assistant"
    assert msg.content == "Hello, world!"
    assert msg.timestamp == "2024-01-01T12:00:00Z"


def test_problem_model_creation():
    """Test ProblemModel struct creation with all fields."""
    pm = ProblemModel(
        system_description="A complex system",
        stakeholders=["user", "admin", "developer"],
        failure_modes=[{"type": "crash"}, {"type": "hang"}],
        dependencies=["db", "cache"],
        assumptions=["assumption1"],
        boundaries=["boundary1"],
        history=["v1.0"],
        success_shape=["metric1"],
        acceptance_criteria=["criteria1"]
    )
    
    assert pm.system_description == "A complex system"
    assert len(pm.stakeholders) == 3
    assert len(pm.failure_modes) == 2
    assert pm.dependencies == ["db", "cache"]


def test_constraint_creation():
    """Test Constraint struct creation with all fields."""
    c = Constraint(
        id="constraint-1",
        boundary="Performance",
        condition="Response time < 200ms",
        violation="Response time exceeds 200ms",
        severity=Severity.must,
        rationale="User experience requirement"
    )
    
    assert c.id == "constraint-1"
    assert c.boundary == "Performance"
    assert c.severity == Severity.must


def test_session_creation():
    """Test Session struct creation with all fields."""
    session_id = str(uuid.uuid4())
    s = Session(
        id=session_id,
        schema_version=1,
        posture=Posture.adversarial,
        phase=Phase.challenge,
        round=3,
        understand_rounds=2,
        challenge_rounds=4,
        conversation=[],
        problem_model=ProblemModel(
            system_description="",
            stakeholders=[],
            failure_modes=[],
            dependencies=[],
            assumptions=[],
            boundaries=[],
            history=[],
            success_shape=[],
            acceptance_criteria=[]
        ),
        prompt_md="# Test Prompt",
        constraints_yaml="constraints: []",
        created_at="2024-01-01T12:00:00Z",
        updated_at="2024-01-01T12:30:00Z"
    )
    
    assert s.id == session_id
    assert s.schema_version == 1
    assert s.posture == Posture.adversarial
    assert s.phase == Phase.challenge
    assert s.round == 3


# ============================================================================
# Enum Tests
# ============================================================================

def test_enum_posture_all_variants():
    """Test all Posture enum variants are accessible."""
    assert Posture.adversarial is not None
    assert Posture.contrarian is not None
    assert Posture.critic is not None
    assert Posture.skeptic is not None
    assert Posture.collaborator is not None
    
    # Verify we can create instances with these values
    assert Posture.adversarial == Posture.adversarial
    
    # Count variants
    posture_values = [p for p in Posture]
    assert len(posture_values) == 5


def test_enum_phase_all_variants():
    """Test all Phase enum variants are accessible."""
    assert Phase.understand is not None
    assert Phase.challenge is not None
    assert Phase.synthesize is not None
    assert Phase.complete is not None
    
    # Count variants
    phase_values = [p for p in Phase]
    assert len(phase_values) == 4


def test_enum_severity_all_variants():
    """Test all Severity enum variants are accessible."""
    assert Severity.must is not None
    assert Severity.should is not None
    assert Severity.may is not None

    # Count variants
    severity_values = [s for s in Severity]
    assert len(severity_values) == 3


# ============================================================================
# Invariant Tests
# ============================================================================

def test_invariant_message_timestamps_iso8601(sample_message):
    """Test that all Message timestamps are ISO 8601 formatted."""
    iso8601_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?$'
    assert re.match(iso8601_pattern, sample_message.timestamp)
    
    # Should be parseable
    try:
        parsed = datetime.fromisoformat(sample_message.timestamp.replace('Z', '+00:00'))
        assert parsed is not None
    except ValueError:
        pytest.fail(f"Message timestamp '{sample_message.timestamp}' is not valid ISO 8601 format")


def test_invariant_session_schema_version_always_1(sample_session):
    """Test that Session.schema_version is always 1."""
    assert sample_session.schema_version == 1
    
    # Create multiple sessions and verify
    for _ in range(5):
        s = Session(
            id=str(uuid.uuid4()),
            schema_version=1,
            posture=Posture.collaborator,
            phase=Phase.understand,
            round=1,
            understand_rounds=2,
            challenge_rounds=2,
            conversation=[],
            problem_model=ProblemModel(
                system_description="",
                stakeholders=[],
                failure_modes=[],
                dependencies=[],
                assumptions=[],
                boundaries=[],
                history=[],
                success_shape=[],
                acceptance_criteria=[]
            ),
            prompt_md="",
            constraints_yaml="",
            created_at="2024-01-01T12:00:00Z",
            updated_at="2024-01-01T12:00:00Z"
        )
        assert s.schema_version == 1


def test_invariant_session_id_is_uuid4(sample_session):
    """Test that Session.id is a valid UUID4 string."""
    # Should be parseable as UUID
    try:
        parsed_uuid = uuid.UUID(sample_session.id)
        assert parsed_uuid.version == 4
    except ValueError:
        pytest.fail(f"Session.id '{sample_session.id}' is not a valid UUID4")
    
    # Create new sessions and verify UUID4 format
    for _ in range(3):
        session_id = str(uuid.uuid4())
        s = Session(
            id=session_id,
            schema_version=1,
            posture=Posture.collaborator,
            phase=Phase.understand,
            round=1,
            understand_rounds=2,
            challenge_rounds=2,
            conversation=[],
            problem_model=ProblemModel(
                system_description="",
                stakeholders=[],
                failure_modes=[],
                dependencies=[],
                assumptions=[],
                boundaries=[],
                history=[],
                success_shape=[],
                acceptance_criteria=[]
            ),
            prompt_md="",
            constraints_yaml="",
            created_at="2024-01-01T12:00:00Z",
            updated_at="2024-01-01T12:00:00Z"
        )
        parsed = uuid.UUID(s.id)
        assert parsed.version == 4


# ============================================================================
# Additional Edge Cases
# ============================================================================

def test_apply_update_multiple_list_fields_simultaneously(sample_problem_model):
    """Test apply_update can update multiple list fields at once."""
    update = {
        "stakeholders": ["new_stakeholder"],
        "dependencies": ["new_dependency"],
        "assumptions": ["new_assumption"]
    }
    
    sample_problem_model.apply_update(update)
    
    assert "new_stakeholder" in sample_problem_model.stakeholders
    assert "new_dependency" in sample_problem_model.dependencies
    assert "new_assumption" in sample_problem_model.assumptions


def test_apply_update_dict_always_appended(sample_problem_model):
    """Test that dicts are always appended even if they appear to be duplicates."""
    duplicate_failure = {"mode": "crash", "impact": "high"}
    initial_count = len(sample_problem_model.failure_modes)
    
    # Apply the same dict twice
    sample_problem_model.apply_update({"failure_modes": [duplicate_failure]})
    sample_problem_model.apply_update({"failure_modes": [duplicate_failure]})
    
    # Dicts should be appended each time
    assert len(sample_problem_model.failure_modes) == initial_count + 2


def test_session_conversation_list_handling(sample_session):
    """Test that conversation list can be accessed and modified."""
    msg1 = Message(role="user", content="Hello", timestamp="2024-01-01T12:00:00Z")
    msg2 = Message(role="assistant", content="Hi", timestamp="2024-01-01T12:01:00Z")
    
    sample_session.conversation.append(msg1)
    sample_session.conversation.append(msg2)
    
    assert len(sample_session.conversation) == 2
    assert sample_session.conversation[0].role == "user"
    assert sample_session.conversation[1].role == "assistant"


def test_constraint_with_different_severities():
    """Test creating constraints with different severity levels."""
    c1 = Constraint(
        id="c1",
        boundary="b1",
        condition="cond1",
        violation="viol1",
        severity=Severity.must,
        rationale="critical"
    )
    
    c2 = Constraint(
        id="c2",
        boundary="b2",
        condition="cond2",
        violation="viol2",
        severity=Severity.should,
        rationale="recommended"
    )
    
    assert c1.severity == Severity.must
    assert c2.severity == Severity.should
    assert c1.severity != c2.severity
