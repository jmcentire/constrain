"""
Contract test suite for src_constrain_posture module.
Tests system prompt generation for phase and posture.
"""
import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from constrain.posture import (
    _format_problem_model,
    _understand_prompt,
    _challenge_prompt,
    _synthesize_prompt,
    _revision_prompt,
    get_system_prompt,
    get_revision_prompt,
    select_posture,
    Posture,
    Phase,
    POSTURE_DESCRIPTIONS,
    _JSON_INSTRUCTION
)
from constrain.models import ProblemModel


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def empty_problem_model():
    """ProblemModel with all empty fields."""
    return ProblemModel(
        system_description="",
        stakeholders=[],
        dependencies=[],
        assumptions=[],
        boundaries=[],
        history=[],
        success_shape=[],
        acceptance_criteria=[],
        failure_modes=[]
    )


@pytest.fixture
def partial_problem_model():
    """ProblemModel with some fields populated."""
    return ProblemModel(
        system_description="A web service for user authentication",
        stakeholders=["End users", "System administrators"],
        dependencies=[],
        assumptions=["Users have internet access"],
        boundaries=[],
        history=[],
        success_shape=[],
        acceptance_criteria=[],
        failure_modes=[]
    )


@pytest.fixture
def full_problem_model():
    """ProblemModel with all fields populated."""
    return ProblemModel(
        system_description="A web service for user authentication",
        stakeholders=["End users", "System administrators"],
        dependencies=["PostgreSQL database", "Redis cache"],
        assumptions=["Users have internet access", "Network is reliable"],
        boundaries=["Only handles authentication, not authorization"],
        history=["Previous system had security vulnerabilities"],
        success_shape=["99.9% uptime", "Sub-200ms response time"],
        acceptance_criteria=["All tests pass", "Security audit completed"],
        failure_modes=[
            {"description": "Database connection timeout"},
            {"description": "Cache invalidation issues"}
        ]
    )


@pytest.fixture
def problem_model_with_failure_modes():
    """ProblemModel with various failure mode formats."""
    return ProblemModel(
        system_description="",
        stakeholders=[],
        dependencies=[],
        assumptions=[],
        boundaries=[],
        history=[],
        success_shape=[],
        acceptance_criteria=[],
        failure_modes=[
            {"description": "Network timeout"},
            {"description": "Simple string failure mode"}
        ]
    )


# ============================================================================
# _format_problem_model TESTS
# ============================================================================

def test_format_problem_model_empty(empty_problem_model):
    """When all ProblemModel fields are empty, returns '(No information gathered yet)'"""
    result = _format_problem_model(empty_problem_model)
    assert result == "(No information gathered yet)"


def test_format_problem_model_with_data(partial_problem_model):
    """When ProblemModel has data in some fields, returns formatted multi-line string with headers and bullet points"""
    result = _format_problem_model(partial_problem_model)
    
    assert "System: A web service for user authentication" in result
    assert "Stakeholders:" in result
    assert "- End users" in result
    assert "- System administrators" in result
    assert "(No information gathered yet)" not in result


def test_format_problem_model_system_description_first(full_problem_model):
    """System description appears first if present in formatted output"""
    result = _format_problem_model(full_problem_model)
    
    # Find the position of System description
    sys_desc_pos = result.find("System:")
    
    # Check other sections appear after system description
    other_sections = ["Stakeholders:", "Dependencies:", "Assumptions:"]
    for section in other_sections:
        if section in result:
            section_pos = result.find(section)
            assert sys_desc_pos < section_pos, f"System Description should appear before {section}"


def test_format_problem_model_failure_modes(problem_model_with_failure_modes):
    """Failure modes are formatted by extracting 'description' key or converting to string"""
    result = _format_problem_model(problem_model_with_failure_modes)
    
    assert "Failure Modes:" in result
    assert "- Network timeout" in result or "Network timeout" in result
    assert "Simple string failure mode" in result


def test_format_problem_model_all_fields(full_problem_model):
    """Formats all ProblemModel fields correctly when all are populated"""
    result = _format_problem_model(full_problem_model)
    
    expected_sections = [
        "System:",
        "Stakeholders:",
        "Dependencies:",
        "Assumptions:",
        "Boundaries:",
        "History:",
        "Success Shape:",
        "Acceptance Criteria:",
        "Failure Modes:"
    ]
    
    for section in expected_sections:
        assert section in result, f"Missing section: {section}"


def test_format_problem_model_missing_attributes():
    """Raises AttributeError when model lacks expected attributes during getattr() calls"""
    # Create a mock object that doesn't have the expected attributes
    mock_model = Mock(spec=[])
    
    with pytest.raises(AttributeError):
        _format_problem_model(mock_model)


# ============================================================================
# _understand_prompt TESTS
# ============================================================================

def test_understand_prompt_happy_path(partial_problem_model):
    """Returns interviewer instructions with formatted problem model and JSON instruction"""
    result = _understand_prompt(partial_problem_model)
    
    assert isinstance(result, str)
    assert len(result) > 0
    assert "interviewer" in result.lower() or "understand" in result.lower()
    assert "Current understanding" in result or "current" in result.lower()


def test_understand_prompt_contains_json_instruction(partial_problem_model):
    """Output includes _JSON_INSTRUCTION template at the end"""
    result = _understand_prompt(partial_problem_model)
    
    assert _JSON_INSTRUCTION in result


# ============================================================================
# _challenge_prompt TESTS
# ============================================================================

def test_challenge_prompt_happy_path(partial_problem_model):
    """Returns challenge phase instructions with posture description and formatted problem model"""
    result = _challenge_prompt(partial_problem_model, Posture.adversarial)
    
    assert isinstance(result, str)
    assert len(result) > 0
    assert "challenge" in result.lower() or "adversarial" in result.lower()
    assert _JSON_INSTRUCTION in result


@pytest.mark.parametrize("posture", [
    Posture.adversarial,
    Posture.contrarian,
    Posture.critic,
    Posture.skeptic,
    Posture.collaborator
])
def test_challenge_prompt_all_postures(partial_problem_model, posture):
    """Successfully generates prompts for all valid posture values"""
    result = _challenge_prompt(partial_problem_model, posture)
    
    assert isinstance(result, str)
    assert len(result) > 0
    assert _JSON_INSTRUCTION in result
    
    # Verify posture description is included
    posture_desc = POSTURE_DESCRIPTIONS[posture]
    assert posture_desc in result


def test_challenge_prompt_invalid_posture(partial_problem_model):
    """Raises KeyError when posture is not found in POSTURE_DESCRIPTIONS"""
    # Create an invalid posture-like object
    invalid_posture = "invalid_posture_value"
    
    with pytest.raises(KeyError):
        _challenge_prompt(partial_problem_model, invalid_posture)


# ============================================================================
# _synthesize_prompt TESTS
# ============================================================================

def test_synthesize_prompt_happy_path(partial_problem_model):
    """Returns synthesis instructions with artifact format specifications and delimiters"""
    result = _synthesize_prompt(partial_problem_model)
    
    assert isinstance(result, str)
    assert len(result) > 0
    assert "synthesize" in result.lower() or "artifact" in result.lower()


def test_synthesize_prompt_delimiter_check(partial_problem_model):
    """Output includes exact delimiters for artifacts"""
    result = _synthesize_prompt(partial_problem_model)
    
    assert "--- PROMPT ---" in result
    assert "--- CONSTRAINTS ---" in result


# ============================================================================
# _revision_prompt TESTS
# ============================================================================

def test_revision_prompt_happy_path(partial_problem_model):
    """Returns revision instructions with feedback and problem model"""
    feedback = "Please add more detail to the acceptance criteria"
    result = _revision_prompt(feedback, partial_problem_model)
    
    assert isinstance(result, str)
    assert len(result) > 0
    assert "revision" in result.lower() or "revise" in result.lower() or "feedback" in result.lower()


def test_revision_prompt_includes_feedback(partial_problem_model):
    """Output includes the provided feedback text"""
    feedback = "This is my specific feedback text for testing"
    result = _revision_prompt(feedback, partial_problem_model)
    
    assert feedback in result


# ============================================================================
# get_system_prompt TESTS
# ============================================================================

def test_get_system_prompt_understand_phase(partial_problem_model):
    """Returns understand prompt when phase is Phase.understand"""
    result = get_system_prompt(Phase.understand, partial_problem_model, None)
    
    assert isinstance(result, str)
    assert len(result) > 0
    # Should match output of _understand_prompt
    expected = _understand_prompt(partial_problem_model)
    assert result == expected


def test_get_system_prompt_challenge_phase(partial_problem_model):
    """Returns challenge prompt when phase is Phase.challenge with valid posture"""
    result = get_system_prompt(Phase.challenge, partial_problem_model, Posture.adversarial)
    
    assert isinstance(result, str)
    assert len(result) > 0
    # Should match output of _challenge_prompt
    expected = _challenge_prompt(partial_problem_model, Posture.adversarial)
    assert result == expected


def test_get_system_prompt_synthesize_phase(partial_problem_model):
    """Returns synthesize prompt when phase is Phase.synthesize"""
    result = get_system_prompt(Phase.synthesize, partial_problem_model, None)
    
    assert isinstance(result, str)
    assert len(result) > 0
    # Should match output of _synthesize_prompt
    expected = _synthesize_prompt(partial_problem_model)
    assert result == expected


def test_get_system_prompt_challenge_missing_posture(partial_problem_model):
    """Raises ValueError when phase is challenge but posture is None"""
    with pytest.raises(ValueError):
        get_system_prompt(Phase.challenge, partial_problem_model, None)


def test_get_system_prompt_unknown_phase(partial_problem_model):
    """Raises ValueError when phase is not a valid Phase enum value"""
    # Create an invalid phase
    invalid_phase = "invalid_phase"
    
    with pytest.raises((ValueError, AttributeError)):
        get_system_prompt(invalid_phase, partial_problem_model, None)


# ============================================================================
# get_revision_prompt TESTS
# ============================================================================

def test_get_revision_prompt_happy_path(partial_problem_model):
    """Returns revision prompt identical to _revision_prompt output"""
    feedback = "Add more constraints"
    result = get_revision_prompt(feedback, partial_problem_model)
    
    assert isinstance(result, str)
    assert len(result) > 0
    
    # Should match _revision_prompt output
    expected = _revision_prompt(feedback, partial_problem_model)
    assert result == expected


# ============================================================================
# select_posture TESTS
# ============================================================================

def test_select_posture_with_override():
    """Returns override value when override parameter is not None"""
    override = Posture.adversarial
    result = select_posture(override)
    
    assert result == Posture.adversarial


def test_select_posture_from_env_var(monkeypatch):
    """Returns posture from CONSTRAIN_POSTURE environment variable when override is None and env var is set"""
    monkeypatch.setenv("CONSTRAIN_POSTURE", "critic")
    
    result = select_posture(None)
    
    assert result == Posture.critic


def test_select_posture_random_selection(monkeypatch):
    """Returns a randomly selected Posture when override is None and env var is not set"""
    monkeypatch.delenv("CONSTRAIN_POSTURE", raising=False)
    
    result = select_posture(None)
    
    # Should be one of the valid Posture values
    assert result in [Posture.adversarial, Posture.contrarian, Posture.critic, 
                      Posture.skeptic, Posture.collaborator]


def test_select_posture_invalid_env_var(monkeypatch):
    """Raises ValueError when CONSTRAIN_POSTURE contains invalid value"""
    monkeypatch.setenv("CONSTRAIN_POSTURE", "invalid_posture")
    
    with pytest.raises(ValueError):
        select_posture(None)


@pytest.mark.parametrize("posture_name,posture_value", [
    ("adversarial", Posture.adversarial),
    ("contrarian", Posture.contrarian),
    ("critic", Posture.critic),
    ("skeptic", Posture.skeptic),
    ("collaborator", Posture.collaborator),
])
def test_select_posture_all_env_values(monkeypatch, posture_name, posture_value):
    """Test that all valid posture names in environment variable work correctly"""
    monkeypatch.setenv("CONSTRAIN_POSTURE", posture_name)
    
    result = select_posture(None)
    
    assert result == posture_value


# ============================================================================
# INVARIANT TESTS
# ============================================================================

def test_invariant_posture_descriptions_count():
    """POSTURE_DESCRIPTIONS contains exactly 5 entries for each Posture enum value"""
    assert len(POSTURE_DESCRIPTIONS) == 5
    
    # Check all Posture values are present
    expected_postures = {Posture.adversarial, Posture.contrarian, Posture.critic, 
                        Posture.skeptic, Posture.collaborator}
    
    assert set(POSTURE_DESCRIPTIONS.keys()) == expected_postures
    
    # Check all values are non-empty strings
    for posture, description in POSTURE_DESCRIPTIONS.items():
        assert isinstance(description, str)
        assert len(description) > 0


def test_invariant_all_prompts_non_empty(partial_problem_model):
    """All prompt functions return non-empty strings"""
    # Test _understand_prompt
    result = _understand_prompt(partial_problem_model)
    assert len(result) > 0
    
    # Test _challenge_prompt
    result = _challenge_prompt(partial_problem_model, Posture.adversarial)
    assert len(result) > 0
    
    # Test _synthesize_prompt
    result = _synthesize_prompt(partial_problem_model)
    assert len(result) > 0
    
    # Test _revision_prompt
    result = _revision_prompt("feedback", partial_problem_model)
    assert len(result) > 0
    
    # Test get_system_prompt
    result = get_system_prompt(Phase.understand, partial_problem_model, None)
    assert len(result) > 0
    
    # Test get_revision_prompt
    result = get_revision_prompt("feedback", partial_problem_model)
    assert len(result) > 0


def test_invariant_formatted_empty_model_postcondition(empty_problem_model):
    """Formatted problem models return '(No information gathered yet)' when all fields are empty"""
    result = _format_problem_model(empty_problem_model)
    assert result == "(No information gathered yet)"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.parametrize("phase,posture", [
    (Phase.understand, None),
    (Phase.challenge, Posture.adversarial),
    (Phase.challenge, Posture.contrarian),
    (Phase.challenge, Posture.critic),
    (Phase.challenge, Posture.skeptic),
    (Phase.challenge, Posture.collaborator),
    (Phase.synthesize, None),
])
def test_get_system_prompt_all_valid_combinations(partial_problem_model, phase, posture):
    """Test get_system_prompt with all valid Phase/Posture combinations"""
    result = get_system_prompt(phase, partial_problem_model, posture)
    
    assert isinstance(result, str)
    assert len(result) > 0


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

def test_format_problem_model_mixed_empty_fields():
    """Test formatting with alternating empty and populated fields"""
    model = ProblemModel(
        system_description="Description here",
        stakeholders=[],
        dependencies=["Dependency 1"],
        assumptions=[],
        boundaries=["Boundary 1"],
        history=[],
        success_shape=[],
        acceptance_criteria=["Criterion 1"],
        failure_modes=[]
    )
    
    result = _format_problem_model(model)
    
    assert "Description here" in result
    assert "Dependency 1" in result
    assert "Boundary 1" in result
    assert "Criterion 1" in result
    assert "(No information gathered yet)" not in result


def test_revision_prompt_empty_feedback(partial_problem_model):
    """Test revision prompt with empty feedback string"""
    result = _revision_prompt("", partial_problem_model)
    
    assert isinstance(result, str)
    assert len(result) > 0


def test_revision_prompt_multiline_feedback(partial_problem_model):
    """Test revision prompt with multiline feedback"""
    feedback = """Line 1 of feedback
Line 2 of feedback
Line 3 of feedback"""
    
    result = _revision_prompt(feedback, partial_problem_model)
    
    assert feedback in result
    assert "Line 1 of feedback" in result
    assert "Line 2 of feedback" in result


def test_select_posture_priority_override_over_env(monkeypatch):
    """Test that override takes priority over environment variable"""
    monkeypatch.setenv("CONSTRAIN_POSTURE", "critic")
    
    result = select_posture(Posture.adversarial)
    
    # Should return override, not env var
    assert result == Posture.adversarial


def test_json_instruction_constant_exists():
    """Test that _JSON_INSTRUCTION constant exists and is non-empty"""
    assert _JSON_INSTRUCTION is not None
    assert isinstance(_JSON_INSTRUCTION, str)
    assert len(_JSON_INSTRUCTION) > 0
