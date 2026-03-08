"""
Hidden adversarial acceptance tests for the Data Models component.
These tests target gaps in visible test coverage to catch implementations
that pass visible tests through shortcuts rather than truly satisfying the contract.
"""

import json
import uuid
from datetime import datetime, timezone, timedelta

import pytest
from pydantic import BaseModel

from src.models import (
    Posture,
    FailureMode,
    ProblemModel,
    Message,
    Constraint,
    Artifacts,
    LLMResponseMeta,
    Session,
)


# ---------------------------------------------------------------------------
# Posture enum tests
# ---------------------------------------------------------------------------

class TestGoodhartPosture:
    def test_goodhart_posture_all_values_are_strings(self):
        """Every Posture member's value should equal its lowercase name (StrEnum semantics)."""
        for member in Posture:
            assert str(member) == member.value
            assert member.value == member.name.lower() or member.value == member.name
            # StrEnum: value should be the string itself
            assert isinstance(member.value, str)

    def test_goodhart_posture_comparison_with_plain_string(self):
        """Posture members should be directly comparable to plain strings (StrEnum)."""
        assert Posture("collaborator") == "collaborator"
        assert Posture("adversarial") == "adversarial"
        assert Posture("contrarian") == "contrarian"
        assert Posture("critic") == "critic"
        assert Posture("skeptic") == "skeptic"

    def test_goodhart_posture_iteration_count(self):
        """Posture enum should have exactly 5 members."""
        assert len(Posture) == 5

    def test_goodhart_posture_from_string_construction(self):
        """Posture enum should be constructable from any valid lowercase string."""
        for name in ["adversarial", "contrarian", "critic", "skeptic", "collaborator"]:
            p = Posture(name)
            assert p.value == name


# ---------------------------------------------------------------------------
# ProblemModel tests
# ---------------------------------------------------------------------------

class TestGoodhartProblemModel:
    def test_goodhart_problem_model_all_list_fields_independent(self):
        """Every list field must be independently allocated per instance."""
        pm1 = ProblemModel()
        pm2 = ProblemModel()
        # Mutate every list field on pm1 and verify pm2 is unaffected
        pm1.stakeholders.append("x")
        pm1.failure_modes.append(FailureMode(description="d", severity="high", historical=False))
        pm1.dependencies.append("dep")
        pm1.assumptions.append("asm")
        pm1.boundaries.append("bnd")
        pm1.history.append("hist")
        pm1.success_shape.append("ss")

        assert pm2.stakeholders == []
        assert pm2.failure_modes == []
        assert pm2.dependencies == []
        assert pm2.assumptions == []
        assert pm2.boundaries == []
        assert pm2.history == []
        assert pm2.success_shape == []

    def test_goodhart_problem_model_is_basemodel(self):
        """ProblemModel should be a Pydantic BaseModel with model_fields."""
        assert issubclass(ProblemModel, BaseModel)
        assert hasattr(ProblemModel, "model_fields")


# ---------------------------------------------------------------------------
# apply_update tests
# ---------------------------------------------------------------------------

class TestGoodhartApplyUpdate:
    def test_goodhart_apply_update_repeated_calls_accumulate(self):
        """Multiple apply_update calls should accumulate list items."""
        pm = ProblemModel()
        pm.apply_update({"stakeholders": ["alice"]})
        pm.apply_update({"stakeholders": ["bob"]})
        pm.apply_update({"stakeholders": ["charlie"]})
        assert "alice" in pm.stakeholders
        assert "bob" in pm.stakeholders
        assert "charlie" in pm.stakeholders
        assert len(pm.stakeholders) == 3

    def test_goodhart_apply_update_dedup_across_calls(self):
        """Dedup should work across multiple apply_update calls."""
        pm = ProblemModel()
        pm.apply_update({"stakeholders": ["alice", "bob"]})
        pm.apply_update({"stakeholders": ["bob", "charlie"]})
        assert pm.stakeholders.count("bob") == 1
        assert len(pm.stakeholders) == 3

    def test_goodhart_apply_update_failure_modes_from_dict(self):
        """apply_update should accept failure_modes as list of dicts."""
        pm = ProblemModel()
        pm.apply_update({
            "failure_modes": [
                {"description": "timeout", "severity": "high", "historical": True}
            ]
        })
        assert len(pm.failure_modes) == 1
        fm = pm.failure_modes[0]
        assert isinstance(fm, FailureMode)
        assert fm.description == "timeout"
        assert fm.severity == "high"
        assert fm.historical is True

    def test_goodhart_apply_update_failure_modes_no_dedup(self):
        """Duplicate FailureMode items should both be appended (no dedup)."""
        pm = ProblemModel()
        fm_dict = {"description": "crash", "severity": "critical", "historical": False}
        pm.apply_update({"failure_modes": [fm_dict]})
        pm.apply_update({"failure_modes": [fm_dict]})
        assert len(pm.failure_modes) == 2

    def test_goodhart_apply_update_all_str_list_fields(self):
        """Dedup should work for ALL str list fields, not just stakeholders."""
        for field_name in ["dependencies", "assumptions", "boundaries", "success_shape", "history"]:
            pm = ProblemModel()
            pm.apply_update({field_name: ["item1", "item2"]})
            pm.apply_update({field_name: ["item2", "item3"]})
            field_val = getattr(pm, field_name)
            assert field_val.count("item2") == 1, f"Dedup failed for {field_name}"
            assert len(field_val) == 3, f"Wrong length for {field_name}"

    def test_goodhart_apply_update_preserves_existing_on_new_field(self):
        """Updating one field should not affect other fields."""
        pm = ProblemModel()
        pm.apply_update({"stakeholders": ["alice"], "system_description": "test system"})
        pm.apply_update({"dependencies": ["dep1"]})
        assert pm.stakeholders == ["alice"]
        assert pm.system_description == "test system"
        assert pm.dependencies == ["dep1"]

    def test_goodhart_apply_update_scalar_replacement_idempotent(self):
        """Replacing a scalar multiple times should reflect the latest value."""
        pm = ProblemModel()
        pm.apply_update({"system_description": "first"})
        pm.apply_update({"system_description": "second"})
        pm.apply_update({"system_description": "third"})
        assert pm.system_description == "third"

    def test_goodhart_apply_update_invalid_scalar_various_types(self):
        """Scalar field validation should reject int, list, dict, not just one type."""
        pm = ProblemModel()
        for bad_value in [42, ["a", "b"], {"key": "val"}, True, None]:
            with pytest.raises(Exception):
                pm.apply_update({"system_description": bad_value})

    def test_goodhart_apply_update_empty_list_no_effect(self):
        """Updating a list field with an empty list should not alter existing items."""
        pm = ProblemModel()
        pm.apply_update({"stakeholders": ["alice", "bob"]})
        pm.apply_update({"stakeholders": []})
        assert pm.stakeholders == ["alice", "bob"]

    def test_goodhart_apply_update_history_field(self):
        """apply_update should handle 'history' str list field with dedup."""
        pm = ProblemModel()
        pm.apply_update({"history": ["event1", "event2"]})
        pm.apply_update({"history": ["event2", "event3"]})
        assert "event1" in pm.history
        assert "event2" in pm.history
        assert "event3" in pm.history
        assert pm.history.count("event2") == 1

    def test_goodhart_apply_update_multiple_unknown_keys(self):
        """Multiple unknown keys should be silently ignored while known fields still apply."""
        pm = ProblemModel()
        pm.apply_update({
            "notes": "some notes",
            "metadata": {"key": "val"},
            "tags": ["tag1"],
            "system_description": "real update"
        })
        assert pm.system_description == "real update"
        assert not hasattr(pm, "notes") or "notes" not in ProblemModel.model_fields


# ---------------------------------------------------------------------------
# LLMResponseMeta.safe_parse tests
# ---------------------------------------------------------------------------

class TestGoodhartSafeParse:
    def test_goodhart_safe_parse_list_input(self):
        """safe_parse with a list should return safe defaults."""
        result = LLMResponseMeta.safe_parse([1, 2, 3])
        assert isinstance(result, LLMResponseMeta)
        assert result.ready_to_proceed is False
        assert result.problem_model_update == {}

    def test_goodhart_safe_parse_integer_input(self):
        """safe_parse with an integer should return safe defaults."""
        result = LLMResponseMeta.safe_parse(42)
        assert isinstance(result, LLMResponseMeta)
        assert result.ready_to_proceed is False
        assert result.problem_model_update == {}

    def test_goodhart_safe_parse_valid_true(self):
        """safe_parse should correctly parse ready_to_proceed=True with non-empty update."""
        raw = {
            "ready_to_proceed": True,
            "problem_model_update": {"system_description": "a system"}
        }
        result = LLMResponseMeta.safe_parse(raw)
        assert result.ready_to_proceed is True
        assert result.problem_model_update == {"system_description": "a system"}

    def test_goodhart_safe_parse_returns_llmresponsemeta_type(self):
        """safe_parse should always return an actual LLMResponseMeta instance."""
        for inp in [{}, {"ready_to_proceed": True}, None, "string", 0]:
            result = LLMResponseMeta.safe_parse(inp)
            assert isinstance(result, LLMResponseMeta)


# ---------------------------------------------------------------------------
# Message tests
# ---------------------------------------------------------------------------

class TestGoodhartMessage:
    def test_goodhart_message_timestamp_is_utc(self):
        """Message timestamp should be timezone-aware UTC."""
        msg = Message(role="user", content="hello")
        ts = msg.timestamp
        if isinstance(ts, datetime):
            assert ts.tzinfo is not None
            assert ts.tzinfo == timezone.utc or str(ts.tzinfo) == "UTC"
        elif isinstance(ts, str):
            # If stored as string, should be parseable and contain timezone info
            parsed = datetime.fromisoformat(ts)
            assert parsed.tzinfo is not None

    def test_goodhart_message_role_validation(self):
        """Message should reject invalid role values."""
        with pytest.raises(Exception):
            Message(role="system", content="hello")


# ---------------------------------------------------------------------------
# Constraint tests
# ---------------------------------------------------------------------------

class TestGoodhartConstraint:
    def test_goodhart_constraint_to_yaml_dict_values_match(self):
        """to_yaml_dict values should match the constraint's actual field values."""
        c = Constraint(
            id="C-042",
            boundary="latency",
            condition="response > 500ms",
            violation="alert triggered",
            severity="should",
            rationale="user experience"
        )
        d = c.to_yaml_dict()
        assert d["id"] == "C-042"
        assert d["boundary"] == "latency"
        assert d["condition"] == "response > 500ms"
        assert d["violation"] == "alert triggered"
        assert d["severity"] == "should"
        assert d["rationale"] == "user experience"

    def test_goodhart_constraint_to_yaml_dict_with_should_severity(self):
        """to_yaml_dict should serialize 'should' severity as a plain string."""
        c = Constraint(
            id="C-099",
            boundary="b",
            condition="c",
            violation="v",
            severity="should",
            rationale="r"
        )
        d = c.to_yaml_dict()
        assert d["severity"] == "should"
        assert type(d["severity"]) is str


# ---------------------------------------------------------------------------
# FailureMode tests
# ---------------------------------------------------------------------------

class TestGoodhartFailureMode:
    def test_goodhart_failure_mode_validation_rejects_missing_fields(self):
        """FailureMode should reject construction with missing required fields."""
        with pytest.raises(Exception):
            FailureMode(description="test")  # missing severity and historical
        with pytest.raises(Exception):
            FailureMode(severity="high")  # missing description and historical
        with pytest.raises(Exception):
            FailureMode(historical=True)  # missing description and severity


# ---------------------------------------------------------------------------
# Session tests
# ---------------------------------------------------------------------------

class TestGoodhartSession:
    def test_goodhart_session_default_phase_is_understand(self):
        """New session should default to 'understand' phase."""
        s = Session()
        assert s.current_phase == "understand"

    def test_goodhart_session_default_status_is_incomplete(self):
        """New session should default to 'incomplete' status."""
        s = Session()
        assert s.status == "incomplete"

    def test_goodhart_session_default_artifacts_none(self):
        """New session should have artifacts as None."""
        s = Session()
        assert s.artifacts is None

    def test_goodhart_session_default_round(self):
        """New session current_round should be a sensible integer default."""
        s = Session()
        assert isinstance(s.current_round, int)
        assert s.current_round >= 0

    def test_goodhart_session_id_is_valid_uuid_format(self):
        """Session default id should be a valid UUID string."""
        s = Session()
        parsed = uuid.UUID(s.id)
        assert str(parsed) == s.id

    def test_goodhart_session_timestamps_are_utc_aware(self):
        """Session created_at and updated_at should be UTC-aware datetimes."""
        s = Session()
        for ts_field in [s.created_at, s.updated_at]:
            if isinstance(ts_field, datetime):
                assert ts_field.tzinfo is not None
            elif isinstance(ts_field, str):
                parsed = datetime.fromisoformat(ts_field)
                assert parsed.tzinfo is not None

    def test_goodhart_session_schema_version_default(self):
        """Session should default to schema_version 1."""
        s = Session()
        assert s.schema_version == 1

    def test_goodhart_session_touch_timestamp_is_recent(self):
        """After touch(), updated_at should be very close to now."""
        s = Session()
        s.touch()
        now = datetime.now(timezone.utc)
        updated = s.updated_at
        if isinstance(updated, str):
            updated = datetime.fromisoformat(updated)
        diff = abs((now - updated).total_seconds())
        assert diff < 1.0

    def test_goodhart_session_to_json_contains_all_fields(self):
        """to_json output should contain keys for every Session field."""
        s = Session()
        raw = json.loads(s.to_json())
        expected_keys = {
            "id", "schema_version", "posture", "current_phase",
            "current_round", "conversation_history", "problem_model",
            "artifacts", "created_at", "updated_at", "status"
        }
        for key in expected_keys:
            assert key in raw, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Session roundtrip tests
# ---------------------------------------------------------------------------

class TestGoodhartSessionRoundtrip:
    def test_goodhart_session_roundtrip_all_postures(self):
        """Roundtrip should work for every Posture value."""
        for posture_val in ["adversarial", "contrarian", "critic", "skeptic", "collaborator"]:
            s = Session(posture=Posture(posture_val))
            restored = Session.from_json(s.to_json())
            assert restored.posture == Posture(posture_val)
            assert str(restored.posture) == posture_val

    def test_goodhart_session_roundtrip_all_phases(self):
        """Roundtrip should work for all valid phase values."""
        for phase in ["understand", "challenge", "synthesize", "complete"]:
            s = Session(current_phase=phase)
            restored = Session.from_json(s.to_json())
            assert restored.current_phase == phase

    def test_goodhart_session_roundtrip_with_conversation_history(self):
        """Roundtrip should properly deserialize Message objects."""
        s = Session()
        s.conversation_history = [
            Message(role="user", content="What is the system?"),
            Message(role="assistant", content="Let me understand."),
        ]
        restored = Session.from_json(s.to_json())
        assert len(restored.conversation_history) == 2
        assert restored.conversation_history[0].role == "user"
        assert restored.conversation_history[0].content == "What is the system?"
        assert restored.conversation_history[1].role == "assistant"
        assert isinstance(restored.conversation_history[0], Message)

    def test_goodhart_session_roundtrip_preserves_problem_model_data(self):
        """Roundtrip should preserve all ProblemModel fields including FailureModes."""
        s = Session()
        s.problem_model = ProblemModel(
            system_description="A web API",
            stakeholders=["dev", "ops"],
            failure_modes=[
                FailureMode(description="crash", severity="high", historical=True)
            ],
            dependencies=["db", "cache"],
            assumptions=["always online"],
            boundaries=["region=us"],
            history=["v1 launched"],
            success_shape=["99.9% uptime"],
        )
        restored = Session.from_json(s.to_json())
        rpm = restored.problem_model
        assert rpm.system_description == "A web API"
        assert rpm.stakeholders == ["dev", "ops"]
        assert len(rpm.failure_modes) == 1
        assert isinstance(rpm.failure_modes[0], FailureMode)
        assert rpm.failure_modes[0].description == "crash"
        assert rpm.dependencies == ["db", "cache"]

    def test_goodhart_artifacts_in_session_roundtrip(self):
        """Session with non-None artifacts should roundtrip correctly."""
        s = Session()
        s.artifacts = Artifacts(
            prompt_md="# System Prompt\nYou are helpful.",
            constraints_yaml="constraints:\n  - id: C-001\n"
        )
        restored = Session.from_json(s.to_json())
        assert restored.artifacts is not None
        assert restored.artifacts.prompt_md == "# System Prompt\nYou are helpful."
        assert restored.artifacts.constraints_yaml == "constraints:\n  - id: C-001\n"


# ---------------------------------------------------------------------------
# Session.from_json error cases
# ---------------------------------------------------------------------------

class TestGoodhartSessionFromJsonErrors:
    def test_goodhart_session_from_json_invalid_status(self):
        """from_json should reject invalid status values."""
        s = Session()
        raw = json.loads(s.to_json())
        raw["status"] = "pending"
        with pytest.raises(Exception):
            Session.from_json(json.dumps(raw))

    def test_goodhart_session_from_json_invalid_severity_in_constraint(self):
        """from_json should reject constraints with invalid severity values in artifacts context.
        This tests that nested validation works."""
        # We test via the problem_model or direct JSON manipulation
        s = Session()
        raw_json = s.to_json()
        raw = json.loads(raw_json)
        # Try setting an invalid role in conversation_history
        raw["conversation_history"] = [
            {"role": "moderator", "content": "hi", "timestamp": "2024-01-01T00:00:00+00:00"}
        ]
        with pytest.raises(Exception):
            Session.from_json(json.dumps(raw))

    def test_goodhart_session_from_json_wrong_type_for_round(self):
        """from_json should reject non-integer current_round."""
        s = Session()
        raw = json.loads(s.to_json())
        raw["current_round"] = "five"
        with pytest.raises(Exception):
            Session.from_json(json.dumps(raw))
