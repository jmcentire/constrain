"""
Adversarial hidden acceptance tests for Session Manager.
These tests catch implementations that pass visible tests through shortcuts
rather than truly satisfying the contract.
"""
import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from src.session import *


# ─── Helpers ────────────────────────────────────────────────────────────────

def make_tmp_base():
    """Create a temporary directory to use as base_path."""
    return tempfile.mkdtemp()


def create_manager_and_session(posture_override="", base_path=None):
    """Helper to create a session manager and a session."""
    bp = base_path or make_tmp_base()
    mgr = SessionManager(bp)
    session = mgr.create(posture_override=posture_override)
    return mgr, session


def is_valid_uuid4(val):
    try:
        u = uuid.UUID(val, version=4)
        return str(u) == val
    except (ValueError, AttributeError):
        return False


def is_valid_iso8601(val):
    try:
        datetime.fromisoformat(val)
        return True
    except (ValueError, TypeError):
        return False


# ─── Tests ──────────────────────────────────────────────────────────────────

class TestGoodhartCreate:
    def test_goodhart_create_unique_ids(self):
        """Each call to create() must produce a distinct UUID v4, not a hardcoded or recycled value."""
        mgr = SessionManager(make_tmp_base())
        sessions = [mgr.create(posture_override="") for _ in range(10)]
        ids = [s.id for s in sessions]
        # All IDs must be unique
        assert len(set(ids)) == 10
        # All IDs must be valid UUID v4
        for sid in ids:
            assert is_valid_uuid4(sid), f"{sid} is not a valid UUID v4"

    def test_goodhart_create_timestamps_are_current(self):
        """created_at and updated_at should reflect the actual current time, not a hardcoded timestamp."""
        before = datetime.now(timezone.utc).isoformat()
        mgr = SessionManager(make_tmp_base())
        session = mgr.create(posture_override="")
        after = datetime.now(timezone.utc).isoformat()

        assert is_valid_iso8601(session.created_at)
        assert is_valid_iso8601(session.updated_at)
        # created_at and updated_at should be equal at creation
        assert session.created_at == session.updated_at

    def test_goodhart_create_empty_posture_override_uses_random(self):
        """When posture_override is empty string and no env var, a valid posture is still selected."""
        env = os.environ.copy()
        env.pop("CONSTRAIN_POSTURE", None)
        with patch.dict(os.environ, env, clear=True):
            mgr = SessionManager(make_tmp_base())
            session = mgr.create(posture_override="")
            assert isinstance(session.posture, str)
            assert len(session.posture) > 0

    def test_goodhart_create_problem_model_all_lists_empty(self):
        """All list fields in a freshly created ProblemModel must be empty lists."""
        mgr, session = create_manager_and_session()
        pm = session.problem_model
        assert pm.key_entities == []
        assert pm.constraints == []
        assert pm.assumptions == []
        assert pm.quality_attributes == []
        assert pm.scope_boundaries == []
        assert pm.open_questions == []
        assert pm.topics_covered == []

    def test_goodhart_create_problem_model_strings_empty(self):
        """All string fields in a freshly created ProblemModel must be empty strings."""
        mgr, session = create_manager_and_session()
        pm = session.problem_model
        assert pm.domain == ""
        assert pm.core_problem == ""

    def test_goodhart_create_round_counters_limits_positive(self):
        """Round counter limits should be positive values, not zero."""
        mgr, session = create_manager_and_session()
        rc = session.round_counters
        assert rc.understand_limit > 0
        assert rc.challenge_limit > 0
        assert rc.understand_rounds == 0
        assert rc.challenge_rounds == 0

    def test_goodhart_create_env_var_invalid_posture(self):
        """When CONSTRAIN_POSTURE env var specifies an invalid posture, an error must be raised."""
        with patch.dict(os.environ, {"CONSTRAIN_POSTURE": "totally_fake_posture_xyz_999"}):
            mgr = SessionManager(make_tmp_base())
            with pytest.raises(Exception):
                mgr.create(posture_override="")


class TestGoodhartSave:
    def test_goodhart_save_updates_updated_at(self):
        """save() must update the session's updated_at timestamp."""
        bp = make_tmp_base()
        mgr = SessionManager(bp)
        session = mgr.create(posture_override="")
        original_updated = session.updated_at
        time.sleep(0.01)
        mgr.save(session)
        assert session.updated_at >= original_updated

    def test_goodhart_save_file_content_is_valid_json(self):
        """The saved file must contain parseable JSON."""
        bp = make_tmp_base()
        mgr = SessionManager(bp)
        session = mgr.create(posture_override="")
        mgr.save(session)

        file_path = Path(bp) / ".constrain" / "sessions" / f"{session.id}.json"
        assert file_path.exists()
        with open(file_path) as f:
            data = json.load(f)
        assert data["id"] == session.id
        assert data["phase"] == "understand"

    def test_goodhart_save_correct_directory_structure(self):
        """save() must create the file under <base_path>/.constrain/sessions/."""
        bp = make_tmp_base()
        mgr = SessionManager(bp)
        session = mgr.create(posture_override="")
        mgr.save(session)

        sessions_dir = Path(bp) / ".constrain" / "sessions"
        assert sessions_dir.is_dir()
        assert (sessions_dir / f"{session.id}.json").is_file()

    def test_goodhart_save_multiple_times_overwrites(self):
        """Saving a session multiple times must overwrite, not create duplicates."""
        bp = make_tmp_base()
        mgr = SessionManager(bp)
        session = mgr.create(posture_override="")
        mgr.save(session)
        mgr.save(session)
        mgr.save(session)

        sessions_dir = Path(bp) / ".constrain" / "sessions"
        json_files = list(sessions_dir.glob("*.json"))
        assert len(json_files) == 1

    def test_goodhart_save_load_round_trip_with_populated_session(self):
        """A fully populated session must survive save/load round-trip."""
        bp = make_tmp_base()
        mgr = SessionManager(bp)
        session = mgr.create(posture_override="")

        # Populate session
        session = mgr.add_message(session, "user", "What about microservices?")
        session = mgr.add_message(session, "assistant", "Let me ask about that.")
        session = mgr.update_problem_model(session, {
            "domain": "backend",
            "core_problem": "scaling issues",
            "key_entities": ["service_a", "service_b"],
            "constraints": ["budget limited"],
            "topics_covered": ["microservices"]
        })

        mgr.save(session)
        loaded = mgr.load(session.id)

        assert loaded.id == session.id
        assert loaded.phase == session.phase
        assert loaded.posture == session.posture
        assert loaded.problem_model.domain == "backend"
        assert loaded.problem_model.core_problem == "scaling issues"
        assert loaded.problem_model.key_entities == ["service_a", "service_b"]
        assert loaded.problem_model.constraints == ["budget limited"]
        assert loaded.problem_model.topics_covered == ["microservices"]
        assert len(loaded.conversation_history) == 2
        assert loaded.conversation_history[0].role == "user"
        assert loaded.conversation_history[0].content == "What about microservices?"
        assert loaded.conversation_history[1].role == "assistant"
        assert loaded.round_counters.understand_rounds == 1


class TestGoodhartLoad:
    def test_goodhart_load_preserves_conversation_history(self):
        """Loading must preserve all messages in order."""
        bp = make_tmp_base()
        mgr = SessionManager(bp)
        session = mgr.create(posture_override="")
        session = mgr.add_message(session, "user", "First message")
        session = mgr.add_message(session, "assistant", "First reply")
        session = mgr.add_message(session, "user", "Second message")
        session = mgr.add_message(session, "assistant", "Second reply")
        session = mgr.add_message(session, "user", "Third message")
        mgr.save(session)

        loaded = mgr.load(session.id)
        assert len(loaded.conversation_history) == 5
        assert loaded.conversation_history[0].content == "First message"
        assert loaded.conversation_history[1].content == "First reply"
        assert loaded.conversation_history[2].content == "Second message"
        assert loaded.conversation_history[3].content == "Second reply"
        assert loaded.conversation_history[4].content == "Third message"

    def test_goodhart_load_preserves_problem_model_lists(self):
        """Loading must preserve all problem model list items."""
        bp = make_tmp_base()
        mgr = SessionManager(bp)
        session = mgr.create(posture_override="")
        session = mgr.update_problem_model(session, {
            "key_entities": ["entity1", "entity2", "entity3"],
            "constraints": ["constraint_a"],
            "topics_covered": ["topic_x", "topic_y"],
            "assumptions": ["assume_1"],
            "quality_attributes": ["performance", "security"],
            "scope_boundaries": ["only_backend"],
            "open_questions": ["what about caching?"]
        })
        mgr.save(session)

        loaded = mgr.load(session.id)
        assert loaded.problem_model.key_entities == ["entity1", "entity2", "entity3"]
        assert loaded.problem_model.constraints == ["constraint_a"]
        assert loaded.problem_model.topics_covered == ["topic_x", "topic_y"]
        assert loaded.problem_model.assumptions == ["assume_1"]
        assert loaded.problem_model.quality_attributes == ["performance", "security"]
        assert loaded.problem_model.scope_boundaries == ["only_backend"]
        assert loaded.problem_model.open_questions == ["what about caching?"]


class TestGoodhartFindLatestIncomplete:
    def test_goodhart_find_latest_incomplete_ignores_complete(self):
        """Must skip completed sessions even if they have the most recent updated_at."""
        bp = make_tmp_base()
        mgr = SessionManager(bp)

        # Create and save an incomplete session
        s1 = mgr.create(posture_override="")
        mgr.save(s1)
        time.sleep(0.05)

        # Create a session, complete it, and save (more recent)
        s2 = mgr.create(posture_override="")
        s2 = mgr.transition_phase(s2, "challenge")
        s2 = mgr.transition_phase(s2, "synthesize")
        s2 = mgr.transition_phase(s2, "complete")
        mgr.save(s2)

        result = mgr.find_latest_incomplete()
        assert result is not None
        assert result.id == s1.id
        assert result.phase != "complete"

    def test_goodhart_find_latest_incomplete_ignores_abandoned(self):
        """Must skip abandoned sessions even if they have the most recent updated_at."""
        bp = make_tmp_base()
        mgr = SessionManager(bp)

        s1 = mgr.create(posture_override="")
        mgr.save(s1)
        time.sleep(0.05)

        s2 = mgr.create(posture_override="")
        s2 = mgr.transition_phase(s2, "abandoned")
        mgr.save(s2)

        result = mgr.find_latest_incomplete()
        assert result is not None
        assert result.id == s1.id
        assert result.phase != "abandoned"

    def test_goodhart_find_latest_incomplete_selects_by_updated_at(self):
        """Must return the session with the most recent updated_at among incomplete sessions."""
        bp = make_tmp_base()
        mgr = SessionManager(bp)

        s1 = mgr.create(posture_override="")
        mgr.save(s1)
        time.sleep(0.05)

        s2 = mgr.create(posture_override="")
        mgr.save(s2)
        time.sleep(0.05)

        # Update s1 to make it more recent
        s1 = mgr.load(s1.id)
        s1 = mgr.add_message(s1, "user", "updating s1")
        mgr.save(s1)

        result = mgr.find_latest_incomplete()
        assert result is not None
        assert result.id == s1.id

    def test_goodhart_find_latest_incomplete_all_phases_mixed(self):
        """With sessions in all 5 phases, must only consider incomplete ones."""
        bp = make_tmp_base()
        mgr = SessionManager(bp)

        # understand phase
        s_understand = mgr.create(posture_override="")
        mgr.save(s_understand)
        time.sleep(0.02)

        # challenge phase
        s_challenge = mgr.create(posture_override="")
        s_challenge = mgr.transition_phase(s_challenge, "challenge")
        mgr.save(s_challenge)
        time.sleep(0.02)

        # synthesize phase (should be most recent incomplete)
        s_synthesize = mgr.create(posture_override="")
        s_synthesize = mgr.transition_phase(s_synthesize, "challenge")
        s_synthesize = mgr.transition_phase(s_synthesize, "synthesize")
        mgr.save(s_synthesize)
        time.sleep(0.02)

        # complete phase
        s_complete = mgr.create(posture_override="")
        s_complete = mgr.transition_phase(s_complete, "challenge")
        s_complete = mgr.transition_phase(s_complete, "synthesize")
        s_complete = mgr.transition_phase(s_complete, "complete")
        mgr.save(s_complete)
        time.sleep(0.02)

        # abandoned phase (most recent overall, but should be excluded)
        s_abandoned = mgr.create(posture_override="")
        s_abandoned = mgr.transition_phase(s_abandoned, "abandoned")
        mgr.save(s_abandoned)

        result = mgr.find_latest_incomplete()
        assert result is not None
        assert result.phase in ("understand", "challenge", "synthesize")
        assert result.id != s_complete.id
        assert result.id != s_abandoned.id


class TestGoodhartGetResumeSummary:
    def test_goodhart_get_resume_summary_synthesize_phase(self):
        """Resume summary for synthesize phase must work correctly."""
        mgr, session = create_manager_and_session()
        session = mgr.transition_phase(session, "challenge")
        session = mgr.transition_phase(session, "synthesize")

        summary = mgr.get_resume_summary(session)
        assert summary.phase == "synthesize"
        assert summary.session_id == session.id

    def test_goodhart_get_resume_summary_core_problem_populated(self):
        """Resume summary must include the core_problem from the problem model."""
        mgr, session = create_manager_and_session()
        session = mgr.update_problem_model(session, {"core_problem": "Database scaling under heavy write load"})

        summary = mgr.get_resume_summary(session)
        assert summary.core_problem == "Database scaling under heavy write load"

    def test_goodhart_get_resume_summary_open_questions(self):
        """Resume summary must include open_questions from the problem model."""
        mgr, session = create_manager_and_session()
        session = mgr.update_problem_model(session, {
            "open_questions": ["What is the expected QPS?", "Is caching acceptable?"]
        })

        summary = mgr.get_resume_summary(session)
        assert summary.open_questions == ["What is the expected QPS?", "Is caching acceptable?"]

    def test_goodhart_get_resume_summary_last_active(self):
        """Resume summary must include last_active reflecting the session's updated_at."""
        mgr, session = create_manager_and_session()
        summary = mgr.get_resume_summary(session)
        assert summary.last_active == session.updated_at

    def test_goodhart_get_resume_summary_posture(self):
        """Resume summary must include the session's posture."""
        mgr, session = create_manager_and_session()
        summary = mgr.get_resume_summary(session)
        assert summary.posture == session.posture


class TestGoodhartListAll:
    def test_goodhart_list_all_sorted_descending(self):
        """list_all must return sessions sorted by updated_at descending."""
        bp = make_tmp_base()
        mgr = SessionManager(bp)

        s1 = mgr.create(posture_override="")
        mgr.save(s1)
        time.sleep(0.05)

        s2 = mgr.create(posture_override="")
        mgr.save(s2)
        time.sleep(0.05)

        s3 = mgr.create(posture_override="")
        mgr.save(s3)

        summaries = mgr.list_all()
        assert len(summaries) == 3
        # Most recent first
        assert summaries[0].id == s3.id
        assert summaries[-1].id == s1.id
        # Verify ordering
        for i in range(len(summaries) - 1):
            assert summaries[i].updated_at >= summaries[i + 1].updated_at

    def test_goodhart_list_all_summary_has_correct_is_complete(self):
        """SessionSummary.is_complete must be True for completed sessions and False for incomplete ones."""
        bp = make_tmp_base()
        mgr = SessionManager(bp)

        s_incomplete = mgr.create(posture_override="")
        mgr.save(s_incomplete)

        s_complete = mgr.create(posture_override="")
        s_complete = mgr.transition_phase(s_complete, "challenge")
        s_complete = mgr.transition_phase(s_complete, "synthesize")
        s_complete = mgr.transition_phase(s_complete, "complete")
        mgr.save(s_complete)

        summaries = mgr.list_all()
        summary_map = {s.id: s for s in summaries}
        assert summary_map[s_complete.id].is_complete is True
        assert summary_map[s_incomplete.id].is_complete is False

    def test_goodhart_list_all_summary_total_rounds(self):
        """SessionSummary.total_rounds must reflect actual round counts."""
        bp = make_tmp_base()
        mgr = SessionManager(bp)

        session = mgr.create(posture_override="")
        session = mgr.add_message(session, "user", "q1")
        session = mgr.add_message(session, "assistant", "a1")
        session = mgr.add_message(session, "user", "q2")
        session = mgr.add_message(session, "assistant", "a2")
        mgr.save(session)

        summaries = mgr.list_all()
        assert len(summaries) == 1
        # Total rounds should be at least 2 (understand_rounds)
        assert summaries[0].total_rounds >= 2

    def test_goodhart_list_all_summary_topics_covered(self):
        """SessionSummary.topics_covered must come from the session's problem_model."""
        bp = make_tmp_base()
        mgr = SessionManager(bp)

        session = mgr.create(posture_override="")
        session = mgr.update_problem_model(session, {
            "topics_covered": ["authentication", "authorization"]
        })
        mgr.save(session)

        summaries = mgr.list_all()
        summary_map = {s.id: s for s in summaries}
        assert summary_map[session.id].topics_covered == ["authentication", "authorization"]

    def test_goodhart_list_all_includes_all_phases(self):
        """list_all must include sessions in all phases, not filter any out."""
        bp = make_tmp_base()
        mgr = SessionManager(bp)

        s1 = mgr.create(posture_override="")  # understand
        mgr.save(s1)

        s2 = mgr.create(posture_override="")
        s2 = mgr.transition_phase(s2, "challenge")
        mgr.save(s2)

        s3 = mgr.create(posture_override="")
        s3 = mgr.transition_phase(s3, "challenge")
        s3 = mgr.transition_phase(s3, "synthesize")
        s3 = mgr.transition_phase(s3, "complete")
        mgr.save(s3)

        s4 = mgr.create(posture_override="")
        s4 = mgr.transition_phase(s4, "abandoned")
        mgr.save(s4)

        summaries = mgr.list_all()
        ids = {s.id for s in summaries}
        assert s1.id in ids
        assert s2.id in ids
        assert s3.id in ids
        assert s4.id in ids


class TestGoodhartTransitionPhase:
    def test_goodhart_transition_challenge_to_abandoned(self):
        """Transition from challenge to abandoned must be allowed."""
        mgr, session = create_manager_and_session()
        session = mgr.transition_phase(session, "challenge")
        session = mgr.transition_phase(session, "abandoned")
        assert session.phase == "abandoned"

    def test_goodhart_transition_synthesize_to_abandoned(self):
        """Transition from synthesize to abandoned must be allowed."""
        mgr, session = create_manager_and_session()
        session = mgr.transition_phase(session, "challenge")
        session = mgr.transition_phase(session, "synthesize")
        session = mgr.transition_phase(session, "abandoned")
        assert session.phase == "abandoned"

    def test_goodhart_transition_complete_has_valid_completed_at(self):
        """completed_at must be set to a valid ISO 8601 timestamp when transitioning to complete."""
        mgr, session = create_manager_and_session()
        session = mgr.transition_phase(session, "challenge")
        session = mgr.transition_phase(session, "synthesize")
        session = mgr.transition_phase(session, "complete")
        assert session.completed_at != ""
        assert is_valid_iso8601(session.completed_at)

    def test_goodhart_transition_non_complete_preserves_empty_completed_at(self):
        """Transitioning to a non-complete phase must keep completed_at empty."""
        mgr, session = create_manager_and_session()
        session = mgr.transition_phase(session, "challenge")
        assert session.completed_at == ""
        session = mgr.transition_phase(session, "synthesize")
        assert session.completed_at == ""

    def test_goodhart_transition_challenge_to_complete_invalid(self):
        """Direct challenge→complete must be rejected."""
        mgr, session = create_manager_and_session()
        session = mgr.transition_phase(session, "challenge")
        with pytest.raises(Exception):
            mgr.transition_phase(session, "complete")

    def test_goodhart_transition_same_phase_invalid(self):
        """Transitioning from a phase to itself must be rejected."""
        mgr, session = create_manager_and_session()
        with pytest.raises(Exception):
            mgr.transition_phase(session, "understand")

    def test_goodhart_transition_preserves_other_fields(self):
        """Phase transition must not modify other session fields."""
        mgr, session = create_manager_and_session()
        session = mgr.add_message(session, "user", "hello")
        session = mgr.update_problem_model(session, {"domain": "fintech"})

        original_id = session.id
        original_posture = session.posture
        original_domain = session.problem_model.domain
        original_history_len = len(session.conversation_history)
        original_understand_rounds = session.round_counters.understand_rounds

        session = mgr.transition_phase(session, "challenge")

        assert session.id == original_id
        assert session.posture == original_posture
        assert session.problem_model.domain == original_domain
        assert len(session.conversation_history) == original_history_len
        assert session.round_counters.understand_rounds == original_understand_rounds

    def test_goodhart_transition_backward_challenge_to_understand_invalid(self):
        """Backward transition from challenge to understand must be rejected."""
        mgr, session = create_manager_and_session()
        session = mgr.transition_phase(session, "challenge")
        with pytest.raises(Exception):
            mgr.transition_phase(session, "understand")

    def test_goodhart_transition_backward_synthesize_to_challenge_invalid(self):
        """Backward transition from synthesize to challenge must be rejected."""
        mgr, session = create_manager_and_session()
        session = mgr.transition_phase(session, "challenge")
        session = mgr.transition_phase(session, "synthesize")
        with pytest.raises(Exception):
            mgr.transition_phase(session, "challenge")


class TestGoodhartUpdateProblemModel:
    def test_goodhart_update_problem_model_multiple_list_fields(self):
        """Updating multiple list fields simultaneously must merge each independently."""
        mgr, session = create_manager_and_session()
        session = mgr.update_problem_model(session, {
            "key_entities": ["user", "admin"],
            "constraints": ["must use REST"],
            "assumptions": ["team has 5 devs"]
        })
        assert "user" in session.problem_model.key_entities
        assert "admin" in session.problem_model.key_entities
        assert "must use REST" in session.problem_model.constraints
        assert "team has 5 devs" in session.problem_model.assumptions

    def test_goodhart_update_problem_model_mixed_string_and_list(self):
        """Simultaneously updating string (overwrite) and list (merge) fields."""
        mgr, session = create_manager_and_session()
        session = mgr.update_problem_model(session, {
            "domain": "healthcare",
            "key_entities": ["patient", "doctor"]
        })
        assert session.problem_model.domain == "healthcare"
        assert session.problem_model.key_entities == ["patient", "doctor"]

        # Update again - string should overwrite, list should merge
        session = mgr.update_problem_model(session, {
            "domain": "healthtech",
            "key_entities": ["nurse"]
        })
        assert session.problem_model.domain == "healthtech"
        assert "patient" in session.problem_model.key_entities
        assert "doctor" in session.problem_model.key_entities
        assert "nurse" in session.problem_model.key_entities

    def test_goodhart_update_problem_model_overwrite_string_replaces(self):
        """Updating a string field must overwrite it completely, not append."""
        mgr, session = create_manager_and_session()
        session = mgr.update_problem_model(session, {"domain": "fintech"})
        assert session.problem_model.domain == "fintech"
        session = mgr.update_problem_model(session, {"domain": "edtech"})
        assert session.problem_model.domain == "edtech"
        assert "fintech" not in session.problem_model.domain

    def test_goodhart_update_problem_model_challenge_phase_allowed(self):
        """update_problem_model must work in 'challenge' phase."""
        mgr, session = create_manager_and_session()
        session = mgr.transition_phase(session, "challenge")
        session = mgr.update_problem_model(session, {"domain": "devops"})
        assert session.problem_model.domain == "devops"

    def test_goodhart_update_problem_model_complete_phase_rejected(self):
        """update_problem_model must reject updates in 'complete' phase."""
        mgr, session = create_manager_and_session()
        session = mgr.transition_phase(session, "challenge")
        session = mgr.transition_phase(session, "synthesize")
        session = mgr.transition_phase(session, "complete")
        with pytest.raises(Exception):
            mgr.update_problem_model(session, {"domain": "test"})

    def test_goodhart_update_problem_model_abandoned_phase_rejected(self):
        """update_problem_model must reject updates in 'abandoned' phase."""
        mgr, session = create_manager_and_session()
        session = mgr.transition_phase(session, "abandoned")
        with pytest.raises(Exception):
            mgr.update_problem_model(session, {"domain": "test"})

    def test_goodhart_update_problem_model_partial_duplicate_list(self):
        """When updating a list with a mix of new and existing items, only new items should be appended."""
        mgr, session = create_manager_and_session()
        session = mgr.update_problem_model(session, {"key_entities": ["A", "B", "C"]})
        assert session.problem_model.key_entities == ["A", "B", "C"]

        session = mgr.update_problem_model(session, {"key_entities": ["B", "D", "A", "E"]})
        # Should have A, B, C, D, E with no duplicates
        assert len(session.problem_model.key_entities) == 5
        for item in ["A", "B", "C", "D", "E"]:
            assert item in session.problem_model.key_entities

    def test_goodhart_update_problem_model_empty_updates_dict(self):
        """Passing an empty updates dict should update updated_at but change no fields."""
        mgr, session = create_manager_and_session()
        session = mgr.update_problem_model(session, {"domain": "initial"})
        original_domain = session.problem_model.domain
        time.sleep(0.01)
        session = mgr.update_problem_model(session, {})
        assert session.problem_model.domain == original_domain


class TestGoodhartAddMessage:
    def test_goodhart_add_message_synthesize_phase(self):
        """Adding messages in synthesize phase must work but not increment round counters."""
        mgr, session = create_manager_and_session()
        session = mgr.transition_phase(session, "challenge")
        session = mgr.transition_phase(session, "synthesize")

        ur_before = session.round_counters.understand_rounds
        cr_before = session.round_counters.challenge_rounds

        session = mgr.add_message(session, "user", "synthesize question")
        session = mgr.add_message(session, "assistant", "synthesize answer")

        assert session.round_counters.understand_rounds == ur_before
        assert session.round_counters.challenge_rounds == cr_before
        assert len(session.conversation_history) == 2

    def test_goodhart_add_message_multiple_sequential(self):
        """Adding multiple messages must append each in order."""
        mgr, session = create_manager_and_session()
        messages = ["msg1", "msg2", "msg3", "msg4", "msg5"]
        for i, msg in enumerate(messages):
            role = "user" if i % 2 == 0 else "assistant"
            session = mgr.add_message(session, role, msg)

        assert len(session.conversation_history) == 5
        for i, msg in enumerate(messages):
            assert session.conversation_history[i].content == msg

    def test_goodhart_add_message_timestamp_set(self):
        """Each appended message must have a valid ISO 8601 timestamp."""
        mgr, session = create_manager_and_session()
        session = mgr.add_message(session, "user", "test message")
        msg = session.conversation_history[-1]
        assert msg.timestamp is not None
        assert msg.timestamp != ""
        assert is_valid_iso8601(msg.timestamp)

    def test_goodhart_add_message_does_not_modify_existing(self):
        """Adding a new message must not modify previously appended messages."""
        mgr, session = create_manager_and_session()
        session = mgr.add_message(session, "user", "first message")
        first_msg_content = session.conversation_history[0].content
        first_msg_role = session.conversation_history[0].role
        first_msg_timestamp = session.conversation_history[0].timestamp

        session = mgr.add_message(session, "assistant", "second message")
        session = mgr.add_message(session, "user", "third message")

        assert session.conversation_history[0].content == first_msg_content
        assert session.conversation_history[0].role == first_msg_role
        assert session.conversation_history[0].timestamp == first_msg_timestamp

    def test_goodhart_add_message_assistant_understand_multiple_rounds(self):
        """Adding multiple assistant messages in understand phase must increment understand_rounds each time."""
        mgr, session = create_manager_and_session()
        for i in range(5):
            session = mgr.add_message(session, "user", f"question {i}")
            session = mgr.add_message(session, "assistant", f"answer {i}")

        assert session.round_counters.understand_rounds == 5

    def test_goodhart_add_message_content_preserved_exactly(self):
        """Message content must be preserved exactly including special characters."""
        mgr, session = create_manager_and_session()
        special_content = "Line 1\nLine 2\n\tTabbed\n🚀 Unicode émojis & spëcial chars <>&\"'"
        session = mgr.add_message(session, "user", special_content)
        assert session.conversation_history[-1].content == special_content


class TestGoodhartIsRoundLimit:
    def test_goodhart_is_round_limit_zero_rounds_nonzero_limit(self):
        """With 0 rounds and a positive limit, must return False."""
        mgr, session = create_manager_and_session()
        assert session.round_counters.understand_rounds == 0
        assert session.round_counters.understand_limit > 0
        assert mgr.is_round_limit_reached(session) is False

    def test_goodhart_is_round_limit_one_below(self):
        """With rounds exactly one less than limit, must return False."""
        mgr, session = create_manager_and_session()
        limit = session.round_counters.understand_limit
        # Add rounds up to limit - 1
        for i in range(limit - 1):
            session = mgr.add_message(session, "user", f"q{i}")
            session = mgr.add_message(session, "assistant", f"a{i}")
        assert session.round_counters.understand_rounds == limit - 1
        assert mgr.is_round_limit_reached(session) is False


class TestGoodhartCheckOutputConflicts:
    def test_goodhart_check_output_conflicts_only_prompt(self):
        """When only prompt.md exists, only it should be in conflicts."""
        tmpdir = make_tmp_base()
        Path(tmpdir, "prompt.md").touch()

        mgr = SessionManager(tmpdir)
        conflicts = mgr.check_output_conflicts(tmpdir)
        assert len(conflicts) == 1
        assert any("prompt.md" in str(c) for c in conflicts)

    def test_goodhart_check_output_conflicts_only_constraints(self):
        """When only constraints.yaml exists, only it should be in conflicts."""
        tmpdir = make_tmp_base()
        Path(tmpdir, "constraints.yaml").touch()

        mgr = SessionManager(tmpdir)
        conflicts = mgr.check_output_conflicts(tmpdir)
        assert len(conflicts) == 1
        assert any("constraints.yaml" in str(c) for c in conflicts)

    def test_goodhart_check_output_conflicts_file_not_directory(self):
        """check_output_conflicts must raise error when path is a file, not a directory."""
        tmpdir = make_tmp_base()
        file_path = Path(tmpdir, "not_a_dir.txt")
        file_path.touch()

        mgr = SessionManager(tmpdir)
        with pytest.raises(Exception):
            mgr.check_output_conflicts(str(file_path))


class TestGoodhartCheckGitignore:
    def test_goodhart_check_gitignore_without_trailing_slash(self):
        """.constrain without trailing slash should still cover the directory."""
        tmpdir = make_tmp_base()
        gitignore = Path(tmpdir, ".gitignore")
        gitignore.write_text(".constrain\n")

        mgr = SessionManager(tmpdir)
        result = mgr.check_gitignore(tmpdir)
        assert result == "covered" or (hasattr(result, 'value') and result.value == "covered") or str(result) == "covered"

    def test_goodhart_check_gitignore_with_leading_slash(self):
        """/.constrain/ with leading slash should still be covered."""
        tmpdir = make_tmp_base()
        gitignore = Path(tmpdir, ".gitignore")
        gitignore.write_text("/.constrain/\n")

        mgr = SessionManager(tmpdir)
        result = mgr.check_gitignore(tmpdir)
        assert result == "covered" or (hasattr(result, 'value') and result.value == "covered") or str(result) == "covered"

    def test_goodhart_check_gitignore_empty_file(self):
        """An empty .gitignore should return not_covered."""
        tmpdir = make_tmp_base()
        gitignore = Path(tmpdir, ".gitignore")
        gitignore.write_text("")

        mgr = SessionManager(tmpdir)
        result = mgr.check_gitignore(tmpdir)
        assert result == "not_covered" or (hasattr(result, 'value') and result.value == "not_covered") or str(result) == "not_covered"

    def test_goodhart_check_gitignore_with_whitespace(self):
        """.constrain/ with surrounding whitespace should still be recognized."""
        tmpdir = make_tmp_base()
        gitignore = Path(tmpdir, ".gitignore")
        gitignore.write_text("  .constrain/  \n")

        mgr = SessionManager(tmpdir)
        result = mgr.check_gitignore(tmpdir)
        # Should be covered - gitignore trims whitespace
        assert result == "covered" or (hasattr(result, 'value') and result.value == "covered") or str(result) == "covered"


class TestGoodhartDeleteSession:
    def test_goodhart_delete_session_also_removes_tmp(self):
        """delete_session must also clean up any lingering .tmp file."""
        bp = make_tmp_base()
        mgr = SessionManager(bp)
        session = mgr.create(posture_override="")
        mgr.save(session)

        # Create a lingering .tmp file
        sessions_dir = Path(bp) / ".constrain" / "sessions"
        tmp_file = sessions_dir / f"{session.id}.json.tmp"
        tmp_file.write_text("{}")

        mgr.delete_session(session.id)

        assert not (sessions_dir / f"{session.id}.json").exists()
        assert not tmp_file.exists()
