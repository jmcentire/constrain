"""
Hidden adversarial acceptance tests for posture.py (Posture & Prompt Generation component).

These tests target gaps in visible test coverage to catch implementations that
hardcode returns, skip validation, or only handle specific visible test inputs.
"""
import os
import pytest
from unittest.mock import patch

from src.models import Phase, Posture, ProblemModel
from src.posture import (
    get_system_prompt,
    get_revision_prompt,
    select_posture,
    get_posture_description,
    _format_problem_model,
    _understand_prompt,
    _challenge_prompt,
    _synthesize_prompt,
    POSTURE_DESCRIPTIONS,
)


def _make_model(**kwargs) -> ProblemModel:
    """Helper to create ProblemModel with given overrides."""
    return ProblemModel(**kwargs)


# ---------------------------------------------------------------------------
# _format_problem_model: field omission, bullet rendering, novel content
# ---------------------------------------------------------------------------

class TestGoodhartFormatProblemModel:

    def test_goodhart_format_model_omits_default_fields(self):
        """Fields at their default (empty) value should be omitted from output."""
        model_desc_only = _make_model(system_description="My web service")
        model_goal_only = _make_model(primary_goal="Improve latency")
        
        out_desc = _format_problem_model(model_desc_only)
        out_goal = _format_problem_model(model_goal_only)
        
        # They should differ since they have different fields populated
        assert out_desc != out_goal, "Format output should differ for models with different populated fields"
        
        # The desc-only output should not mention goal-related labels when goal is empty
        assert "My web service" in out_desc
        assert "Improve latency" in out_goal
        assert "Improve latency" not in out_desc
        assert "My web service" not in out_goal

    def test_goodhart_format_model_list_bullets(self):
        """List fields should be rendered as bulleted sub-items, each on its own line."""
        model = _make_model(dependencies=["Redis cache", "PostgreSQL 15", "RabbitMQ"])
        out = _format_problem_model(model)
        
        # Each item should appear in the output
        assert "Redis cache" in out
        assert "PostgreSQL 15" in out
        assert "RabbitMQ" in out
        
        # Items should be on separate lines (bulleted)
        lines = out.split("\n")
        redis_lines = [l for l in lines if "Redis cache" in l]
        pg_lines = [l for l in lines if "PostgreSQL 15" in l]
        assert len(redis_lines) >= 1
        assert len(pg_lines) >= 1
        # They should be on different lines
        assert redis_lines[0] != pg_lines[0]

    def test_goodhart_format_model_novel_content_passthrough(self):
        """Format must actually serialize field values, not return pre-baked content."""
        model = _make_model(
            system_description="xK9mQ_unique_marker_alpha",
            primary_goal="zW3pR_unique_marker_beta",
        )
        out = _format_problem_model(model)
        assert "xK9mQ_unique_marker_alpha" in out
        assert "zW3pR_unique_marker_beta" in out

    def test_goodhart_format_model_challenged_assumptions_and_risks(self):
        """Format must handle challenged_assumptions and risk_areas fields."""
        model = _make_model(
            challenged_assumptions=["Users always have fast internet - CHALLENGED: many on 3G"],
            risk_areas=["Single point of failure in auth service"],
        )
        out = _format_problem_model(model)
        assert "Users always have fast internet" in out
        assert "Single point of failure in auth service" in out

    def test_goodhart_format_model_open_questions(self):
        """Format must render open_questions list field properly."""
        model = _make_model(
            open_questions=["How will auth scale?", "What is the SLA target?"]
        )
        out = _format_problem_model(model)
        assert "How will auth scale?" in out
        assert "What is the SLA target?" in out

    def test_goodhart_format_model_long_lists(self):
        """Format should handle lists with many items, rendering all of them."""
        deps = [f"dep_{i}" for i in range(10)]
        model = _make_model(dependencies=deps)
        out = _format_problem_model(model)
        for dep in deps:
            assert dep in out, f"Expected '{dep}' in formatted output"


# ---------------------------------------------------------------------------
# _understand_prompt: model embedding, variability
# ---------------------------------------------------------------------------

class TestGoodhartUnderstandPrompt:

    def test_goodhart_understand_prompt_contains_model_data(self):
        """Understand prompt must embed actual model state, not a static template."""
        model = _make_model(primary_goal="SENTINEL_understand_8k2m")
        out = _understand_prompt(model)
        assert "SENTINEL_understand_8k2m" in out

    def test_goodhart_understand_prompt_varies_with_model(self):
        """Understand prompt must change when problem model changes."""
        model_a = _make_model(system_description="AAA_system_unique")
        model_b = _make_model(system_description="BBB_system_unique")
        out_a = _understand_prompt(model_a)
        out_b = _understand_prompt(model_b)
        assert out_a != out_b, "Understand prompt should differ for different models"
        assert "AAA_system_unique" in out_a
        assert "BBB_system_unique" in out_b


# ---------------------------------------------------------------------------
# _challenge_prompt: model embedding, JSON block in all postures
# ---------------------------------------------------------------------------

class TestGoodhartChallengePrompt:

    def test_goodhart_challenge_prompt_contains_model_data(self):
        """Challenge prompts must include actual problem model data for all postures."""
        model = _make_model(system_description="UNIQUE_SENTINEL_7x9q")
        for posture in Posture:
            out = _challenge_prompt(model, posture)
            assert "UNIQUE_SENTINEL_7x9q" in out, (
                f"Challenge prompt for {posture.name} should contain model data"
            )

    def test_goodhart_challenge_json_block_present_all_postures(self):
        """JSON block instruction must be present in challenge prompts for ALL posture variants."""
        model = _make_model()
        # Get a reference JSON block indicator from UNDERSTAND (which we know has it)
        understand_out = _understand_prompt(model)
        # Find a distinctive JSON-related substring that should be the shared _JSON_BLOCK_INSTRUCTION
        # We check for common indicators of JSON block instruction
        for posture in Posture:
            out = _challenge_prompt(model, posture)
            # Should contain JSON-related formatting instruction
            json_indicators = ["json", "JSON", "```"]
            has_json = any(ind.lower() in out.lower() for ind in json_indicators)
            assert has_json, (
                f"Challenge prompt for {posture.name} should contain JSON block instruction"
            )

    def test_goodhart_challenge_prompt_model_propagation(self):
        """Challenge prompt must change when problem model changes."""
        model_a = _make_model(system_description="MODEL_A_content_unique_789")
        model_b = _make_model(system_description="MODEL_B_content_unique_321")
        out_a = _challenge_prompt(model_a, Posture.CRITIC)
        out_b = _challenge_prompt(model_b, Posture.CRITIC)
        assert out_a != out_b
        assert "MODEL_A_content_unique_789" in out_a
        assert "MODEL_B_content_unique_321" in out_b


# ---------------------------------------------------------------------------
# _synthesize_prompt: model embedding, artifact specs, YAML schema
# ---------------------------------------------------------------------------

class TestGoodhartSynthesizePrompt:

    def test_goodhart_synthesize_prompt_contains_model_data(self):
        """Synthesize prompt must embed actual problem model state."""
        model = _make_model(stakes="SENTINEL_synth_4j7n")
        out = _synthesize_prompt(model)
        assert "SENTINEL_synth_4j7n" in out

    def test_goodhart_synthesize_both_artifacts_mentioned(self):
        """Synthesize prompt must mention both output artifacts with delimiters."""
        model = _make_model()
        out = _synthesize_prompt(model)
        out_lower = out.lower()
        assert "prompt.md" in out_lower or "prompt.md" in out
        assert "constraints.yaml" in out_lower or "constraints.yaml" in out

    def test_goodhart_synthesize_yaml_schema_spec(self):
        """Synthesize prompt must include YAML schema specification details."""
        model = _make_model()
        out = _synthesize_prompt(model)
        out_lower = out.lower()
        assert "yaml" in out_lower, "Synthesize prompt should contain YAML specification"


# ---------------------------------------------------------------------------
# get_system_prompt: dispatch correctness, all postures
# ---------------------------------------------------------------------------

class TestGoodhartGetSystemPrompt:

    def test_goodhart_get_system_prompt_dispatches_correctly(self):
        """get_system_prompt must produce structurally different prompts per phase."""
        model = _make_model(system_description="dispatch_test_content")
        
        out_understand = get_system_prompt(Phase.UNDERSTAND, model, None)
        out_challenge = get_system_prompt(Phase.CHALLENGE, model, Posture.SKEPTIC)
        out_synthesize = get_system_prompt(Phase.SYNTHESIZE, model, None)
        
        assert out_understand != out_challenge
        assert out_understand != out_synthesize
        assert out_challenge != out_synthesize

    def test_goodhart_get_system_prompt_challenge_each_posture(self):
        """get_system_prompt with CHALLENGE must work for ALL posture variants with distinct content."""
        model = _make_model(system_description="posture_variant_test")
        results = {}
        for posture in Posture:
            result = get_system_prompt(Phase.CHALLENGE, model, posture)
            assert result, f"Challenge prompt for {posture.name} should be non-empty"
            assert "posture_variant_test" in result, (
                f"Challenge prompt for {posture.name} should contain model data"
            )
            results[posture] = result
        
        # All 5 should be pairwise distinct
        values = list(results.values())
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                assert values[i] != values[j], (
                    f"Challenge prompts for different postures should be distinct"
                )


# ---------------------------------------------------------------------------
# get_revision_prompt: feedback verbatim, model embedding, regeneration
# ---------------------------------------------------------------------------

class TestGoodhartGetRevisionPrompt:

    def test_goodhart_revision_prompt_feedback_verbatim(self):
        """Revision prompt must contain the feedback string exactly as given."""
        feedback = 'Please change:\n1. Add "security" section\n2. Remove redundant constraints\n{keep braces}'
        model = _make_model()
        out = get_revision_prompt(feedback, model)
        assert feedback in out, "Feedback must appear verbatim in revision prompt"

    def test_goodhart_revision_prompt_contains_novel_model_data(self):
        """Revision prompt must serialize the provided problem model, not a static representation."""
        model = _make_model(system_description="REVISION_SENTINEL_3m8k")
        out = get_revision_prompt("fix it", model)
        assert "REVISION_SENTINEL_3m8k" in out

    def test_goodhart_revision_prompt_regeneration_instruction(self):
        """Revision prompt must instruct full regeneration, not incremental patching."""
        model = _make_model()
        out = get_revision_prompt("Add more detail to constraints", model)
        out_lower = out.lower()
        # Should contain regeneration-oriented language
        regen_terms = ["regenerat", "recreat", "generat", "produce", "create", "write"]
        patch_only = all(term not in out_lower for term in regen_terms)
        assert not patch_only, (
            "Revision prompt should contain regeneration-oriented language"
        )

    def test_goodhart_revision_prompt_both_artifacts_mentioned(self):
        """Revision prompt must specify both prompt.md and constraints.yaml."""
        model = _make_model()
        out = get_revision_prompt("improve clarity", model)
        assert "prompt.md" in out or "prompt.md" in out.lower()
        assert "constraints.yaml" in out or "constraints.yaml" in out.lower()

    def test_goodhart_revision_prompt_tab_only_feedback_rejected(self):
        """Feedback consisting only of tab characters should be rejected as empty."""
        model = _make_model()
        with pytest.raises(Exception):
            get_revision_prompt("\t\t\t", model)


# ---------------------------------------------------------------------------
# select_posture: comprehensive coverage
# ---------------------------------------------------------------------------

class TestGoodhartSelectPosture:

    def test_goodhart_select_posture_env_mixed_case_variants(self):
        """select_posture should handle various mixed-case env var values."""
        test_cases = [
            ("Adversarial", Posture.ADVERSARIAL),
            ("sKePtIc", Posture.SKEPTIC),
            ("collaborator", Posture.COLLABORATOR),
            ("CONTRARIAN", Posture.CONTRARIAN),
            ("Critic", Posture.CRITIC),
        ]
        for env_val, expected in test_cases:
            with patch.dict(os.environ, {"CONSTRAIN_POSTURE": env_val}):
                result = select_posture(None)
                assert result == expected, (
                    f"CONSTRAIN_POSTURE='{env_val}' should resolve to {expected.name}"
                )

    def test_goodhart_select_posture_random_covers_multiple(self):
        """Random selection must be able to produce multiple distinct postures."""
        with patch.dict(os.environ, {}, clear=False):
            # Ensure CONSTRAIN_POSTURE is not set
            env = os.environ.copy()
            env.pop("CONSTRAIN_POSTURE", None)
            with patch.dict(os.environ, env, clear=True):
                results = set()
                for _ in range(200):
                    result = select_posture(None)
                    assert isinstance(result, Posture)
                    results.add(result)
                assert len(results) >= 2, (
                    "Random posture selection should produce multiple distinct values over many calls"
                )

    def test_goodhart_select_posture_override_all_variants(self):
        """Override must work for ALL five posture variants."""
        for posture in Posture:
            result = select_posture(posture)
            assert result == posture, (
                f"Override with {posture.name} should return {posture.name}"
            )

    def test_goodhart_select_posture_env_partial_match_rejected(self):
        """Partial posture names in env var should be rejected."""
        for partial in ["ADVER", "COLL", "CRIT", "SKEP", "CONTRA"]:
            with patch.dict(os.environ, {"CONSTRAIN_POSTURE": partial}):
                with pytest.raises(Exception):
                    select_posture(None)

    def test_goodhart_select_posture_env_empty_string(self):
        """Empty CONSTRAIN_POSTURE env var should either raise or fall through."""
        with patch.dict(os.environ, {"CONSTRAIN_POSTURE": ""}):
            try:
                result = select_posture(None)
                # If it doesn't raise, it should return a valid Posture (fell through)
                assert isinstance(result, Posture)
            except Exception:
                # Raising an error is also acceptable behavior
                pass

    def test_goodhart_select_posture_env_with_whitespace(self):
        """CONSTRAIN_POSTURE with leading/trailing whitespace should either work or raise clear error."""
        with patch.dict(os.environ, {"CONSTRAIN_POSTURE": " ADVERSARIAL "}):
            try:
                result = select_posture(None)
                # If it works, should resolve to ADVERSARIAL
                assert result == Posture.ADVERSARIAL
            except Exception:
                # Raising invalid_env_posture is also acceptable
                pass


# ---------------------------------------------------------------------------
# get_posture_description: semantic content
# ---------------------------------------------------------------------------

class TestGoodhartGetPostureDescription:

    def test_goodhart_posture_description_meaningful_content(self):
        """Posture descriptions should semantically relate to the posture's documented behavior."""
        desc_adv = get_posture_description(Posture.ADVERSARIAL).lower()
        desc_con = get_posture_description(Posture.CONTRARIAN).lower()
        desc_cri = get_posture_description(Posture.CRITIC).lower()
        desc_ske = get_posture_description(Posture.SKEPTIC).lower()
        desc_col = get_posture_description(Posture.COLLABORATOR).lower()
        
        # Each description should contain at least one semantically relevant term
        adv_terms = ["break", "attack", "exploit", "vulnerabilit", "weak", "fail", "flaw", "adversar"]
        assert any(t in desc_adv for t in adv_terms), (
            f"ADVERSARIAL description should relate to breaking/attacking: '{desc_adv}'"
        )
        
        con_terms = ["opposite", "contrar", "against", "oppos", "argu", "reverse", "counter", "other side"]
        assert any(t in desc_con for t in con_terms), (
            f"CONTRARIAN description should relate to opposing: '{desc_con}'"
        )
        
        cri_terms = ["quality", "complete", "rigor", "evaluat", "standard", "thorough", "critic", "assess"]
        assert any(t in desc_cri for t in cri_terms), (
            f"CRITIC description should relate to quality/rigor: '{desc_cri}'"
        )
        
        ske_terms = ["evidence", "question", "doubt", "certain", "proof", "skepti", "assum", "support", "claim"]
        assert any(t in desc_ske for t in ske_terms), (
            f"SKEPTIC description should relate to evidence/questioning: '{desc_ske}'"
        )
        
        col_terms = ["constructi", "gap", "alternativ", "collaborat", "suggest", "improv", "together", "help", "build"]
        assert any(t in desc_col for t in col_terms), (
            f"COLLABORATOR description should relate to constructive approach: '{desc_col}'"
        )


# ---------------------------------------------------------------------------
# POSTURE_DESCRIPTIONS constant: key type check
# ---------------------------------------------------------------------------

class TestGoodhartPostureDescriptions:

    def test_goodhart_posture_descriptions_keys_are_enum_members(self):
        """POSTURE_DESCRIPTIONS keys must be actual Posture enum members, not strings."""
        for key in POSTURE_DESCRIPTIONS:
            assert isinstance(key, Posture), (
                f"POSTURE_DESCRIPTIONS key {key!r} should be a Posture enum member"
            )

    def test_goodhart_posture_descriptions_values_are_nonempty_strings(self):
        """All POSTURE_DESCRIPTIONS values must be non-empty strings."""
        for posture, desc in POSTURE_DESCRIPTIONS.items():
            assert isinstance(desc, str), f"Description for {posture} should be a string"
            assert len(desc.strip()) > 0, f"Description for {posture} should be non-empty"
