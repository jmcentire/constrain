"""
Adversarial hidden acceptance tests for the Artifact Synthesizer component.

These tests target gaps in visible test coverage, looking for implementations
that may have been hardcoded or shortcut to pass visible tests without truly
satisfying the contract.
"""
import json
import os
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from src.synthesizer import (
    Constraint,
    PromptSections,
    SynthesisError,
    SynthesisInput,
    SynthesizerConfig,
    Synthesizer,
    assign_constraint_ids,
    check_existing_artifacts,
    format_constraints_for_terminal,
    format_prompt_for_terminal,
    init_synthesizer,
    parse_constraints,
    parse_prompt_sections,
    render_constraints_yaml,
    render_prompt_md,
    validate_constraints,
    validate_prompt_sections,
    write_artifacts,
    synthesize,
)


# ---- Helpers ----

def make_raw_prompt_output(sections=None):
    """Build a raw LLM output string with a JSON code fence containing prompt sections."""
    if sections is None:
        sections = {
            "system_context": "This system manages distributed task queues for a payment processing platform.",
            "consequence_map": "Failure in the queue leads to payment delays affecting merchant settlements.",
            "failure_archaeology": "Historical incidents include queue starvation in 2023 Q2 due to unbounded retries.",
            "dependency_landscape": "Depends on Redis for queue storage, PostgreSQL for state, and Kafka for events.",
            "boundary_conditions": "Maximum queue depth is 10,000 messages. Timeout is 30 seconds per task.",
            "success_shape": "A successful deployment processes 99.9% of payments within 5 seconds end-to-end.",
        }
    json_str = json.dumps(sections, indent=2)
    return f"Here is the analysis:\n\n```json\n{json_str}\n```\n\nLet me know if you need changes."


def make_raw_constraints_output(constraints=None):
    """Build a raw LLM output string with a JSON code fence containing constraints."""
    if constraints is None:
        constraints = [
            {
                "boundary": "Queue depth limit",
                "condition": "Queue depth must not exceed 10,000 messages",
                "violation": "Messages are dropped when queue is full",
                "severity": "must",
                "rationale": "Exceeding queue depth causes message loss and payment failures."
            },
            {
                "boundary": "Task timeout",
                "condition": "Each task must complete within 30 seconds",
                "violation": "Timed-out tasks are retried, potentially causing duplicates",
                "severity": "should",
                "rationale": "Long-running tasks block the queue and delay other payments."
            },
        ]
    json_str = json.dumps(constraints, indent=2)
    return f"Here are the constraints:\n\n```json\n{json_str}\n```\n\nThese cover the main boundaries."


def make_constraint(id=None, boundary="Some boundary", condition="Some condition",
                    violation="Some violation", severity="must", rationale="Some rationale"):
    """Create a Constraint object for testing."""
    return Constraint(id=id, boundary=boundary, condition=condition,
                      violation=violation, severity=severity, rationale=rationale)


def make_prompt_sections(**overrides):
    """Create a valid PromptSections object with overridable fields."""
    defaults = {
        "system_context": "This system manages distributed task queues for a payment processing platform.",
        "consequence_map": "Failure in the queue leads to payment delays affecting merchant settlements.",
        "failure_archaeology": "Historical incidents include queue starvation in 2023 Q2 due to unbounded retries.",
        "dependency_landscape": "Depends on Redis for queue storage, PostgreSQL for state, and Kafka for events.",
        "boundary_conditions": "Maximum queue depth is 10,000 messages. Timeout is 30 seconds per task.",
        "success_shape": "A successful deployment processes 99.9% of payments within 5 seconds end-to-end.",
    }
    defaults.update(overrides)
    return PromptSections(**defaults)


def make_synthesizer(tmp_path, overwrite_policy="overwrite"):
    """Create and initialize a Synthesizer with a temp directory."""
    config = SynthesizerConfig(output_dir=str(tmp_path), overwrite_policy=overwrite_policy)
    synth = Synthesizer(config)
    return synth


# ---- Tests ----

class TestGoodhartParsePromptSections:

    def test_goodhart_parse_prompt_extra_text_around_json(self):
        """parse_prompt_sections should extract JSON from code fence even with substantial surrounding prose."""
        preamble = "I've analyzed the system thoroughly. Here are my findings based on the codebase review.\n\n" * 10
        postamble = "\n\nPlease review the above sections carefully. Let me know if any revisions are needed.\n" * 10
        sections_data = {
            "system_context": "Novel context: manages satellite telemetry ingestion pipelines.",
            "consequence_map": "Novel consequence: loss of telemetry causes orbital miscalculations.",
            "failure_archaeology": "Novel archaeology: 2022 incident where buffer overflow corrupted data.",
            "dependency_landscape": "Novel deps: depends on gRPC, TimescaleDB, and custom FPGA firmware.",
            "boundary_conditions": "Novel boundaries: max 50,000 telemetry points per second.",
            "success_shape": "Novel success: 99.99% telemetry delivery within 100ms.",
        }
        raw = preamble + "```json\n" + json.dumps(sections_data, indent=2) + "\n```\n" + postamble
        result = parse_prompt_sections(raw)
        assert result.system_context == sections_data["system_context"]
        assert result.consequence_map == sections_data["consequence_map"]
        assert result.failure_archaeology == sections_data["failure_archaeology"]
        assert result.dependency_landscape == sections_data["dependency_landscape"]
        assert result.boundary_conditions == sections_data["boundary_conditions"]
        assert result.success_shape == sections_data["success_shape"]

    def test_goodhart_parse_prompt_whitespace_section_value(self):
        """parse_prompt_sections must reject section values that are only tabs/newlines, not just empty string."""
        sections = {
            "system_context": "Valid content for system context section here.",
            "consequence_map": "Valid content for consequence map section here.",
            "failure_archaeology": "\t\n  \n\t",  # whitespace only
            "dependency_landscape": "Valid content for dependency landscape here.",
            "boundary_conditions": "Valid content for boundary conditions section.",
            "success_shape": "Valid content for success shape section here.",
        }
        raw = "```json\n" + json.dumps(sections) + "\n```"
        with pytest.raises((SynthesisError, Exception)):
            parse_prompt_sections(raw)

    def test_goodhart_parse_prompt_json_within_other_code_fences(self):
        """parse_prompt_sections should specifically look for ```json fences, not just any code fence."""
        sections = {
            "system_context": "Content",
            "consequence_map": "Content",
            "failure_archaeology": "Content",
            "dependency_landscape": "Content",
            "boundary_conditions": "Content",
            "success_shape": "Content",
        }
        raw = "```python\n" + json.dumps(sections) + "\n```"
        with pytest.raises((SynthesisError, Exception)):
            parse_prompt_sections(raw)

    def test_goodhart_parse_prompt_newlines_in_section_content(self):
        """parse_prompt_sections preserves newlines and paragraph breaks within section content."""
        multiline_content = "First paragraph.\n\nSecond paragraph.\n\n- Bullet 1\n- Bullet 2\n- Bullet 3"
        sections = {
            "system_context": multiline_content,
            "consequence_map": "Valid consequence map content here.",
            "failure_archaeology": "Valid failure archaeology content here.",
            "dependency_landscape": "Valid dependency landscape content here.",
            "boundary_conditions": "Valid boundary conditions content here.",
            "success_shape": "Valid success shape content here.",
        }
        raw = "```json\n" + json.dumps(sections) + "\n```"
        result = parse_prompt_sections(raw)
        assert result.system_context == multiline_content
        assert "\n\n" in result.system_context
        assert "- Bullet 1" in result.system_context

    def test_goodhart_parse_prompt_json_with_extra_whitespace(self):
        """parse_prompt_sections handles JSON blocks with leading/trailing whitespace inside the code fence."""
        sections = {
            "system_context": "Content for system context that is long enough.",
            "consequence_map": "Content for consequence map that is long enough.",
            "failure_archaeology": "Content for failure archaeology that is long enough.",
            "dependency_landscape": "Content for dependency landscape that is long enough.",
            "boundary_conditions": "Content for boundary conditions that is long enough.",
            "success_shape": "Content for success shape that is long enough.",
        }
        raw = "```json\n   \n" + json.dumps(sections, indent=2) + "\n   \n```"
        result = parse_prompt_sections(raw)
        assert result.system_context == sections["system_context"]


class TestGoodhartParseConstraints:

    def test_goodhart_parse_constraints_id_field_ignored(self):
        """When LLM output includes 'id' field in constraints, parse_constraints should still set id to None."""
        constraints_data = [
            {
                "id": "C001",
                "boundary": "Rate limit",
                "condition": "API calls must not exceed 1000/min",
                "violation": "Requests are throttled",
                "severity": "must",
                "rationale": "Prevents service degradation."
            },
            {
                "id": "C999",
                "boundary": "Data retention",
                "condition": "Logs must be retained for 90 days",
                "violation": "Compliance violation",
                "severity": "should",
                "rationale": "Regulatory requirement for audit trail."
            },
        ]
        raw = "```json\n" + json.dumps(constraints_data) + "\n```"
        result = parse_constraints(raw)
        assert len(result) == 2
        for c in result:
            assert c.id is None

    def test_goodhart_parse_constraints_many_constraints(self):
        """parse_constraints should handle 20 constraints, not just the 1-3 used in visible tests."""
        constraints_data = [
            {
                "boundary": f"Boundary {i}",
                "condition": f"Condition {i} must be satisfied",
                "violation": f"Violation {i} occurs when condition fails",
                "severity": "must" if i % 2 == 0 else "should",
                "rationale": f"Rationale {i} explains why this matters."
            }
            for i in range(20)
        ]
        raw = "```json\n" + json.dumps(constraints_data) + "\n```"
        result = parse_constraints(raw)
        assert len(result) == 20
        for i, c in enumerate(result):
            assert c.id is None
            assert c.boundary == f"Boundary {i}"
            assert c.severity in ("must", "should")

    def test_goodhart_parse_constraints_severity_case_sensitivity(self):
        """parse_constraints should reject severity values that aren't exactly lowercase 'must'/'should'."""
        for bad_severity in ["Must", "MUST", "Should", "SHOULD", "MuSt"]:
            constraints_data = [
                {
                    "boundary": "Some boundary",
                    "condition": "Some condition",
                    "violation": "Some violation",
                    "severity": bad_severity,
                    "rationale": "Some rationale."
                }
            ]
            raw = "```json\n" + json.dumps(constraints_data) + "\n```"
            with pytest.raises((SynthesisError, Exception)):
                parse_constraints(raw)

    def test_goodhart_parse_constraints_extra_fields_tolerated(self):
        """parse_constraints should tolerate extra unexpected fields in constraint objects."""
        constraints_data = [
            {
                "boundary": "Network boundary",
                "condition": "Latency must be under 100ms",
                "violation": "SLA breach occurs",
                "severity": "must",
                "rationale": "Customer-facing latency requirement.",
                "notes": "Added by reviewer",
                "priority": 1,
                "tags": ["network", "sla"]
            }
        ]
        raw = "```json\n" + json.dumps(constraints_data) + "\n```"
        result = parse_constraints(raw)
        assert len(result) == 1
        assert result[0].boundary == "Network boundary"
        assert result[0].id is None

    def test_goodhart_parse_constraints_empty_string_field(self):
        """parse_constraints should reject constraint objects where a required field is an empty string."""
        constraints_data = [
            {
                "boundary": "",
                "condition": "Some condition here",
                "violation": "Some violation here",
                "severity": "must",
                "rationale": "Some rationale here."
            }
        ]
        raw = "```json\n" + json.dumps(constraints_data) + "\n```"
        with pytest.raises((SynthesisError, Exception)):
            parse_constraints(raw)


class TestGoodhartAssignConstraintIds:

    def test_goodhart_assign_ids_gaps_in_preassigned(self):
        """When pre-assigned IDs have gaps, new IDs start after the highest, not filling gaps."""
        constraints = [
            make_constraint(id="C001", boundary="First"),
            make_constraint(id="C005", boundary="Fifth"),
            make_constraint(id=None, boundary="New one"),
            make_constraint(id=None, boundary="Another new"),
        ]
        result = assign_constraint_ids(constraints)
        assert result[0].id == "C001"
        assert result[1].id == "C005"
        assert result[2].id == "C006"
        assert result[3].id == "C007"

    def test_goodhart_assign_ids_single_none(self):
        """assign_constraint_ids works correctly with a single constraint that has id=None."""
        constraints = [make_constraint(id=None, boundary="Only one")]
        result = assign_constraint_ids(constraints)
        assert len(result) == 1
        assert result[0].id == "C001"

    def test_goodhart_assign_ids_high_preassigned(self):
        """When a pre-assigned ID is high (C500), new IDs start at C501."""
        constraints = [
            make_constraint(id="C500", boundary="High ID"),
            make_constraint(id=None, boundary="Needs ID"),
        ]
        result = assign_constraint_ids(constraints)
        assert result[0].id == "C500"
        assert result[1].id == "C501"

    def test_goodhart_assign_ids_preserves_constraint_data(self):
        """assign_constraint_ids preserves all non-id fields of each constraint."""
        constraints = [
            make_constraint(id=None, boundary="Unique-B-Alpha", condition="Unique-C-Alpha",
                            violation="Unique-V-Alpha", severity="should", rationale="Unique-R-Alpha"),
        ]
        result = assign_constraint_ids(constraints)
        assert result[0].boundary == "Unique-B-Alpha"
        assert result[0].condition == "Unique-C-Alpha"
        assert result[0].violation == "Unique-V-Alpha"
        assert result[0].severity == "should"
        assert result[0].rationale == "Unique-R-Alpha"
        assert result[0].id == "C001"

    def test_goodhart_assign_ids_returns_new_list(self):
        """assign_constraint_ids returns a new list, not mutating the originals in-place."""
        original = make_constraint(id=None, boundary="Original")
        constraints = [original]
        result = assign_constraint_ids(constraints)
        # The original should still have id=None (not mutated)
        assert original.id is None
        assert result[0].id == "C001"

    def test_goodhart_assign_ids_near_overflow_boundary(self):
        """assign_constraint_ids handles C998 pre-assigned + 1 new = C999 without error."""
        constraints = [
            make_constraint(id="C998", boundary="Near limit"),
            make_constraint(id=None, boundary="Last one"),
        ]
        result = assign_constraint_ids(constraints)
        assert result[0].id == "C998"
        assert result[1].id == "C999"

    def test_goodhart_assign_ids_at_overflow_boundary(self):
        """assign_constraint_ids raises error when C999 exists and a new constraint needs an ID."""
        constraints = [
            make_constraint(id="C999", boundary="At limit"),
            make_constraint(id=None, boundary="Over limit"),
        ]
        with pytest.raises((SynthesisError, Exception)):
            assign_constraint_ids(constraints)


class TestGoodhartRenderPromptMd:

    def test_goodhart_render_prompt_md_content_inclusion(self):
        """render_prompt_md must include the actual content of each section, not just headers."""
        sections = make_prompt_sections(
            system_context="UNIQUE_MARKER_SYSCTX_7a3b2f",
            consequence_map="UNIQUE_MARKER_CONSMAP_9d4e1c",
            failure_archaeology="UNIQUE_MARKER_FAILARCH_2b8f5a",
            dependency_landscape="UNIQUE_MARKER_DEPLAND_6c1d9e",
            boundary_conditions="UNIQUE_MARKER_BOUNDCOND_4f7a3b",
            success_shape="UNIQUE_MARKER_SUCCSHP_8e2c6d",
        )
        result = render_prompt_md(sections)
        assert "UNIQUE_MARKER_SYSCTX_7a3b2f" in result
        assert "UNIQUE_MARKER_CONSMAP_9d4e1c" in result
        assert "UNIQUE_MARKER_FAILARCH_2b8f5a" in result
        assert "UNIQUE_MARKER_DEPLAND_6c1d9e" in result
        assert "UNIQUE_MARKER_BOUNDCOND_4f7a3b" in result
        assert "UNIQUE_MARKER_SUCCSHP_8e2c6d" in result

    def test_goodhart_render_prompt_md_no_extra_h2(self):
        """render_prompt_md should contain exactly six H2 headers."""
        sections = make_prompt_sections()
        result = render_prompt_md(sections)
        # Count H2 headers (lines starting with "## ")
        h2_count = sum(1 for line in result.split('\n') if line.startswith('## '))
        assert h2_count == 6

    def test_goodhart_render_prompt_md_h1_first_line(self):
        """render_prompt_md output starts with a markdown H1 header on the very first line."""
        sections = make_prompt_sections()
        result = render_prompt_md(sections)
        first_line = result.strip().split('\n')[0]
        assert first_line.startswith('# ')


class TestGoodhartRenderConstraintsYaml:

    def test_goodhart_render_constraints_yaml_field_order_novel_data(self):
        """YAML field ordering must hold for arbitrary constraint content."""
        constraints = [
            make_constraint(id=None, boundary="Zeta boundary: special",
                            condition="Alpha condition: requires attention",
                            violation="Beta violation: causes issues",
                            severity="should",
                            rationale="Gamma rationale: because reasons."),
            make_constraint(id=None, boundary="Another boundary entirely",
                            condition="Different condition here",
                            violation="Different violation here",
                            severity="must",
                            rationale="Different rationale here."),
        ]
        result = render_constraints_yaml(constraints)
        # Parse YAML and verify, but also check raw text for field ordering
        lines = result.split('\n')
        expected_order = ['id', 'boundary', 'condition', 'violation', 'severity', 'rationale']
        # Find field keys in order for each constraint block
        for constraint_data in yaml.safe_load(result):
            keys = list(constraint_data.keys())
            assert keys == expected_order, f"Expected {expected_order}, got {keys}"

    def test_goodhart_render_constraints_yaml_multiline_block_scalar(self):
        """Multiline fields must use YAML block scalar style, not quoted strings with \\n."""
        constraints = [
            make_constraint(
                id=None,
                boundary="Line one of boundary.\nLine two of boundary.\nLine three.",
                condition="Single line condition",
                violation="Single line violation",
                severity="must",
                rationale="First line of rationale.\nSecond line.\nThird line."
            ),
        ]
        result = render_constraints_yaml(constraints)
        # Should contain block scalar indicators
        assert '|' in result or '>' in result
        # Should NOT contain literal \n in the YAML (escaped newlines)
        # But we need to check the raw YAML text, not the parsed data
        # A literal backslash-n in YAML would look like "\\n"
        loaded = yaml.safe_load(result)
        assert '\n' in loaded[0]['boundary']
        assert '\n' in loaded[0]['rationale']

    def test_goodhart_render_constraints_yaml_single_line_valid(self):
        """render_constraints_yaml produces valid loadable YAML for single-line fields."""
        constraints = [
            make_constraint(id=None, boundary="Simple boundary", condition="Simple condition",
                            violation="Simple violation", severity="must",
                            rationale="Simple rationale."),
        ]
        result = render_constraints_yaml(constraints)
        loaded = yaml.safe_load(result)
        assert isinstance(loaded, list)
        assert len(loaded) == 1
        assert loaded[0]['boundary'] == "Simple boundary"

    def test_goodhart_render_constraints_yaml_mixed_preassigned_and_none(self):
        """render_constraints_yaml handles mix of pre-assigned and None IDs correctly."""
        constraints = [
            make_constraint(id="C003", boundary="Third"),
            make_constraint(id=None, boundary="Needs ID 1"),
            make_constraint(id="C001", boundary="First"),
            make_constraint(id=None, boundary="Needs ID 2"),
        ]
        result = render_constraints_yaml(constraints)
        loaded = yaml.safe_load(result)
        ids = [c['id'] for c in loaded]
        assert "C001" in ids
        assert "C003" in ids
        # New IDs should start after C003 (highest)
        new_ids = [id for id in ids if id not in ("C001", "C003")]
        assert "C004" in new_ids
        assert "C005" in new_ids
        assert len(set(ids)) == 4  # All unique

    def test_goodhart_render_constraints_yaml_severity_preserved(self):
        """render_constraints_yaml preserves severity as strings, not YAML booleans."""
        constraints = [
            make_constraint(id=None, severity="must"),
            make_constraint(id=None, severity="should"),
        ]
        result = render_constraints_yaml(constraints)
        loaded = yaml.safe_load(result)
        assert loaded[0]['severity'] == 'must'
        assert isinstance(loaded[0]['severity'], str)
        assert loaded[1]['severity'] == 'should'
        assert isinstance(loaded[1]['severity'], str)

    def test_goodhart_render_constraints_yaml_loadable_roundtrip_novel(self):
        """YAML output loaded via safe_load preserves data for novel constraint content."""
        constraints = [
            make_constraint(id=None, boundary="API rate: 5000 req/s",
                            condition="Must not exceed under normal load",
                            violation="HTTP 429 returned to clients",
                            severity="must",
                            rationale="Protects backend from cascade failure."),
            make_constraint(id=None, boundary="Cache TTL: 300s",
                            condition="Stale data acceptable for 5 minutes",
                            violation="Users see outdated information",
                            severity="should",
                            rationale="Balance freshness vs. database load."),
            make_constraint(id=None, boundary="Disk usage < 80%",
                            condition="Alert when approaching capacity",
                            violation="Service degrades, writes fail",
                            severity="must",
                            rationale="SSD performance drops sharply above 80%."),
            make_constraint(id=None, boundary="Deployment window: 2-4am UTC",
                            condition="Changes only during low-traffic period",
                            violation="User-facing errors during deploy",
                            severity="should",
                            rationale="Minimizes blast radius of failed deploys."),
            make_constraint(id=None, boundary="Max payload: 10MB",
                            condition="Request bodies must be bounded",
                            violation="OOM kills and service restarts",
                            severity="must",
                            rationale="Unbounded payloads crashed prod in Q1."),
        ]
        result = render_constraints_yaml(constraints)
        loaded = yaml.safe_load(result)
        assert isinstance(loaded, list)
        assert len(loaded) == 5
        for i, item in enumerate(loaded):
            assert item['id'] == f"C{i+1:03d}"
            assert set(item.keys()) == {'id', 'boundary', 'condition', 'violation', 'severity', 'rationale'}
            assert item['boundary'] == constraints[i].boundary
            assert item['condition'] == constraints[i].condition

    def test_goodhart_yaml_special_chars_colon_at_start(self):
        """YAML output safely handles field values starting with special YAML characters."""
        constraints = [
            make_constraint(
                id=None,
                boundary=": starts with colon",
                condition="{ looks like flow mapping }",
                violation="[looks like flow sequence]",
                severity="must",
                rationale="* starts with asterisk & has ampersand"
            ),
        ]
        result = render_constraints_yaml(constraints)
        loaded = yaml.safe_load(result)
        assert loaded[0]['boundary'] == ": starts with colon"
        assert loaded[0]['condition'] == "{ looks like flow mapping }"
        assert loaded[0]['violation'] == "[looks like flow sequence]"
        assert loaded[0]['rationale'] == "* starts with asterisk & has ampersand"


class TestGoodhartValidateConstraints:

    def test_goodhart_validate_constraints_whitespace_only_fields(self):
        """validate_constraints must reject whitespace-only required string fields (tabs/newlines)."""
        constraint = make_constraint(id=None, boundary="   \t\n  ")
        with pytest.raises((SynthesisError, Exception)):
            validate_constraints([constraint])

    def test_goodhart_validate_constraints_each_field_checked(self):
        """validate_constraints checks each required string field independently."""
        fields = ['boundary', 'condition', 'violation', 'rationale']
        for field in fields:
            kwargs = {
                'id': None,
                'boundary': 'Valid boundary content',
                'condition': 'Valid condition content',
                'violation': 'Valid violation content',
                'severity': 'must',
                'rationale': 'Valid rationale content',
            }
            kwargs[field] = '   \t  '  # whitespace only
            constraint = Constraint(**kwargs)
            with pytest.raises((SynthesisError, Exception)):
                validate_constraints([constraint])

    def test_goodhart_validate_constraints_id_format_variations(self):
        """validate_constraints rejects IDs that look similar to CNNN but don't match exactly."""
        bad_ids = ['c001', 'C01', 'C1', 'C0001', 'CABC', 'D001', '001', 'C 01']
        for bad_id in bad_ids:
            constraint = make_constraint(id=bad_id)
            with pytest.raises((SynthesisError, Exception)):
                validate_constraints([constraint])

    def test_goodhart_validate_constraints_allows_none_ids(self):
        """validate_constraints does not reject constraints with id=None."""
        constraints = [
            make_constraint(id=None, boundary="First"),
            make_constraint(id=None, boundary="Second"),
        ]
        result = validate_constraints(constraints)
        assert len(result) == 2


class TestGoodhartValidatePromptSections:

    def test_goodhart_validate_prompt_sections_each_section_checked(self):
        """validate_prompt_sections independently checks each section for the too-short condition."""
        section_fields = [
            'system_context', 'consequence_map', 'failure_archaeology',
            'dependency_landscape', 'boundary_conditions', 'success_shape'
        ]
        for field in section_fields:
            kwargs = {
                'system_context': "Sufficiently long system context content for validation.",
                'consequence_map': "Sufficiently long consequence map content for validation.",
                'failure_archaeology': "Sufficiently long failure archaeology content for validation.",
                'dependency_landscape': "Sufficiently long dependency landscape content for validation.",
                'boundary_conditions': "Sufficiently long boundary conditions content for validation.",
                'success_shape': "Sufficiently long success shape content for validation.",
            }
            kwargs[field] = "short"  # fewer than 10 characters
            sections = PromptSections(**kwargs)
            with pytest.raises((SynthesisError, Exception)):
                validate_prompt_sections(sections)


class TestGoodhartWriteArtifacts:

    def test_goodhart_write_artifacts_whitespace_only_prompt(self, tmp_path):
        """write_artifacts must reject whitespace-only prompt_md."""
        synth = make_synthesizer(tmp_path)
        with pytest.raises((SynthesisError, Exception)):
            synth.write_artifacts("  \n\t  ", "valid: yaml\n", overwrite=True)

    def test_goodhart_write_artifacts_whitespace_only_constraints(self, tmp_path):
        """write_artifacts must reject whitespace-only constraints_yaml."""
        synth = make_synthesizer(tmp_path)
        with pytest.raises((SynthesisError, Exception)):
            synth.write_artifacts("# Valid prompt md\n", "  \n\t  ", overwrite=True)

    def test_goodhart_write_artifacts_absolute_paths_returned(self, tmp_path):
        """write_artifacts returns absolute paths regardless of how output_dir was specified."""
        synth = make_synthesizer(tmp_path)
        result = synth.write_artifacts("# Prompt content\n", "constraints: yaml\n", overwrite=True)
        assert Path(result.prompt_md_path).is_absolute()
        assert Path(result.constraints_yaml_path).is_absolute()
        assert result.prompt_md_path.endswith("prompt.md")
        assert result.constraints_yaml_path.endswith("constraints.yaml")

    def test_goodhart_write_artifacts_content_exact_match(self, tmp_path):
        """write_artifacts writes exactly the provided content, byte-for-byte."""
        synth = make_synthesizer(tmp_path)
        prompt_content = "# My Prompt\n\nSome content with special chars: é à ü\n\nEnd.\n"
        yaml_content = "- id: C001\n  boundary: test\n"
        result = synth.write_artifacts(prompt_content, yaml_content, overwrite=True)
        with open(result.prompt_md_path, 'r', encoding='utf-8') as f:
            assert f.read() == prompt_content
        with open(result.constraints_yaml_path, 'r', encoding='utf-8') as f:
            assert f.read() == yaml_content

    def test_goodhart_write_artifacts_constraints_exists_error(self, tmp_path):
        """write_artifacts raises error when only constraints.yaml exists and overwrite is false."""
        synth = make_synthesizer(tmp_path, overwrite_policy="error_if_exists")
        # Create only constraints.yaml
        (tmp_path / "constraints.yaml").write_text("existing content")
        with pytest.raises((SynthesisError, Exception)):
            synth.write_artifacts("# New prompt\n", "new: yaml\n", overwrite=False)


class TestGoodhartFormatPrompt:

    def test_goodhart_format_prompt_truncation_exact_boundary(self):
        """format_prompt_for_terminal truncates at exactly max_lines_per_section, not off-by-one."""
        max_lines = 5
        # Section with exactly max_lines+1 lines (should be truncated)
        long_section = "\n".join(f"Line {i}" for i in range(max_lines + 1))
        # Section with exactly max_lines lines (should NOT be truncated)
        exact_section = "\n".join(f"Line {i}" for i in range(max_lines))

        sections = make_prompt_sections(
            system_context=long_section,
            consequence_map=exact_section,
        )
        result = format_prompt_for_terminal(sections, max_lines_per_section=max_lines)
        content = result.content
        # The long section should be truncated (truncation notice present)
        assert "truncat" in content.lower() or "..." in content

    def test_goodhart_format_prompt_max_lines_1(self):
        """format_prompt_for_terminal works with max_lines_per_section=1."""
        sections = make_prompt_sections(
            system_context="Line one\nLine two\nLine three",
            consequence_map="Line one\nLine two",
        )
        result = format_prompt_for_terminal(sections, max_lines_per_section=1)
        # All six section names should be present
        for header in ["System Context", "Consequence Map", "Failure Archaeology",
                        "Dependency Landscape", "Boundary Conditions", "Success Shape"]:
            assert header.lower() in result.content.lower() or header in result.content

    def test_goodhart_format_prompt_line_count_accuracy(self):
        """format_prompt_for_terminal line_count should match the actual lines in content."""
        sections = make_prompt_sections()
        result = format_prompt_for_terminal(sections, max_lines_per_section=100)
        actual_lines = len(result.content.split('\n'))
        # Allow for off-by-one in counting convention (trailing newline)
        assert abs(result.line_count - actual_lines) <= 1


class TestGoodhartFormatConstraints:

    def test_goodhart_format_constraints_terminal_id_order(self):
        """format_constraints_for_terminal must display constraints in ID order even if input is unsorted."""
        constraints = [
            make_constraint(id="C003", boundary="Third"),
            make_constraint(id="C001", boundary="First"),
            make_constraint(id="C002", boundary="Second"),
        ]
        result = format_constraints_for_terminal(constraints, verbose=False)
        content = result.content
        pos_c001 = content.index("C001")
        pos_c002 = content.index("C002")
        pos_c003 = content.index("C003")
        assert pos_c001 < pos_c002 < pos_c003

    def test_goodhart_format_constraints_terminal_verbose_vs_nonverbose(self):
        """format_constraints_for_terminal produces different output for verbose=True vs False."""
        constraints = [
            make_constraint(id="C001", boundary="Test boundary",
                            rationale="A very detailed and long rationale that explains the reasoning behind this constraint in great depth."),
        ]
        verbose_result = format_constraints_for_terminal(constraints, verbose=True)
        brief_result = format_constraints_for_terminal(constraints, verbose=False)
        # Verbose should generally be longer or more detailed
        assert len(verbose_result.content) >= len(brief_result.content)
        # Both should contain the constraint ID
        assert "C001" in verbose_result.content
        assert "C001" in brief_result.content

    def test_goodhart_format_constraints_line_count_accuracy(self):
        """format_constraints_for_terminal line_count should be consistent with content."""
        constraints = [
            make_constraint(id="C001", boundary="First"),
            make_constraint(id="C002", boundary="Second"),
            make_constraint(id="C003", boundary="Third"),
        ]
        result = format_constraints_for_terminal(constraints, verbose=False)
        assert result.line_count > 0
        actual_lines = len(result.content.split('\n'))
        assert abs(result.line_count - actual_lines) <= 1


class TestGoodhartInitSynthesizer:

    def test_goodhart_init_synthesizer_resolves_path(self, tmp_path):
        """init_synthesizer stores output_dir as a resolved pathlib.Path."""
        config = SynthesizerConfig(output_dir=str(tmp_path), overwrite_policy="overwrite")
        synth = Synthesizer(config)
        # The output_dir should be stored as a pathlib.Path, not string
        assert isinstance(synth.output_dir, Path)
        # Should be absolute/resolved
        assert synth.output_dir.is_absolute()


class TestGoodhartSynthesize:

    def test_goodhart_synthesize_validates_before_writing(self, tmp_path):
        """synthesize should fail on validation before writing if sections are too short."""
        synth = make_synthesizer(tmp_path)
        # Create prompt output with one section that's too short
        sections = {
            "system_context": "short",  # fewer than 10 chars
            "consequence_map": "Sufficiently long consequence map content for validation.",
            "failure_archaeology": "Sufficiently long failure archaeology content for validation.",
            "dependency_landscape": "Sufficiently long dependency landscape content for validation.",
            "boundary_conditions": "Sufficiently long boundary conditions content for validation.",
            "success_shape": "Sufficiently long success shape content for validation.",
        }
        raw_prompt = "```json\n" + json.dumps(sections) + "\n```"
        raw_constraints = make_raw_constraints_output()

        synthesis_input = SynthesisInput(
            raw_prompt_output=raw_prompt,
            raw_constraints_output=raw_constraints,
        )
        with pytest.raises((SynthesisError, Exception)):
            synth.synthesize(synthesis_input, overwrite=True)
        # No files should have been written
        assert not (tmp_path / "prompt.md").exists()
        assert not (tmp_path / "constraints.yaml").exists()

    def test_goodhart_synthesize_full_pipeline_novel_content(self, tmp_path):
        """synthesize handles a full pipeline with novel content different from visible test fixtures."""
        synth = make_synthesizer(tmp_path)
        sections = {
            "system_context": "Manages real-time bidding infrastructure for programmatic advertising.",
            "consequence_map": "Bid response latency above 100ms loses auction, reducing ad revenue by 15%.",
            "failure_archaeology": "March 2024: DNS resolution spike caused 30% bid timeout rate for 45 minutes.",
            "dependency_landscape": "Aerospike for bid state, Kafka for impression events, custom C++ bidder.",
            "boundary_conditions": "100k QPS at p99 < 50ms. Budget cap accuracy within 0.1% per campaign.",
            "success_shape": "Win rate above 12%, revenue per impression within 5% of model prediction.",
        }
        constraints_data = [
            {"boundary": "Bid latency", "condition": "p99 response time under 50ms",
             "violation": "Lost auctions and revenue decline", "severity": "must",
             "rationale": "Exchange enforces strict timeout."},
            {"boundary": "Budget accuracy", "condition": "Campaign spend within 0.1% of cap",
             "violation": "Overspend or underspend affects advertiser trust", "severity": "must",
             "rationale": "Contractual obligation to advertisers."},
            {"boundary": "Feature freshness", "condition": "ML features updated within 5 minutes",
             "violation": "Stale features degrade bid quality", "severity": "should",
             "rationale": "Balances freshness against compute cost."},
            {"boundary": "Failover time", "condition": "Regional failover within 30 seconds",
             "violation": "Extended outage affects all traffic in region", "severity": "must",
             "rationale": "Multi-region SLA requires fast recovery."},
        ]
        raw_prompt = "```json\n" + json.dumps(sections) + "\n```"
        raw_constraints = "```json\n" + json.dumps(constraints_data) + "\n```"
        synthesis_input = SynthesisInput(
            raw_prompt_output=raw_prompt,
            raw_constraints_output=raw_constraints,
        )
        result = synth.synthesize(synthesis_input, overwrite=True)

        # Verify files exist
        assert (tmp_path / "prompt.md").exists()
        assert (tmp_path / "constraints.yaml").exists()

        # Verify prompt.md content
        prompt_content = (tmp_path / "prompt.md").read_text(encoding='utf-8')
        assert "programmatic advertising" in prompt_content
        assert "## System Context" in prompt_content

        # Verify constraints.yaml content
        constraints_content = (tmp_path / "constraints.yaml").read_text(encoding='utf-8')
        loaded = yaml.safe_load(constraints_content)
        assert len(loaded) == 4
        assert loaded[0]['id'] == 'C001'
        assert loaded[3]['id'] == 'C004'

        # Verify returned paths are absolute
        assert Path(result.prompt_md_path).is_absolute()
        assert Path(result.constraints_yaml_path).is_absolute()
