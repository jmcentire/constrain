"""Tests for integrated artifact production (FA-C-009 through FA-C-024).

These tests verify that Constrain produces the right artifacts for
downstream consumption by Pact, Arbiter, Baton, Sentinel, and Ledger.
"""

import pytest
import yaml
from pathlib import Path
from unittest.mock import Mock, patch
from dataclasses import dataclass, field as dc_field
from typing import Optional

from constrain.synthesizer import (
    SynthesisArtifacts,
    parse_synthesis_output,
    validate_artifacts,
    validate_yaml_content,
    write_artifacts,
)
from constrain.models import Phase, Posture, Severity


# ============================================================================
# Helpers
# ============================================================================

_DEFAULT_SCHEMA_HINTS = (
    "version: '1.0'\ngenerated_by: constrain\nsession_id: test-123\n"
    "storage_backends:\n"
    "  - owner_component: db-service\n"
    "    type: postgres\n"
    "    description: main relational store\n"
    "field_hints:\n"
    "  - backend_owner: db-service\n"
    "    field_description: user email address\n"
    "    likely_classification: PII\n"
    "    likely_annotations: [gdpr_erasable]\n"
    "    rationale: personal contact info subject to GDPR\n"
    "  - backend_owner: db-service\n"
    "    field_description: auth token hash\n"
    "    likely_classification: AUTH\n"
    "    likely_annotations: [encrypted_at_rest]\n"
    "    rationale: credential material\n"
    "  - backend_owner: db-service\n"
    "    field_description: system name\n"
    "    likely_classification: PUBLIC\n"
    "    likely_annotations: []\n"
    "    rationale: non-sensitive metadata"
)


def _make_full_raw_output(
    prompt="# System\n\n## Trust and Authority Model\nTrust info\n\n## Component Topology\nTopology info",
    constraints="constraints:\n  - id: C001\n    boundary: auth\n    condition: tokens expire\n    violation: stale session\n    severity: must\n    rationale: security\n    classification_tier: AUTH\n    affected_components: [auth-service]",
    trust_policy="version: '1.0'\ngenerated_by: constrain\nsession_id: test-123\nsystem: test\ntrust:\n  floor: 0.10\n  authority_override_floor: 0.40\n  decay_lambda: 0.05\n  taint_lock_tiers: [PII, FINANCIAL, AUTH, COMPLIANCE]\n  conflict_trust_delta_threshold: 0.20\nclassifications:\n  - field_pattern: user.email\n    tier: PII\n    authoritative_component: auth-service\n    canary_eligible: true\n    canary_pattern: null\n    rationale: personal data\nsoak:\n  base_durations:\n    PUBLIC: 1h\n    PII: 6h\n    FINANCIAL: 24h\n    AUTH: 48h\n    COMPLIANCE: 72h\n  target_requests: 1000\nauthority_map:\n  - component: auth-service\n    domains: [user.*]\n    rationale: owns user data\nhuman_gates:\n  always:\n    - tier: FINANCIAL\n    - tier: AUTH\n    - tier: COMPLIANCE\n  on_low_trust_authoritative: true\n  on_unresolvable_conflict: true",
    component_map="version: '1.0'\ngenerated_by: constrain\nsession_id: test-123\ncomponents:\n  - name: auth-service\n    description: handles authentication\n    type: service\n    port: 8080\n    protocol: http\n    data_access:\n      reads: [PII, AUTH]\n      writes: [AUTH]\n      rationale: manages auth tokens\n    authority:\n      domains: [user.*]\n      rationale: owns user data\n    dependencies: [db-service]\n    constraints: [C001]\n  - name: db-service\n    description: database layer\n    type: service\n    port: 5432\n    protocol: tcp\n    data_access:\n      reads: [PII, AUTH, FINANCIAL]\n      writes: [PII, AUTH, FINANCIAL]\n      rationale: persistence layer\n    authority:\n      domains: []\n      rationale: null\n    dependencies: []\n    constraints: []\nedges:\n  - from: auth-service\n    to: db-service\n    protocol: tcp\n    description: auth token storage",
    schema_hints=_DEFAULT_SCHEMA_HINTS,
):
    return (
        f"--- PROMPT ---\n{prompt}\n"
        f"--- CONSTRAINTS ---\n{constraints}\n"
        f"--- TRUST_POLICY ---\n{trust_policy}\n"
        f"--- COMPONENT_MAP ---\n{component_map}\n"
        f"--- SCHEMA_HINTS ---\n{schema_hints}"
    )


# ============================================================================
# FA-C-009: Synthesize phase produces trust_policy.yaml
# ============================================================================

class TestFAC009:
    def test_parse_produces_trust_policy(self):
        raw = _make_full_raw_output()
        artifacts = parse_synthesis_output(raw)
        assert artifacts.trust_policy_yaml, "trust_policy_yaml must not be empty"

    def test_trust_policy_is_valid_yaml(self):
        raw = _make_full_raw_output()
        artifacts = parse_synthesis_output(raw)
        data = yaml.safe_load(artifacts.trust_policy_yaml)
        assert isinstance(data, dict)

    def test_write_trust_policy(self, tmp_path):
        raw = _make_full_raw_output()
        artifacts = parse_synthesis_output(raw)
        written = write_artifacts(
            artifacts.prompt_md,
            artifacts.constraints_yaml,
            tmp_path,
            trust_policy_yaml=artifacts.trust_policy_yaml,
            component_map_yaml=artifacts.component_map_yaml,
        )
        names = {p.name for p in written}
        assert "trust_policy.yaml" in names
        content = (tmp_path / "trust_policy.yaml").read_text()
        assert "trust:" in content


# ============================================================================
# FA-C-010: Synthesize phase produces component_map.yaml
# ============================================================================

class TestFAC010:
    def test_parse_produces_component_map(self):
        raw = _make_full_raw_output()
        artifacts = parse_synthesis_output(raw)
        assert artifacts.component_map_yaml, "component_map_yaml must not be empty"

    def test_component_map_is_valid_yaml(self):
        raw = _make_full_raw_output()
        artifacts = parse_synthesis_output(raw)
        data = yaml.safe_load(artifacts.component_map_yaml)
        assert isinstance(data, dict)

    def test_write_component_map(self, tmp_path):
        raw = _make_full_raw_output()
        artifacts = parse_synthesis_output(raw)
        written = write_artifacts(
            artifacts.prompt_md,
            artifacts.constraints_yaml,
            tmp_path,
            trust_policy_yaml=artifacts.trust_policy_yaml,
            component_map_yaml=artifacts.component_map_yaml,
        )
        names = {p.name for p in written}
        assert "component_map.yaml" in names


# ============================================================================
# FA-C-011: trust_policy.yaml has at least one classification when data is handled
# ============================================================================

class TestFAC011:
    def test_classifications_present(self):
        raw = _make_full_raw_output()
        artifacts = parse_synthesis_output(raw)
        data = yaml.safe_load(artifacts.trust_policy_yaml)
        assert "classifications" in data
        assert len(data["classifications"]) >= 1


# ============================================================================
# FA-C-012: Every component has a data_access entry
# ============================================================================

class TestFAC012:
    def test_all_components_have_data_access(self):
        raw = _make_full_raw_output()
        artifacts = parse_synthesis_output(raw)
        data = yaml.safe_load(artifacts.component_map_yaml)
        for comp in data["components"]:
            assert "data_access" in comp, f"Component '{comp['name']}' missing data_access"
            assert "reads" in comp["data_access"]
            assert "writes" in comp["data_access"]


# ============================================================================
# FA-C-013: No two components claim authority for overlapping domains
# ============================================================================

class TestFAC013:
    def test_no_overlapping_authority(self):
        raw = _make_full_raw_output()
        artifacts = parse_synthesis_output(raw)
        warnings = validate_artifacts(artifacts)
        overlap_warnings = [w for w in warnings if "Overlapping authority" in w]
        assert not overlap_warnings, f"Overlapping authority detected: {overlap_warnings}"

    def test_detect_overlapping_authority(self):
        """Validator catches overlapping authority domains."""
        cm = (
            "version: '1.0'\n"
            "components:\n"
            "  - name: svc-a\n"
            "    authority:\n"
            "      domains: [user.email]\n"
            "  - name: svc-b\n"
            "    authority:\n"
            "      domains: [user.email]\n"
        )
        artifacts = SynthesisArtifacts(
            prompt_md="test",
            constraints_yaml="constraints: []",
            trust_policy_yaml="trust: {}",
            component_map_yaml=cm,
        )
        warnings = validate_artifacts(artifacts)
        assert any("Overlapping authority" in w for w in warnings)


# ============================================================================
# FA-C-014: Every constraint has classification_tier field
# ============================================================================

class TestFAC014:
    def test_constraints_have_classification_tier(self):
        raw = _make_full_raw_output()
        artifacts = parse_synthesis_output(raw)
        data = yaml.safe_load(artifacts.constraints_yaml)
        for c in data.get("constraints", []):
            assert "classification_tier" in c, f"Constraint '{c.get('id')}' missing classification_tier"

    def test_classification_tier_can_be_null(self):
        """classification_tier may be null for non-data constraints."""
        constraints = (
            "constraints:\n"
            "  - id: C001\n"
            "    boundary: test\n"
            "    condition: test\n"
            "    violation: test\n"
            "    severity: must\n"
            "    rationale: test\n"
            "    classification_tier: null\n"
            "    affected_components: []\n"
        )
        data = yaml.safe_load(constraints)
        assert data["constraints"][0]["classification_tier"] is None


# ============================================================================
# FA-C-015: Edges consistent with component dependencies
# ============================================================================

class TestFAC015:
    def test_edges_match_dependencies(self):
        raw = _make_full_raw_output()
        artifacts = parse_synthesis_output(raw)
        warnings = validate_artifacts(artifacts)
        edge_warnings = [w for w in warnings if "depends on" in w and "no edge" in w]
        assert not edge_warnings, f"Missing edges: {edge_warnings}"

    def test_detect_missing_edge(self):
        """Validator catches dependency without corresponding edge."""
        cm = (
            "version: '1.0'\n"
            "components:\n"
            "  - name: svc-a\n"
            "    dependencies: [svc-b]\n"
            "  - name: svc-b\n"
            "    dependencies: []\n"
            "edges: []\n"
        )
        artifacts = SynthesisArtifacts(
            prompt_md="test",
            constraints_yaml="constraints: []",
            trust_policy_yaml="trust: {}",
            component_map_yaml=cm,
        )
        warnings = validate_artifacts(artifacts)
        assert any("svc-a" in w and "svc-b" in w for w in warnings)


# ============================================================================
# FA-C-016: prompt.md contains Trust and Authority Model section
# ============================================================================

class TestFAC016:
    def test_prompt_has_trust_section(self):
        raw = _make_full_raw_output()
        artifacts = parse_synthesis_output(raw)
        assert "## Trust and Authority Model" in artifacts.prompt_md


# ============================================================================
# FA-C-017: prompt.md contains Component Topology section
# ============================================================================

class TestFAC017:
    def test_prompt_has_topology_section(self):
        raw = _make_full_raw_output()
        artifacts = parse_synthesis_output(raw)
        assert "## Component Topology" in artifacts.prompt_md


# ============================================================================
# FA-C-018: constrain show displays all five artifacts when complete
# ============================================================================

class TestFAC018:
    def test_show_displays_all_artifacts(self):
        """cmd_show displays all artifacts when present."""
        from constrain.cli import cmd_show
        from click.testing import CliRunner

        mock_session = Mock()
        mock_session.prompt_md = "# Prompt"
        mock_session.constraints_yaml = "constraints: []"
        mock_session.trust_policy_yaml = "trust: {}"
        mock_session.component_map_yaml = "components: []"
        mock_session.schema_hints_yaml = "storage_backends: []"
        mock_session.phase = Phase.complete
        mock_session.id = "test-123"

        mock_mgr = Mock()
        mock_mgr.list_all.return_value = [{"id": "test-123", "is_complete": True}]
        mock_mgr.load.return_value = mock_session

        runner = CliRunner()
        with patch("constrain.cli.SessionManager", return_value=mock_mgr):
            result = runner.invoke(cmd_show)

        assert "=== prompt.md ===" in result.output
        assert "=== constraints.yaml ===" in result.output
        assert "=== trust_policy.yaml ===" in result.output
        assert "=== component_map.yaml ===" in result.output
        assert "=== schema_hints.yaml ===" in result.output


# ============================================================================
# FA-C-019: Challenge phase includes conflict-resolution probes
# ============================================================================

class TestFAC019:
    def test_challenge_prompt_has_conflict_probes(self):
        from constrain.posture import get_system_prompt
        from constrain.models import ProblemModel

        pm = ProblemModel(system_description="test system")
        for posture in Posture:
            prompt = get_system_prompt(Phase.challenge, pm, posture)
            assert "conflict" in prompt.lower() or "disagree" in prompt.lower(), (
                f"Challenge prompt for {posture.value} missing conflict-resolution probes"
            )


# ============================================================================
# FA-C-020: authority_override_floor >= trust.floor
# ============================================================================

class TestFAC020:
    def test_valid_override_floor(self):
        raw = _make_full_raw_output()
        artifacts = parse_synthesis_output(raw)
        data = yaml.safe_load(artifacts.trust_policy_yaml)
        trust = data.get("trust", {})
        assert trust["authority_override_floor"] >= trust["floor"]

    def test_detect_invalid_override_floor(self):
        """Validator catches authority_override_floor < floor."""
        tp = (
            "trust:\n"
            "  floor: 0.50\n"
            "  authority_override_floor: 0.20\n"
        )
        artifacts = SynthesisArtifacts(
            prompt_md="test",
            constraints_yaml="constraints: []",
            trust_policy_yaml=tp,
            component_map_yaml="components: []",
        )
        warnings = validate_artifacts(artifacts)
        assert any("authority_override_floor" in w for w in warnings)


# ============================================================================
# Synthesizer: 4-delimiter parsing
# ============================================================================

class TestFourDelimiterParsing:
    def test_all_four_delimiters(self):
        raw = _make_full_raw_output()
        artifacts = parse_synthesis_output(raw)
        assert artifacts.prompt_md
        assert artifacts.constraints_yaml
        assert artifacts.trust_policy_yaml
        assert artifacts.component_map_yaml

    def test_backward_compat_two_delimiters(self):
        """Old-style output with only PROMPT and CONSTRAINTS still works."""
        raw = "--- PROMPT ---\nprompt content\n--- CONSTRAINTS ---\nconstraints content"
        artifacts = parse_synthesis_output(raw)
        assert artifacts.prompt_md == "prompt content"
        assert artifacts.constraints_yaml == "constraints content"
        assert artifacts.trust_policy_yaml == ""
        assert artifacts.component_map_yaml == ""

    def test_iteration_backward_compat(self):
        """SynthesisArtifacts supports indexing."""
        artifacts = SynthesisArtifacts("p", "c", "t", "m")
        assert artifacts[0] == "p"
        assert artifacts[1] == "c"
        assert artifacts[2] == "t"
        assert artifacts[3] == "m"

    def test_yaml_validation_catches_invalid(self):
        with pytest.raises(ValueError, match="Invalid YAML"):
            validate_yaml_content(":\n  :\n  - [invalid", "test.yaml")

    def test_yaml_validation_empty_returns_none(self):
        assert validate_yaml_content("", "test.yaml") is None


# ============================================================================
# Cross-validation
# ============================================================================

class TestCrossValidation:
    def test_trust_policy_references_unknown_component(self):
        tp = (
            "authority_map:\n"
            "  - component: ghost-service\n"
            "    domains: [x]\n"
        )
        cm = (
            "components:\n"
            "  - name: real-service\n"
        )
        artifacts = SynthesisArtifacts("", "constraints: []", tp, cm)
        warnings = validate_artifacts(artifacts)
        assert any("ghost-service" in w for w in warnings)

    def test_clean_artifacts_produce_no_warnings(self):
        raw = _make_full_raw_output()
        artifacts = parse_synthesis_output(raw)
        warnings = validate_artifacts(artifacts)
        assert warnings == [], f"Unexpected warnings: {warnings}"


# ============================================================================
# CLI: export and validate commands
# ============================================================================

class TestExportCommand:
    def test_export_baton(self):
        from constrain.cli import cmd_export
        from click.testing import CliRunner

        mock_session = Mock()
        mock_session.component_map_yaml = (
            "components:\n"
            "  - name: svc\n"
            "    type: service\n"
            "    port: 8080\n"
            "    protocol: http\n"
            "edges:\n"
            "  - from: svc\n"
            "    to: db\n"
            "    protocol: tcp\n"
        )
        mock_session.phase = Phase.complete

        mock_mgr = Mock()
        mock_mgr.list_all.return_value = [{"id": "t", "is_complete": True}]
        mock_mgr.load.return_value = mock_session

        runner = CliRunner()
        with runner.isolated_filesystem():
            with patch("constrain.cli.SessionManager", return_value=mock_mgr):
                result = runner.invoke(cmd_export, ["--format", "baton"])
            assert result.exit_code == 0
            assert Path("baton.yaml").exists()
            data = yaml.safe_load(Path("baton.yaml").read_text())
            assert data["nodes"][0]["name"] == "svc"

    def test_export_arbiter(self):
        from constrain.cli import cmd_export
        from click.testing import CliRunner

        mock_session = Mock()
        mock_session.trust_policy_yaml = (
            "trust:\n  floor: 0.10\n"
            "classifications: []\n"
            "soak: {}\n"
            "authority_map: []\n"
            "human_gates: {}\n"
        )
        mock_session.phase = Phase.complete

        mock_mgr = Mock()
        mock_mgr.list_all.return_value = [{"id": "t", "is_complete": True}]
        mock_mgr.load.return_value = mock_session

        runner = CliRunner()
        with runner.isolated_filesystem():
            with patch("constrain.cli.SessionManager", return_value=mock_mgr):
                result = runner.invoke(cmd_export, ["--format", "arbiter"])
            assert result.exit_code == 0
            assert Path("arbiter.yaml").exists()

    def test_export_pact(self):
        from constrain.cli import cmd_export
        from click.testing import CliRunner
        from constrain.models import ProblemModel

        mock_session = Mock()
        mock_session.prompt_md = "# System briefing"
        mock_session.problem_model = ProblemModel(
            system_description="Test system",
            acceptance_criteria=["Users can log in"],
        )
        mock_session.phase = Phase.complete

        mock_mgr = Mock()
        mock_mgr.list_all.return_value = [{"id": "t", "is_complete": True}]
        mock_mgr.load.return_value = mock_session

        runner = CliRunner()
        with runner.isolated_filesystem():
            with patch("constrain.cli.SessionManager", return_value=mock_mgr):
                result = runner.invoke(cmd_export, ["--format", "pact"])
            assert result.exit_code == 0
            content = Path("task.md").read_text()
            assert "Test system" in content
            assert "Users can log in" in content


class TestValidateCommand:
    def test_validate_clean_session(self):
        from constrain.cli import cmd_validate
        from click.testing import CliRunner

        raw = _make_full_raw_output()
        artifacts = parse_synthesis_output(raw)

        mock_session = Mock()
        mock_session.prompt_md = artifacts.prompt_md
        mock_session.constraints_yaml = artifacts.constraints_yaml
        mock_session.trust_policy_yaml = artifacts.trust_policy_yaml
        mock_session.component_map_yaml = artifacts.component_map_yaml
        mock_session.schema_hints_yaml = artifacts.schema_hints_yaml
        mock_session.phase = Phase.complete

        mock_mgr = Mock()
        mock_mgr.list_all.return_value = [{"id": "t", "is_complete": True}]
        mock_mgr.load.return_value = mock_session

        runner = CliRunner()
        with patch("constrain.cli.SessionManager", return_value=mock_mgr):
            result = runner.invoke(cmd_validate)

        assert result.exit_code == 0
        assert "Validation passed" in result.output


# ============================================================================
# Model: new fields
# ============================================================================

class TestModelNewFields:
    def test_session_has_trust_policy_field(self):
        from constrain.models import Session
        s = Session(posture=Posture.collaborator)
        assert hasattr(s, "trust_policy_yaml")
        assert s.trust_policy_yaml == ""

    def test_session_has_component_map_field(self):
        from constrain.models import Session
        s = Session(posture=Posture.collaborator)
        assert hasattr(s, "component_map_yaml")
        assert s.component_map_yaml == ""

    def test_constraint_has_classification_tier(self):
        from constrain.models import Constraint
        c = Constraint(
            id="C001", boundary="test", condition="test",
            violation="test", severity=Severity.must, rationale="test",
        )
        assert c.classification_tier is None

    def test_constraint_has_affected_components(self):
        from constrain.models import Constraint
        c = Constraint(
            id="C001", boundary="test", condition="test",
            violation="test", severity=Severity.must, rationale="test",
        )
        assert c.affected_components == []

    def test_severity_may(self):
        assert Severity.may.value == "may"


# ============================================================================
# Write artifacts: all four files
# ============================================================================

class TestWriteAllArtifacts:
    def test_writes_five_files(self, tmp_path):
        written = write_artifacts(
            "prompt", "constraints", tmp_path,
            trust_policy_yaml="trust: {}",
            component_map_yaml="components: []",
            schema_hints_yaml="storage_backends: []",
        )
        names = {p.name for p in written}
        assert names == {
            "prompt.md", "constraints.yaml", "trust_policy.yaml",
            "component_map.yaml", "schema_hints.yaml",
        }

    def test_skips_empty_new_artifacts(self, tmp_path):
        """Empty optional artifacts are not written (backward compat)."""
        written = write_artifacts("prompt", "constraints", tmp_path)
        names = {p.name for p in written}
        assert names == {"prompt.md", "constraints.yaml"}

    def test_overwrite_all_five(self, tmp_path):
        # Write once
        write_artifacts(
            "p1", "c1", tmp_path,
            trust_policy_yaml="t1", component_map_yaml="m1", schema_hints_yaml="s1",
        )
        # Overwrite
        written = write_artifacts(
            "p2", "c2", tmp_path, overwrite=True,
            trust_policy_yaml="t2", component_map_yaml="m2", schema_hints_yaml="s2",
        )
        assert (tmp_path / "prompt.md").read_text() == "p2"
        assert (tmp_path / "trust_policy.yaml").read_text() == "t2"
        assert (tmp_path / "schema_hints.yaml").read_text() == "s2"


# ============================================================================
# FA-C-021: Synthesize phase produces schema_hints.yaml
# ============================================================================

class TestFAC021:
    def test_parse_produces_schema_hints(self):
        raw = _make_full_raw_output()
        artifacts = parse_synthesis_output(raw)
        assert artifacts.schema_hints_yaml, "schema_hints_yaml must not be empty"

    def test_schema_hints_is_valid_yaml(self):
        raw = _make_full_raw_output()
        artifacts = parse_synthesis_output(raw)
        data = yaml.safe_load(artifacts.schema_hints_yaml)
        assert isinstance(data, dict)

    def test_write_schema_hints(self, tmp_path):
        raw = _make_full_raw_output()
        artifacts = parse_synthesis_output(raw)
        written = write_artifacts(
            artifacts.prompt_md,
            artifacts.constraints_yaml,
            tmp_path,
            trust_policy_yaml=artifacts.trust_policy_yaml,
            component_map_yaml=artifacts.component_map_yaml,
            schema_hints_yaml=artifacts.schema_hints_yaml,
        )
        names = {p.name for p in written}
        assert "schema_hints.yaml" in names
        content = (tmp_path / "schema_hints.yaml").read_text()
        assert "storage_backends" in content or "field_hints" in content

    def test_session_has_schema_hints_field(self):
        from constrain.models import Session
        s = Session(posture=Posture.collaborator)
        assert hasattr(s, "schema_hints_yaml")
        assert s.schema_hints_yaml == ""


# ============================================================================
# FA-C-022: Challenge phase includes at least one storage obligation probe
# ============================================================================

class TestFAC022:
    def test_challenge_prompt_has_storage_probes(self):
        from constrain.posture import get_system_prompt
        from constrain.models import ProblemModel

        pm = ProblemModel(system_description="test system")
        for posture in Posture:
            prompt = get_system_prompt(Phase.challenge, pm, posture)
            has_storage_probe = any(
                kw in prompt.lower()
                for kw in ["storage", "database", "erasure", "retained", "encrypted", "tokens"]
            )
            assert has_storage_probe, (
                f"Challenge prompt for {posture.value} missing storage obligation probes"
            )


# ============================================================================
# FA-C-023: schema_hints field_hints with sensitive tiers must have annotations
# ============================================================================

class TestFAC023:
    def test_sensitive_hints_have_annotations(self):
        raw = _make_full_raw_output()
        artifacts = parse_synthesis_output(raw)
        data = yaml.safe_load(artifacts.schema_hints_yaml)
        sensitive_tiers = {"PII", "FINANCIAL", "AUTH", "COMPLIANCE"}
        for hint in data.get("field_hints", []):
            if hint.get("likely_classification") in sensitive_tiers:
                assert hint.get("likely_annotations"), (
                    f"field_hint with tier '{hint['likely_classification']}' "
                    f"has no likely_annotations"
                )

    def test_public_hints_may_have_empty_annotations(self):
        raw = _make_full_raw_output()
        artifacts = parse_synthesis_output(raw)
        data = yaml.safe_load(artifacts.schema_hints_yaml)
        public_hints = [
            h for h in data.get("field_hints", [])
            if h.get("likely_classification") == "PUBLIC"
        ]
        # PUBLIC hints are allowed to have empty annotations
        for hint in public_hints:
            assert "likely_annotations" in hint  # field must exist, but can be empty

    def test_validator_catches_missing_annotations(self):
        sh = (
            "field_hints:\n"
            "  - backend_owner: svc\n"
            "    field_description: credit card number\n"
            "    likely_classification: FINANCIAL\n"
            "    likely_annotations: []\n"
            "    rationale: payment data\n"
        )
        artifacts = SynthesisArtifacts(
            prompt_md="test",
            constraints_yaml="constraints: []",
            trust_policy_yaml="trust: {}",
            component_map_yaml="components: []",
            schema_hints_yaml=sh,
        )
        warnings = validate_artifacts(artifacts)
        assert any("FINANCIAL" in w and "no likely_annotations" in w for w in warnings)


# ============================================================================
# FA-C-024: constrain export --format ledger emits valid schema_hints.yaml
# ============================================================================

class TestFAC024:
    def test_export_ledger(self):
        from constrain.cli import cmd_export
        from click.testing import CliRunner

        mock_session = Mock()
        mock_session.schema_hints_yaml = _DEFAULT_SCHEMA_HINTS
        mock_session.phase = Phase.complete

        mock_mgr = Mock()
        mock_mgr.list_all.return_value = [{"id": "t", "is_complete": True}]
        mock_mgr.load.return_value = mock_session

        runner = CliRunner()
        with runner.isolated_filesystem():
            with patch("constrain.cli.SessionManager", return_value=mock_mgr):
                result = runner.invoke(cmd_export, ["--format", "ledger"])
            assert result.exit_code == 0, result.output
            assert Path("schema_hints.yaml").exists()
            data = yaml.safe_load(Path("schema_hints.yaml").read_text())
            assert "storage_backends" in data
            assert "field_hints" in data
            assert data["generated_from"] == "constrain"

    def test_export_ledger_valid_yaml(self):
        from constrain.cli import cmd_export
        from click.testing import CliRunner

        mock_session = Mock()
        mock_session.schema_hints_yaml = _DEFAULT_SCHEMA_HINTS
        mock_session.phase = Phase.complete

        mock_mgr = Mock()
        mock_mgr.list_all.return_value = [{"id": "t", "is_complete": True}]
        mock_mgr.load.return_value = mock_session

        runner = CliRunner()
        with runner.isolated_filesystem():
            with patch("constrain.cli.SessionManager", return_value=mock_mgr):
                result = runner.invoke(cmd_export, ["--format", "ledger"])
            content = Path("schema_hints.yaml").read_text()
            # Must be valid YAML
            data = yaml.safe_load(content)
            assert isinstance(data, dict)

    def test_export_ledger_no_schema_hints(self):
        from constrain.cli import cmd_export
        from click.testing import CliRunner

        mock_session = Mock()
        mock_session.schema_hints_yaml = ""
        mock_session.phase = Phase.complete

        mock_mgr = Mock()
        mock_mgr.list_all.return_value = [{"id": "t", "is_complete": True}]
        mock_mgr.load.return_value = mock_session

        runner = CliRunner()
        with runner.isolated_filesystem():
            with patch("constrain.cli.SessionManager", return_value=mock_mgr):
                result = runner.invoke(cmd_export, ["--format", "ledger"])
            assert result.exit_code != 0
            assert "No schema_hints.yaml" in result.output
