"""Tests for artifact archival and context loading."""

import pytest
from pathlib import Path
from unittest.mock import patch, Mock

from constrain.archive import (
    archive_artifacts,
    extract_slug,
    list_archived_sessions,
    load_archived_artifacts,
    slugify,
    _extract_slug_from_markdown,
    _extract_slug_from_yaml,
    _unique_dir,
)


# ============================================================================
# slugify
# ============================================================================

class TestSlugify:
    def test_basic(self):
        assert slugify("Auth Service") == "auth-service"

    def test_special_chars(self):
        assert slugify("user.email & tokens!") == "useremail-tokens"
        result = slugify("Hello, World! (test)")
        assert result == "hello-world-test"

    def test_max_length(self):
        result = slugify("a very long name that exceeds the limit", max_length=10)
        assert len(result) <= 10

    def test_empty(self):
        assert slugify("") == ""

    def test_whitespace_normalization(self):
        assert slugify("  multiple   spaces  ") == "multiple-spaces"

    def test_strips_hyphens(self):
        assert slugify("--leading-trailing--") == "leading-trailing"


# ============================================================================
# markdown slug extraction
# ============================================================================

class TestMarkdownSlug:
    def test_heading(self):
        content = "# Auth Service\n\nSome content."
        assert _extract_slug_from_markdown(content) == "auth-service"

    def test_heading_with_colon_system(self):
        content = "# System: Payment Gateway\n\nDetails."
        assert _extract_slug_from_markdown(content) == "payment-gateway"

    def test_heading_with_colon_task(self):
        content = "# Task: Implement OAuth2\n\nDetails."
        assert _extract_slug_from_markdown(content) == "implement-oauth2"

    def test_generic_heading_skipped(self):
        content = "# Task\n\nImplement user login flow."
        slug = _extract_slug_from_markdown(content)
        assert slug == "implement-user-login-flow"

    def test_generic_heading_with_colon_specific(self):
        content = "# System Briefing: API Gateway\n\nContent."
        slug = _extract_slug_from_markdown(content)
        assert slug == "api-gateway"

    def test_multiple_headings_first_wins(self):
        content = "# Real Heading\n\n## Section Two\n"
        assert _extract_slug_from_markdown(content) == "real-heading"

    def test_empty(self):
        assert _extract_slug_from_markdown("") == ""

    def test_no_heading_no_content(self):
        assert _extract_slug_from_markdown("   \n  \n") == ""

    def test_skips_auto_maintained(self):
        content = "# Design Document\n\n*Auto-maintained by pact.*\n\n## Status: Active\n"
        slug = _extract_slug_from_markdown(content)
        assert slug == "status-active"

    def test_content_after_generic_heading(self):
        content = "# Operating Procedures\n\n## Tech Stack\n- Python\n"
        slug = _extract_slug_from_markdown(content)
        assert slug == "tech-stack"


# ============================================================================
# YAML slug extraction
# ============================================================================

class TestYamlSlug:
    def test_system_key(self):
        content = "system: auth-gateway\nconstraints: []"
        assert _extract_slug_from_yaml(content) == "auth-gateway"

    def test_name_key(self):
        content = "name: payment-service\nversion: 1.0"
        assert _extract_slug_from_yaml(content) == "payment-service"

    def test_no_name_keys(self):
        content = "constraints:\n  - id: C001\n"
        assert _extract_slug_from_yaml(content) == ""

    def test_invalid_yaml(self):
        content = ":\n  :\n  - [bad"
        assert _extract_slug_from_yaml(content) == ""

    def test_empty(self):
        assert _extract_slug_from_yaml("") == ""

    def test_list_yaml(self):
        content = "- item1\n- item2"
        assert _extract_slug_from_yaml(content) == ""


# ============================================================================
# extract_slug (integration)
# ============================================================================

class TestExtractSlug:
    def test_markdown_file(self, tmp_path):
        p = tmp_path / "prompt.md"
        p.write_text("# System: Auth Service\n\nContent.", encoding="utf-8")
        assert extract_slug(p) == "auth-service"

    def test_yaml_file(self, tmp_path):
        p = tmp_path / "constraints.yaml"
        p.write_text("system: payment-api\nconstraints: []", encoding="utf-8")
        assert extract_slug(p) == "payment-api"

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.md"
        p.write_text("", encoding="utf-8")
        assert extract_slug(p) == ""

    def test_nonexistent_file(self, tmp_path):
        p = tmp_path / "nope.md"
        assert extract_slug(p) == ""

    def test_unknown_extension(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text('{"name": "test"}', encoding="utf-8")
        assert extract_slug(p) == ""


# ============================================================================
# _unique_dir
# ============================================================================

class TestUniqueDir:
    def test_no_collision(self, tmp_path):
        result = _unique_dir(tmp_path, "auth-service")
        assert result == tmp_path / "auth-service"
        assert not result.exists()

    def test_collision(self, tmp_path):
        (tmp_path / "auth-service").mkdir()
        result = _unique_dir(tmp_path, "auth-service")
        assert result == tmp_path / "auth-service-2"

    def test_multiple_collisions(self, tmp_path):
        (tmp_path / "slug").mkdir()
        (tmp_path / "slug-2").mkdir()
        (tmp_path / "slug-3").mkdir()
        result = _unique_dir(tmp_path, "slug")
        assert result == tmp_path / "slug-4"


# ============================================================================
# archive_artifacts
# ============================================================================

class TestArchiveArtifacts:
    def test_no_existing_files(self, tmp_path):
        archive_base = tmp_path / ".constrain" / "archive"
        subdir, archived = archive_artifacts(
            tmp_path, ["prompt.md", "constraints.yaml"], archive_base
        )
        assert subdir is None
        assert archived == []

    def test_archives_to_slug_dir(self, tmp_path):
        (tmp_path / "prompt.md").write_text("# System: Auth Service\n", encoding="utf-8")
        (tmp_path / "constraints.yaml").write_text("system: auth\nconstraints: []", encoding="utf-8")
        archive_base = tmp_path / ".constrain" / "archive"

        subdir, archived = archive_artifacts(
            tmp_path,
            ["prompt.md", "constraints.yaml"],
            archive_base,
            slug_source_priority=["prompt.md"],
        )

        assert subdir is not None
        assert subdir.name == "auth-service"
        assert len(archived) == 2
        # Files moved away from project dir
        assert not (tmp_path / "prompt.md").exists()
        assert not (tmp_path / "constraints.yaml").exists()
        # Files exist in archive
        assert (subdir / "prompt.md").exists()
        assert (subdir / "constraints.yaml").exists()

    def test_slug_fallback_to_timestamp(self, tmp_path):
        # File with no extractable slug
        (tmp_path / "data.yaml").write_text("key: value", encoding="utf-8")
        archive_base = tmp_path / "archive"

        subdir, archived = archive_artifacts(
            tmp_path, ["data.yaml"], archive_base
        )

        assert subdir is not None
        # Slug should be a timestamp like 20260327-143022
        assert len(subdir.name) >= 8  # at minimum YYYYMMDD

    def test_partial_files(self, tmp_path):
        """Only existing files are archived, missing ones are ignored."""
        (tmp_path / "prompt.md").write_text("# Test\n", encoding="utf-8")
        archive_base = tmp_path / "archive"

        subdir, archived = archive_artifacts(
            tmp_path,
            ["prompt.md", "constraints.yaml", "trust_policy.yaml"],
            archive_base,
        )

        assert len(archived) == 1
        assert archived[0][0].name == "prompt.md"

    def test_collision_increments(self, tmp_path):
        """Second archive of same slug gets -2 suffix."""
        archive_base = tmp_path / "archive"

        # First archive
        (tmp_path / "prompt.md").write_text("# Auth Service\n", encoding="utf-8")
        subdir1, _ = archive_artifacts(
            tmp_path, ["prompt.md"], archive_base
        )
        assert subdir1.name == "auth-service"

        # Create file again and archive again
        (tmp_path / "prompt.md").write_text("# Auth Service\nRevised.", encoding="utf-8")
        subdir2, _ = archive_artifacts(
            tmp_path, ["prompt.md"], archive_base
        )
        assert subdir2.name == "auth-service-2"

    def test_creates_archive_base(self, tmp_path):
        """Archive base directory is created if it doesn't exist."""
        archive_base = tmp_path / "deep" / "nested" / "archive"
        (tmp_path / "file.md").write_text("# Test\n", encoding="utf-8")

        subdir, _ = archive_artifacts(
            tmp_path, ["file.md"], archive_base
        )

        assert archive_base.exists()
        assert subdir.parent == archive_base


# ============================================================================
# list_archived_sessions
# ============================================================================

class TestListArchivedSessions:
    def test_no_archive_dir(self, tmp_path):
        assert list_archived_sessions(tmp_path / "nonexistent") == []

    def test_lists_sessions(self, tmp_path):
        archive_base = tmp_path / "archive"
        (archive_base / "auth-service").mkdir(parents=True)
        (archive_base / "auth-service" / "prompt.md").write_text("test", encoding="utf-8")
        (archive_base / "payment-api").mkdir()
        (archive_base / "payment-api" / "constraints.yaml").write_text("c", encoding="utf-8")
        (archive_base / "payment-api" / "prompt.md").write_text("p", encoding="utf-8")

        result = list_archived_sessions(archive_base)
        assert len(result) == 2
        slugs = {s["slug"] for s in result}
        assert slugs == {"auth-service", "payment-api"}


# ============================================================================
# load_archived_artifacts
# ============================================================================

class TestLoadArchivedArtifacts:
    def test_no_archive(self, tmp_path):
        assert load_archived_artifacts(tmp_path / "nonexistent") == {}

    def test_load_specific_slug(self, tmp_path):
        archive_base = tmp_path / "archive"
        (archive_base / "auth").mkdir(parents=True)
        (archive_base / "auth" / "prompt.md").write_text("# Auth", encoding="utf-8")
        (archive_base / "auth" / "constraints.yaml").write_text("c: 1", encoding="utf-8")

        result = load_archived_artifacts(archive_base, slug="auth")
        assert "prompt.md" in result
        assert result["prompt.md"] == "# Auth"
        assert "constraints.yaml" in result

    def test_load_latest(self, tmp_path):
        import time
        archive_base = tmp_path / "archive"
        (archive_base / "old").mkdir(parents=True)
        (archive_base / "old" / "prompt.md").write_text("old", encoding="utf-8")
        time.sleep(0.05)  # ensure different mtime
        (archive_base / "new").mkdir(parents=True)
        (archive_base / "new" / "prompt.md").write_text("new", encoding="utf-8")

        result = load_archived_artifacts(archive_base)
        assert result["prompt.md"] == "new"

    def test_nonexistent_slug(self, tmp_path):
        archive_base = tmp_path / "archive"
        archive_base.mkdir(parents=True)
        assert load_archived_artifacts(archive_base, slug="nope") == {}


# ============================================================================
# CLI integration
# ============================================================================

class TestCLIArchiveCommand:
    def test_archive_list_empty(self):
        from constrain.cli import cmd_archive
        from click.testing import CliRunner

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cmd_archive, ["list"])
        assert result.exit_code == 0
        assert "No archived sessions" in result.output

    def test_archive_list_with_sessions(self):
        from constrain.cli import cmd_archive
        from click.testing import CliRunner

        runner = CliRunner()
        with runner.isolated_filesystem():
            archive_dir = Path(".constrain/archive/auth-service")
            archive_dir.mkdir(parents=True)
            (archive_dir / "prompt.md").write_text("test", encoding="utf-8")

            result = runner.invoke(cmd_archive, ["list"])
        assert result.exit_code == 0
        assert "auth-service" in result.output

    def test_archive_show(self):
        from constrain.cli import cmd_archive
        from click.testing import CliRunner

        runner = CliRunner()
        with runner.isolated_filesystem():
            archive_dir = Path(".constrain/archive/auth-service")
            archive_dir.mkdir(parents=True)
            (archive_dir / "prompt.md").write_text("# Auth Service", encoding="utf-8")

            result = runner.invoke(cmd_archive, ["show", "auth-service"])
        assert result.exit_code == 0
        assert "# Auth Service" in result.output

    def test_archive_show_nonexistent(self):
        from constrain.cli import cmd_archive
        from click.testing import CliRunner

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cmd_archive, ["show", "nope"])
        assert result.exit_code != 0
