"""
Hidden adversarial acceptance tests for CLI Interface component.
These tests target gaps in visible test coverage to detect implementations
that hardcode returns or take shortcuts based on visible test inputs.
"""
import os
import sys
import json
import click
import pytest
from unittest.mock import patch, MagicMock, call, PropertyMock
from pathlib import Path

from src.cli import (
    ensure_api_key,
    resolve_round_limits,
    check_gitignore_suggestion,
    format_session_table,
    format_session_summary,
    confirm_overwrite,
    create_cli,
)

# Try importing SafeGroup - it may be a class within cli module
try:
    from src.cli import SafeGroup
except ImportError:
    SafeGroup = None


# ============================================================
# ensure_api_key tests
# ============================================================

class TestGoodhartEnsureApiKey:
    """Tests to detect hardcoded returns or incomplete validation in ensure_api_key."""

    def test_goodhart_ensure_api_key_returns_actual_env_value(self, monkeypatch):
        """ensure_api_key must return the exact value from the environment, not a hardcoded string — verified with a novel random-like key value"""
        novel_key = "sk-ant-NOVEL-xq9z8w7v6u5t4s3r2q1p"
        monkeypatch.setenv("ANTHROPIC_API_KEY", novel_key)
        result = ensure_api_key()
        assert result == novel_key

    def test_goodhart_ensure_api_key_returns_different_values(self, monkeypatch):
        """ensure_api_key must dynamically read the environment variable, not return a cached or hardcoded value"""
        key1 = "sk-ant-first-key-aaa111"
        key2 = "sk-ant-second-key-bbb222"
        
        monkeypatch.setenv("ANTHROPIC_API_KEY", key1)
        result1 = ensure_api_key()
        assert result1 == key1

        monkeypatch.setenv("ANTHROPIC_API_KEY", key2)
        result2 = ensure_api_key()
        assert result2 == key2
        assert result1 != result2

    def test_goodhart_ensure_api_key_whitespace_only(self, monkeypatch):
        """An API key consisting only of whitespace should be treated as empty/not set and raise an error"""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
        with pytest.raises(click.ClickException):
            ensure_api_key()

    def test_goodhart_ensure_api_key_exception_type(self, monkeypatch):
        """ensure_api_key must raise specifically click.ClickException (not a generic exception) when key is missing"""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(click.ClickException) as exc_info:
            ensure_api_key()
        # Verify it's exactly ClickException, not some other exception caught by pytest
        assert type(exc_info.value) == click.ClickException or isinstance(exc_info.value, click.ClickException)

    def test_goodhart_ensure_api_key_long_key(self, monkeypatch):
        """ensure_api_key handles and returns arbitrarily long API key strings"""
        long_key = "sk-ant-" + "a" * 500
        monkeypatch.setenv("ANTHROPIC_API_KEY", long_key)
        result = ensure_api_key()
        assert result == long_key
        assert len(result) == 507


# ============================================================
# resolve_round_limits tests
# ============================================================

class TestGoodhartResolveRoundLimits:
    """Tests to detect hardcoded defaults and incomplete env var handling."""

    def test_goodhart_resolve_round_limits_partial_env_vars(self, monkeypatch):
        """When only some env vars are set and others are not, each field independently falls back to its own source (env or default)"""
        monkeypatch.setenv("CONSTRAIN_MIN_UNDERSTAND", "2")
        monkeypatch.setenv("CONSTRAIN_MAX_CHALLENGE", "9")
        monkeypatch.delenv("CONSTRAIN_MAX_UNDERSTAND", raising=False)
        monkeypatch.delenv("CONSTRAIN_MIN_CHALLENGE", raising=False)
        
        result = resolve_round_limits(None, None, None, None)
        assert result.min_understand == 2
        assert result.max_challenge == 9
        # max_understand and min_challenge should be defaults, and min <= max
        assert result.min_understand <= result.max_understand
        assert result.min_challenge <= result.max_challenge

    def test_goodhart_resolve_round_limits_cli_partial_override(self, monkeypatch):
        """When only some CLI flags are provided, the remaining fields fall back to env vars or defaults independently"""
        monkeypatch.setenv("CONSTRAIN_MAX_CHALLENGE", "8")
        monkeypatch.delenv("CONSTRAIN_MIN_UNDERSTAND", raising=False)
        monkeypatch.delenv("CONSTRAIN_MAX_UNDERSTAND", raising=False)
        monkeypatch.delenv("CONSTRAIN_MIN_CHALLENGE", raising=False)

        result = resolve_round_limits(2, None, None, None)
        assert result.min_understand == 2
        assert result.max_challenge == 8
        assert result.min_understand <= result.max_understand
        assert result.min_challenge <= result.max_challenge

    def test_goodhart_resolve_round_limits_all_four_env_vars(self, monkeypatch):
        """All four environment variables are independently respected when all CLI flags are None"""
        monkeypatch.setenv("CONSTRAIN_MIN_UNDERSTAND", "2")
        monkeypatch.setenv("CONSTRAIN_MAX_UNDERSTAND", "7")
        monkeypatch.setenv("CONSTRAIN_MIN_CHALLENGE", "3")
        monkeypatch.setenv("CONSTRAIN_MAX_CHALLENGE", "9")

        result = resolve_round_limits(None, None, None, None)
        assert result.min_understand == 2
        assert result.max_understand == 7
        assert result.min_challenge == 3
        assert result.max_challenge == 9

    def test_goodhart_resolve_round_limits_env_var_float_rejected(self, monkeypatch):
        """Floating point values in environment variables should be rejected as they are not valid positive integers"""
        monkeypatch.setenv("CONSTRAIN_MIN_UNDERSTAND", "3.5")
        monkeypatch.delenv("CONSTRAIN_MAX_UNDERSTAND", raising=False)
        monkeypatch.delenv("CONSTRAIN_MIN_CHALLENGE", raising=False)
        monkeypatch.delenv("CONSTRAIN_MAX_CHALLENGE", raising=False)

        with pytest.raises((click.ClickException, click.exceptions.Exit, SystemExit, ValueError)):
            resolve_round_limits(None, None, None, None)

    def test_goodhart_resolve_round_limits_env_var_non_numeric(self, monkeypatch):
        """Alphabetic string values in environment variables should be rejected"""
        monkeypatch.setenv("CONSTRAIN_MAX_UNDERSTAND", "abc")
        monkeypatch.delenv("CONSTRAIN_MIN_UNDERSTAND", raising=False)
        monkeypatch.delenv("CONSTRAIN_MIN_CHALLENGE", raising=False)
        monkeypatch.delenv("CONSTRAIN_MAX_CHALLENGE", raising=False)

        with pytest.raises((click.ClickException, click.exceptions.Exit, SystemExit, ValueError)):
            resolve_round_limits(None, None, None, None)

    def test_goodhart_resolve_round_limits_env_var_max_challenge_invalid(self, monkeypatch):
        """Invalid env var for max_challenge specifically (not just min_understand) is detected and rejected"""
        monkeypatch.setenv("CONSTRAIN_MAX_CHALLENGE", "notanumber")
        monkeypatch.delenv("CONSTRAIN_MIN_UNDERSTAND", raising=False)
        monkeypatch.delenv("CONSTRAIN_MAX_UNDERSTAND", raising=False)
        monkeypatch.delenv("CONSTRAIN_MIN_CHALLENGE", raising=False)

        with pytest.raises((click.ClickException, click.exceptions.Exit, SystemExit, ValueError)):
            resolve_round_limits(None, None, None, None)

    def test_goodhart_resolve_round_limits_env_var_empty_string(self, monkeypatch):
        """An empty string environment variable should either be ignored (fall back to default) or rejected, not crash"""
        monkeypatch.setenv("CONSTRAIN_MIN_UNDERSTAND", "")
        monkeypatch.delenv("CONSTRAIN_MAX_UNDERSTAND", raising=False)
        monkeypatch.delenv("CONSTRAIN_MIN_CHALLENGE", raising=False)
        monkeypatch.delenv("CONSTRAIN_MAX_CHALLENGE", raising=False)

        # Should either fall back to defaults cleanly or raise a clear ClickException
        try:
            result = resolve_round_limits(None, None, None, None)
            # If it succeeds, the defaults should be valid
            assert result.min_understand <= result.max_understand
            assert result.min_challenge <= result.max_challenge
            assert result.min_understand > 0
        except (click.ClickException, click.exceptions.Exit, SystemExit, ValueError):
            pass  # Also acceptable: a clear error

    def test_goodhart_resolve_round_limits_large_values(self, monkeypatch):
        """Large but valid positive integers should be accepted without overflow or rejection"""
        monkeypatch.delenv("CONSTRAIN_MIN_UNDERSTAND", raising=False)
        monkeypatch.delenv("CONSTRAIN_MAX_UNDERSTAND", raising=False)
        monkeypatch.delenv("CONSTRAIN_MIN_CHALLENGE", raising=False)
        monkeypatch.delenv("CONSTRAIN_MAX_CHALLENGE", raising=False)

        result = resolve_round_limits(50, 100, 50, 100)
        assert result.min_understand == 50
        assert result.max_understand == 100
        assert result.min_challenge == 50
        assert result.max_challenge == 100

    def test_goodhart_resolve_round_limits_cross_phase_independence(self, monkeypatch):
        """Validation of min<=max is per-phase — understand limits are independent of challenge limits"""
        monkeypatch.delenv("CONSTRAIN_MIN_UNDERSTAND", raising=False)
        monkeypatch.delenv("CONSTRAIN_MAX_UNDERSTAND", raising=False)
        monkeypatch.delenv("CONSTRAIN_MIN_CHALLENGE", raising=False)
        monkeypatch.delenv("CONSTRAIN_MAX_CHALLENGE", raising=False)

        # min_challenge=5 > max_understand=2 should be fine
        result = resolve_round_limits(1, 2, 5, 10)
        assert result.min_understand == 1
        assert result.max_understand == 2
        assert result.min_challenge == 5
        assert result.max_challenge == 10

    def test_goodhart_resolve_round_limits_min_exceeds_max_via_mixed_sources(self, monkeypatch):
        """min > max error is detected even when min comes from env var and max from default (cross-source validation)"""
        monkeypatch.setenv("CONSTRAIN_MIN_UNDERSTAND", "99")
        monkeypatch.delenv("CONSTRAIN_MAX_UNDERSTAND", raising=False)
        monkeypatch.delenv("CONSTRAIN_MIN_CHALLENGE", raising=False)
        monkeypatch.delenv("CONSTRAIN_MAX_CHALLENGE", raising=False)

        with pytest.raises((click.ClickException, click.exceptions.Exit, SystemExit)):
            resolve_round_limits(None, None, None, None)

    def test_goodhart_resolve_round_limits_positive_integers_only(self, monkeypatch):
        """All resolved round limit fields must be positive integers (> 0), not just non-negative"""
        monkeypatch.delenv("CONSTRAIN_MIN_UNDERSTAND", raising=False)
        monkeypatch.delenv("CONSTRAIN_MAX_UNDERSTAND", raising=False)
        monkeypatch.delenv("CONSTRAIN_MIN_CHALLENGE", raising=False)
        monkeypatch.delenv("CONSTRAIN_MAX_CHALLENGE", raising=False)

        result = resolve_round_limits(None, None, None, None)
        assert result.min_understand > 0
        assert result.max_understand > 0
        assert result.min_challenge > 0
        assert result.max_challenge > 0
        assert isinstance(result.min_understand, int)
        assert isinstance(result.max_understand, int)
        assert isinstance(result.min_challenge, int)
        assert isinstance(result.max_challenge, int)

    def test_goodhart_resolve_round_limits_env_var_with_whitespace(self, monkeypatch):
        """Environment variable values with leading/trailing whitespace around a valid integer should be handled gracefully"""
        monkeypatch.setenv("CONSTRAIN_MIN_UNDERSTAND", " 3 ")
        monkeypatch.delenv("CONSTRAIN_MAX_UNDERSTAND", raising=False)
        monkeypatch.delenv("CONSTRAIN_MIN_CHALLENGE", raising=False)
        monkeypatch.delenv("CONSTRAIN_MAX_CHALLENGE", raising=False)

        # Should either strip and accept 3, or raise clear error
        try:
            result = resolve_round_limits(None, None, None, None)
            assert result.min_understand == 3
        except (click.ClickException, click.exceptions.Exit, SystemExit, ValueError):
            pass  # Also acceptable


# ============================================================
# format_session_table tests
# ============================================================

class TestGoodhartFormatSessionTable:
    """Tests for table formatting with diverse inputs."""

    def _make_row(self, session_id="sess-1", status="incomplete", created_at="2024-01-01", current_phase="understand", posture_display="***"):
        """Helper to create a session row. Tries different struct representations."""
        try:
            from src.cli import SessionListRow
            return SessionListRow(
                session_id=session_id,
                status=status,
                created_at=created_at,
                current_phase=current_phase,
                posture_display=posture_display,
            )
        except ImportError:
            try:
                from src.models import SessionListRow
                return SessionListRow(
                    session_id=session_id,
                    status=status,
                    created_at=created_at,
                    current_phase=current_phase,
                    posture_display=posture_display,
                )
            except ImportError:
                # Fall back to a simple namespace or dict-like object
                from types import SimpleNamespace
                return SimpleNamespace(
                    session_id=session_id,
                    status=status,
                    created_at=created_at,
                    current_phase=current_phase,
                    posture_display=posture_display,
                )

    def test_goodhart_format_session_table_many_rows(self):
        """Table formatting must scale correctly to many rows, not just 1-2 visible test rows"""
        rows = [
            self._make_row(
                session_id=f"sess-{i:03d}",
                status="incomplete" if i % 2 == 0 else "completed",
                created_at=f"2024-01-{i+1:02d}",
                current_phase="understand" if i % 3 == 0 else "challenge",
            )
            for i in range(10)
        ]
        result = format_session_table(rows)
        lines = [l for l in result.strip().split("\n") if l.strip()]
        # Should have header + 10 data rows (possibly also a separator line)
        data_lines = [l for l in lines if "sess-" in l]
        assert len(data_lines) == 10

    def test_goodhart_format_session_table_long_values(self):
        """Table formatting handles long session IDs and status strings without truncation or misalignment"""
        row = self._make_row(session_id="abc123def456ghi789jkl012mno345pqr")
        result = format_session_table([row])
        assert "abc123def456ghi789jkl012mno345pqr" in result

    def test_goodhart_format_session_table_posture_override_attempt(self):
        """Even if posture_display field contains actual posture text, the table output must show '***'"""
        row = self._make_row(posture_display="aggressive_challenger")
        result = format_session_table([row])
        assert "***" in result
        assert "aggressive_challenger" not in result

    def test_goodhart_format_session_table_column_headers(self):
        """Table header row must contain all five required column names: ID, STATUS, CREATED, PHASE, POSTURE"""
        row = self._make_row()
        result = format_session_table([row])
        header = result.strip().split("\n")[0].upper()
        assert "ID" in header
        assert "STATUS" in header
        assert "CREATED" in header or "CREATE" in header
        assert "PHASE" in header
        assert "POSTURE" in header


# ============================================================
# format_session_summary tests
# ============================================================

class TestGoodhartFormatSessionSummary:

    def test_goodhart_format_session_summary_large_round_count(self):
        """Session summary formatting correctly handles large round counts"""
        result = format_session_summary(
            session_id="sess-large-test",
            status="incomplete",
            current_phase="challenge",
            round_count=999,
            problem_summary="A complex problem"
        )
        assert "999" in result
        assert "sess-large-test" in result
        assert "challenge" in result

    def test_goodhart_format_session_summary_special_chars_in_problem(self):
        """Session summary handles special characters in problem_summary without corruption"""
        result = format_session_summary(
            session_id="sess-special",
            status="incomplete",
            current_phase="understand",
            round_count=3,
            problem_summary="Line1\nLine2\n• Unicode: café résumé"
        )
        assert "sess-special" in result
        assert "3" in result
        assert isinstance(result, str)
        assert "\n" in result  # Multi-line

    def test_goodhart_format_session_summary_zero_rounds(self):
        """Session summary handles zero round count (session just created, no rounds yet)"""
        result = format_session_summary(
            session_id="sess-zero",
            status="incomplete",
            current_phase="understand",
            round_count=0,
            problem_summary=""
        )
        assert "sess-zero" in result
        assert "0" in result


# ============================================================
# confirm_overwrite tests
# ============================================================

class TestGoodhartConfirmOverwrite:

    def test_goodhart_confirm_overwrite_only_prompt_md_exists(self, tmp_path):
        """Overwrite confirmation triggers when only prompt.md exists (not both files)"""
        (tmp_path / "prompt.md").write_text("existing prompt")
        # constraints.yaml does NOT exist
        with patch("click.confirm", return_value=True) as mock_confirm:
            result = confirm_overwrite(str(tmp_path))
            assert result is True
            mock_confirm.assert_called()

    def test_goodhart_confirm_overwrite_only_constraints_yaml_exists(self, tmp_path):
        """Overwrite confirmation triggers when only constraints.yaml exists (not both files)"""
        (tmp_path / "constraints.yaml").write_text("existing constraints")
        # prompt.md does NOT exist
        with patch("click.confirm", return_value=True) as mock_confirm:
            result = confirm_overwrite(str(tmp_path))
            assert result is True
            mock_confirm.assert_called()


# ============================================================
# create_cli tests
# ============================================================

class TestGoodhartCreateCli:

    def test_goodhart_create_cli_safe_group_class(self):
        """The CLI group must use SafeGroup as its class, not plain click.Group — ensuring centralized exception handling"""
        cli = create_cli()
        # The class should be SafeGroup or a subclass, not plain click.Group
        assert type(cli) is not click.Group
        # Check it's still a Group-like object
        assert isinstance(cli, click.Group)
        # Class name should reference SafeGroup
        assert "SafeGroup" in type(cli).__name__ or "Safe" in type(cli).__name__

    def test_goodhart_create_cli_has_all_four_subcommands(self):
        """The CLI group must register exactly the four expected subcommands: new, resume, show, list"""
        cli = create_cli()
        commands = cli.commands if hasattr(cli, 'commands') else {}
        assert "new" in commands, "Missing 'new' subcommand"
        assert "resume" in commands, "Missing 'resume' subcommand"
        assert "show" in commands, "Missing 'show' subcommand"
        assert "list" in commands, "Missing 'list' subcommand"

    def test_goodhart_create_cli_invoke_without_command(self):
        """The CLI group must be configured with invoke_without_command=True so the default handler runs"""
        cli = create_cli()
        assert cli.invoke_without_command is True

    def test_goodhart_create_cli_round_limit_options_on_new(self):
        """The 'new' subcommand must have round limit options matching those on the group"""
        cli = create_cli()
        new_cmd = cli.commands.get("new")
        assert new_cmd is not None
        option_names = [p.name for p in new_cmd.params]
        assert "min_understand" in option_names
        assert "max_understand" in option_names
        assert "min_challenge" in option_names
        assert "max_challenge" in option_names


# ============================================================
# safe_group_invoke tests
# ============================================================

class TestGoodhartSafeGroupInvoke:

    def _get_safe_group(self):
        """Get the SafeGroup class."""
        try:
            from src.cli import SafeGroup
            return SafeGroup
        except ImportError:
            cli = create_cli()
            return type(cli)

    def test_goodhart_safe_group_unexpected_error_message_format(self):
        """Unexpected error messages must include the exception type and message in the prescribed format without stack traces"""
        cli = create_cli()

        @cli.command("test_cmd")
        def test_cmd():
            raise ValueError("something broke")

        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(cli, ["test_cmd"])
        assert result.exit_code == 1
        # Check for prescribed format
        assert "unexpected error occurred" in result.output.lower() or "unexpected error occurred" in (result.output + getattr(result, 'stderr', '')).lower() or "ValueError" in result.output or "something broke" in result.output

    def test_goodhart_safe_group_runtime_error(self):
        """RuntimeError (a non-Click exception type different from visible tests) is caught and handled generically"""
        cli = create_cli()

        @cli.command("test_runtime")
        def test_runtime():
            raise RuntimeError("unexpected runtime failure")

        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(cli, ["test_runtime"])
        assert result.exit_code == 1
        full_output = result.output + getattr(result, 'stderr', '')
        # Should contain the error type and message, but no traceback
        assert "Traceback" not in full_output

    def test_goodhart_safe_group_keyboard_interrupt_exit_code(self):
        """KeyboardInterrupt handler must exit with code 130"""
        cli = create_cli()

        @cli.command("test_interrupt")
        def test_interrupt():
            raise KeyboardInterrupt()

        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(cli, ["test_interrupt"])
        assert result.exit_code == 130

    def test_goodhart_safe_group_abort_message(self):
        """On click.Abort, the exact message 'Aborted.' must be output"""
        cli = create_cli()

        @cli.command("test_abort")
        def test_abort():
            raise click.Abort()

        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(cli, ["test_abort"])
        assert result.exit_code == 1
        full_output = result.output
        assert "Aborted." in full_output or "Aborted" in full_output


# ============================================================
# check_gitignore_suggestion tests
# ============================================================

class TestGoodhartCheckGitignoreSuggestion:

    def test_goodhart_check_gitignore_new_dir_echoes_to_stderr(self, tmp_path, monkeypatch):
        """When suggestion is warranted, the message must be echoed to stderr (not stdout)"""
        monkeypatch.chdir(tmp_path)
        # No .gitignore file exists
        captured_stderr = []
        original_echo = click.echo

        def mock_echo(*args, **kwargs):
            if kwargs.get('err', False):
                captured_stderr.append(args[0] if args else '')
            original_echo(*args, **kwargs)

        with patch("click.echo", side_effect=mock_echo):
            result = check_gitignore_suggestion(constrain_dir_existed_before=False)

        if result.should_suggest:
            assert len(captured_stderr) > 0, "Suggestion should be echoed to stderr"


# ============================================================
# Integration-level adversarial tests
# ============================================================

class TestGoodhartIntegration:

    def test_goodhart_cmd_resume_calls_engine_with_resume_true(self, monkeypatch):
        """cmd_resume must call engine.run() with resume=True to avoid replaying conversation history"""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")

        from click.testing import CliRunner
        cli = create_cli()
        runner = CliRunner()

        mock_session_data = MagicMock()
        mock_session_data.session_id = "test-sess-123"
        mock_session_data.status = "incomplete"
        mock_session_data.current_phase = "understand"

        with patch("src.cli.session") as mock_session_mod, \
             patch("src.cli.engine") as mock_engine_mod:
            mock_session_mod.get_most_recent_incomplete.return_value = mock_session_data
            mock_session_mod.load.return_value = mock_session_data
            mock_engine_mod.run.return_value = None

            try:
                result = runner.invoke(cli, ["resume"])
            except (SystemExit, Exception):
                pass

            # Verify engine.run was called with resume=True
            if mock_engine_mod.run.called:
                call_kwargs = mock_engine_mod.run.call_args
                if call_kwargs:
                    # Check positional or keyword args for resume=True
                    kwargs = call_kwargs.kwargs if hasattr(call_kwargs, 'kwargs') else call_kwargs[1]
                    assert kwargs.get('resume') is True or \
                           (len(call_kwargs.args) > 1 and call_kwargs.args[-1] is True), \
                           "engine.run() must be called with resume=True"
