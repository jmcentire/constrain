"""Click CLI for Constrain."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click

from .backends import create_backend
from .engine import ConversationEngine, EngineConfig
from .models import Phase, Posture, Session
from .session import SessionManager
from .synthesizer import write_artifacts


class SafeGroup(click.Group):
    """Click group with centralized exception handling."""

    def invoke(self, ctx: click.Context) -> None:
        try:
            super().invoke(ctx)
        except click.ClickException:
            raise
        except click.Abort:
            click.echo("Aborted.", err=True)
            sys.exit(1)
        except KeyboardInterrupt:
            click.echo("\nInterrupted.", err=True)
            sys.exit(130)
        except SystemExit:
            raise
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)


def ensure_api_key() -> None:
    """Validate that the required API key is set for the active backend."""
    backend_name = os.environ.get("CONSTRAIN_BACKEND", "anthropic")
    if backend_name == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise click.ClickException(
                "ANTHROPIC_API_KEY not set. Export it or add it to your shell profile."
            )
    elif backend_name == "openai":
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        base = os.environ.get("OPENAI_BASE_URL", "").strip()
        if not key and not base:
            raise click.ClickException(
                "OPENAI_API_KEY not set (or set OPENAI_BASE_URL for local models)."
            )


def _round_options(f):
    """Shared round-limit options for commands that run sessions."""
    f = click.option(
        "--min-understand", type=int, default=None,
        help="Minimum rounds in understand phase (default: 2).",
    )(f)
    f = click.option(
        "--max-understand", type=int, default=None,
        help="Maximum rounds in understand phase (default: 10).",
    )(f)
    f = click.option(
        "--min-challenge", type=int, default=None,
        help="Minimum rounds in challenge phase (default: 2).",
    )(f)
    f = click.option(
        "--max-challenge", type=int, default=None,
        help="Maximum rounds in challenge phase (default: 10).",
    )(f)
    return f


def _resolve_int(cli_val: int | None, env_var: str, default: int) -> int:
    if cli_val is not None:
        return cli_val
    env = os.environ.get(env_var)
    if env:
        try:
            v = int(env)
            if v < 1:
                raise ValueError
            return v
        except ValueError:
            raise click.ClickException(
                f"Invalid value for {env_var}: expected a positive integer, got '{env}'."
            )
    return default


def resolve_config(
    min_understand: int | None,
    max_understand: int | None,
    min_challenge: int | None,
    max_challenge: int | None,
) -> EngineConfig:
    cfg = EngineConfig(
        understand_min=_resolve_int(min_understand, "CONSTRAIN_MIN_UNDERSTAND", 2),
        understand_max=_resolve_int(max_understand, "CONSTRAIN_MAX_UNDERSTAND", 10),
        challenge_min=_resolve_int(min_challenge, "CONSTRAIN_MIN_CHALLENGE", 2),
        challenge_max=_resolve_int(max_challenge, "CONSTRAIN_MAX_CHALLENGE", 10),
    )
    if cfg.understand_min > cfg.understand_max:
        raise click.ClickException(
            f"min-understand ({cfg.understand_min}) cannot exceed max-understand ({cfg.understand_max})."
        )
    if cfg.challenge_min > cfg.challenge_max:
        raise click.ClickException(
            f"min-challenge ({cfg.challenge_min}) cannot exceed max-challenge ({cfg.challenge_max})."
        )
    return cfg


def _run_engine(
    session: Session,
    mgr: SessionManager,
    config: EngineConfig,
    prime_paths: list[Path] | None = None,
    backend_name: str | None = None,
    model: str | None = None,
) -> None:
    """Create engine and run session, then write artifacts on completion."""
    backend = create_backend(backend=backend_name, model=model)
    engine = ConversationEngine(session=session, session_mgr=mgr, backend=backend, config=config)

    # Document priming: ingest files, then interactive loop
    if prime_paths or session.round == 0:
        has_paths = bool(prime_paths)
        if has_paths:
            engine.prime_interactive(initial_paths=prime_paths)
        elif session.round == 0:
            # No --prime flag, but fresh session — offer interactive priming
            if click.confirm("Prime with documents before starting?", default=False):
                engine.prime_interactive()

    engine.run_session()

    if session.phase == Phase.complete and session.prompt_md:
        cwd = Path.cwd()
        overwrite = _confirm_overwrite(cwd)
        prompt_path, constraints_path = write_artifacts(
            session.prompt_md, session.constraints_yaml, cwd, overwrite=overwrite,
        )
        click.echo(f"\nArtifacts written:")
        click.echo(f"  {prompt_path}")
        click.echo(f"  {constraints_path}")


def _confirm_overwrite(cwd: Path) -> bool:
    """Check for existing artifacts and confirm overwrite."""
    prompt = cwd / "prompt.md"
    constraints = cwd / "constraints.yaml"
    existing = [p for p in (prompt, constraints) if p.exists()]
    if not existing:
        return False  # no conflict, overwrite flag irrelevant
    names = ", ".join(p.name for p in existing)
    if not click.confirm(f"{names} already exist. Overwrite?"):
        raise click.Abort()
    return True


@click.group(cls=SafeGroup, invoke_without_command=True)
@_round_options
@click.option(
    "--prime", "-p", multiple=True, type=click.Path(exists=True),
    help="Document(s) to ingest before the interview. Repeat for multiple files.",
)
@click.option(
    "--backend", "-b", default=None,
    help="LLM backend: anthropic (default), openai. Env: CONSTRAIN_BACKEND.",
)
@click.option(
    "--model", "-m", default=None,
    help="Model name override. Env: CONSTRAIN_MODEL.",
)
@click.pass_context
def cli(ctx, min_understand, max_understand, min_challenge, max_challenge, prime, backend, model):
    """Constrain: find the boundary between specification and intent."""
    if backend:
        os.environ["CONSTRAIN_BACKEND"] = backend
    if model:
        os.environ["CONSTRAIN_MODEL"] = model

    if ctx.invoked_subcommand is not None:
        return

    ensure_api_key()
    config = resolve_config(min_understand, max_understand, min_challenge, max_challenge)
    mgr = SessionManager(Path.cwd())

    incomplete = mgr.find_latest_incomplete()
    if incomplete and not prime:
        click.echo(f"Incomplete session found: {incomplete.id[:8]}...")
        click.echo(f"  Phase: {incomplete.phase.value}")
        click.echo(
            f"  Rounds: {incomplete.understand_rounds} understand, "
            f"{incomplete.challenge_rounds} challenge"
        )
        choice = click.prompt(
            "Resume this session? [y/n/list]",
            type=click.Choice(["y", "n", "list"], case_sensitive=False),
            default="y",
        )
        if choice == "list":
            _do_list(mgr)
            return
        if choice == "y":
            _run_engine(incomplete, mgr, config, backend_name=backend, model=model)
            return

    session = mgr.create()
    mgr.save(session)
    _run_engine(session, mgr, config, prime_paths=[Path(p) for p in prime], backend_name=backend, model=model)


@cli.command("new")
@_round_options
@click.option(
    "--prime", "-p", multiple=True, type=click.Path(exists=True),
    help="Document(s) to ingest before the interview.",
)
def cmd_new(min_understand, max_understand, min_challenge, max_challenge, prime):
    """Start a new session (ignores incomplete sessions)."""
    ensure_api_key()
    config = resolve_config(min_understand, max_understand, min_challenge, max_challenge)
    backend_name = os.environ.get("CONSTRAIN_BACKEND")
    model = os.environ.get("CONSTRAIN_MODEL")
    mgr = SessionManager(Path.cwd())
    session = mgr.create()
    mgr.save(session)
    _run_engine(session, mgr, config, prime_paths=[Path(p) for p in prime], backend_name=backend_name, model=model)


@cli.command("resume")
@click.argument("session_id", required=False)
def cmd_resume(session_id):
    """Resume an incomplete session."""
    ensure_api_key()
    mgr = SessionManager(Path.cwd())

    if session_id:
        try:
            session = mgr.load(session_id)
        except FileNotFoundError:
            raise click.ClickException(f"Session '{session_id}' not found.")
        except ValueError as e:
            raise click.ClickException(f"Could not load session '{session_id}': {e}")
        if session.phase == Phase.complete:
            raise click.ClickException(
                f"Session '{session_id}' is already completed. "
                "Use 'constrain show' to view its artifacts."
            )
    else:
        session = mgr.find_latest_incomplete()
        if session is None:
            raise click.ClickException(
                "No incomplete sessions found. Use 'constrain new' to start one."
            )

    config = resolve_config(None, None, None, None)
    backend_name = os.environ.get("CONSTRAIN_BACKEND")
    model = os.environ.get("CONSTRAIN_MODEL")
    _run_engine(session, mgr, config, backend_name=backend_name, model=model)


@cli.command("show")
def cmd_show():
    """Display artifacts from the most recent completed session."""
    mgr = SessionManager(Path.cwd())
    sessions = mgr.list_all()
    completed = [s for s in sessions if s["is_complete"]]
    if not completed:
        raise click.ClickException("No completed sessions found. Use 'constrain' to start one.")

    latest = completed[0]
    session = mgr.load(latest["id"])

    if not session.prompt_md:
        raise click.ClickException(
            f"Artifacts not found in session {session.id[:8]}. They may not have been generated."
        )

    click.echo("=== prompt.md ===\n")
    click.echo(session.prompt_md)
    click.echo("\n=== constraints.yaml ===\n")
    click.echo(session.constraints_yaml)


@cli.command("list")
def cmd_list():
    """List all sessions."""
    mgr = SessionManager(Path.cwd())
    _do_list(mgr)


def _do_list(mgr: SessionManager) -> None:
    sessions = mgr.list_all()
    if not sessions:
        raise click.ClickException("No sessions found. Use 'constrain' to start one.")

    # Table header
    click.echo(f"{'ID':<10} {'PHASE':<12} {'ROUNDS':<10} {'POSTURE':<10} {'UPDATED'}")
    click.echo("-" * 65)
    for s in sessions:
        sid = s["id"][:8]
        phase = s["phase"]
        rounds = s["understand_rounds"] + s["challenge_rounds"]
        posture = "***"
        updated = s["updated_at"][:19]
        click.echo(f"{sid:<10} {phase:<12} {rounds:<10} {posture:<10} {updated}")


@cli.command("mcp-server")
@click.option("--project-dir", type=click.Path(exists=True), default=None,
              help="Project directory (default: cwd).")
def cmd_mcp_server(project_dir):
    """Run the Constrain MCP server (stdio transport)."""
    if project_dir:
        os.environ["CONSTRAIN_PROJECT_DIR"] = str(project_dir)
    from .mcp_server import main as mcp_main
    mcp_main()


def main() -> None:
    """Entry point for the constrain CLI."""
    cli()
