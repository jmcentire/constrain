"""Claude Code CLI backend."""

from __future__ import annotations

from .local_agent import LocalAgentBackend

DEFAULT_MODEL = "sonnet"


class ClaudeBackend(LocalAgentBackend):
    def __init__(self, model: str | None = None, cwd: str | None = None, max_tokens: int = 4096) -> None:
        super().__init__(
            name="claude",
            command="claude",
            args_template=[
                "--print",
                "--permission-mode",
                "dontAsk",
                "--model",
                "{model}",
                "--system-prompt",
                "{system}",
                "{prompt}",
            ],
            model=model,
            default_model=DEFAULT_MODEL,
            cwd=cwd,
            max_tokens=max_tokens,
        )
