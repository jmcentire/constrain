"""Codex CLI backend."""

from __future__ import annotations

from .local_agent import LocalAgentBackend

DEFAULT_MODEL = "gpt-5.3-codex"


class CodexBackend(LocalAgentBackend):
    def __init__(self, model: str | None = None, cwd: str | None = None, max_tokens: int = 4096) -> None:
        super().__init__(
            name="codex",
            command="codex",
            args_template=[
                "exec",
                "--ask-for-approval",
                "never",
                "--sandbox",
                "read-only",
                "--model",
                "{model}",
                "{prompt}",
            ],
            model=model,
            default_model=DEFAULT_MODEL,
            cwd=cwd,
            max_tokens=max_tokens,
        )
