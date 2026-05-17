"""Cursor agent CLI backend."""

from __future__ import annotations

from .local_agent import LocalAgentBackend

DEFAULT_MODEL = ""


class CursorBackend(LocalAgentBackend):
    def __init__(self, model: str | None = None, cwd: str | None = None, max_tokens: int = 4096) -> None:
        super().__init__(
            name="cursor",
            command="cursor-agent",
            args_template=["--print", "--model", "{model}", "{prompt}"],
            model=model,
            default_model=DEFAULT_MODEL,
            cwd=cwd,
            max_tokens=max_tokens,
        )
