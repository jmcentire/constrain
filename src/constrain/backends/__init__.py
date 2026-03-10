"""Backend abstraction for LLM providers."""

from __future__ import annotations

import os
from typing import Protocol


class BackendError(Exception):
    """Base class for backend errors."""


class BackendRateLimitError(BackendError):
    """Rate limit exceeded — retry after delay."""


class BackendTimeoutError(BackendError):
    """Request timed out."""


class BackendConnectionError(BackendError):
    """Could not connect to the backend."""


class BackendAuthError(BackendError):
    """Authentication failed — do not retry."""


class Backend(Protocol):
    """Protocol for LLM backends. Implementations must provide complete()."""

    def complete(self, system: str, messages: list[dict], max_tokens: int = 4096) -> str:
        """Send a completion request. Returns the assistant's text content."""
        ...


def create_backend(
    backend: str | None = None,
    model: str | None = None,
    **kwargs,
) -> Backend:
    """Factory: create a backend from env vars or explicit args.

    Reads CONSTRAIN_BACKEND (default: "anthropic") and CONSTRAIN_MODEL.
    """
    backend = backend or os.environ.get("CONSTRAIN_BACKEND", "anthropic")
    model = model or os.environ.get("CONSTRAIN_MODEL")

    if backend == "anthropic":
        from .anthropic import AnthropicBackend

        return AnthropicBackend(model=model, **kwargs)
    elif backend == "openai":
        from .openai import OpenAIBackend

        return OpenAIBackend(model=model, **kwargs)
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Supported: anthropic, openai. "
            "Set CONSTRAIN_BACKEND to one of these."
        )
