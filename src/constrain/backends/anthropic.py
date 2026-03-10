"""Anthropic (Claude) backend."""

from __future__ import annotations

from . import (
    BackendAuthError,
    BackendConnectionError,
    BackendRateLimitError,
    BackendTimeoutError,
)

DEFAULT_MODEL = "claude-sonnet-4-20250514"


class AnthropicBackend:
    """Backend using the Anthropic Python SDK."""

    def __init__(self, model: str | None = None, client=None) -> None:
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for the Anthropic backend.\n"
                "Install with: pip install constrain[anthropic]"
            )
        self._anthropic = _anthropic
        self.model = model or DEFAULT_MODEL
        self.client = client or _anthropic.Anthropic()

    def complete(self, system: str, messages: list[dict], max_tokens: int = 4096) -> str:
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
            if not resp.content:
                raise RuntimeError("API returned an empty response")
            return resp.content[0].text
        except self._anthropic.RateLimitError as e:
            raise BackendRateLimitError(str(e)) from e
        except self._anthropic.APITimeoutError as e:
            raise BackendTimeoutError(str(e)) from e
        except self._anthropic.APIConnectionError as e:
            raise BackendConnectionError(str(e)) from e
        except self._anthropic.AuthenticationError as e:
            raise BackendAuthError(str(e)) from e
