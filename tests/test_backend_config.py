import pytest

from constrain.backends import create_backend
from constrain.backends.local_agent import LocalAgentBackend


def test_create_backend_uses_env_max_tokens_for_local_agent(monkeypatch):
    monkeypatch.setenv("CONSTRAIN_CODEX_COMMAND", "python3")
    monkeypatch.setenv("CONSTRAIN_MAX_TOKENS", "32000")

    backend = create_backend("codex")

    assert backend.max_tokens == 32000


def test_create_backend_rejects_invalid_env_max_tokens(monkeypatch):
    monkeypatch.setenv("CONSTRAIN_MAX_TOKENS", "nope")

    with pytest.raises(ValueError, match="CONSTRAIN_MAX_TOKENS"):
        create_backend("codex")


def test_local_agent_backend_runs_configured_command():
    backend = LocalAgentBackend(
        name="testagent",
        command="python3",
        args_template=["-c", "print('agent output')"],
        max_tokens=16000,
    )

    assert backend.complete("system", [{"role": "user", "content": "hello"}]) == "agent output"
