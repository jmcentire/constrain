from constrain.engine import ConversationEngine
from constrain.models import Phase, Posture, Session


class _Backend:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def complete(self, system, messages):
        self.calls.append((system, messages))
        if self.responses:
            return self.responses.pop(0)
        return (
            "--- PROMPT ---\n"
            "# Generated prompt\n"
            "--- CONSTRAINTS ---\n"
            "constraints: []\n"
            "--- TRUST_POLICY ---\n"
            "trust: {}\n"
            "--- COMPONENT_MAP ---\n"
            "components: []\n"
            "--- SCHEMA_HINTS ---\n"
            "field_hints: []\n"
        )


class _IO:
    def __init__(self):
        self.messages = []

    def display(self, text):
        self.messages.append(text)

    def prompt(self, prefix):
        raise EOFError


class _SessionManager:
    def __init__(self):
        self.saved = False

    def transition_phase(self, session, phase):
        session.phase = phase

    def save(self, session):
        self.saved = True


def test_synthesis_treats_cross_validation_value_error_as_non_fatal(monkeypatch):
    session = Session(posture=Posture.adversarial, phase=Phase.synthesize)
    session_mgr = _SessionManager()
    io = _IO()
    engine = ConversationEngine(session, session_mgr, backend=_Backend(), io=io)

    def fail_validation(artifacts):
        raise ValueError("component missing from constraints")

    monkeypatch.setattr("constrain.engine.validate_artifacts", fail_validation)

    engine._run_synthesis()

    assert session.phase is Phase.complete
    assert session_mgr.saved
    assert session.prompt_md == "# Generated prompt"
    assert session.constraints_yaml == "constraints: []"
    assert any(
        "Cross-validation error (non-fatal): component missing from constraints" in message
        for message in io.messages
    )


def test_synthesis_repairs_invalid_schema_hints_yaml_before_saving():
    invalid = (
        "--- PROMPT ---\n"
        "# Generated prompt\n"
        "--- CONSTRAINTS ---\n"
        "constraints: []\n"
        "--- TRUST_POLICY ---\n"
        "trust: {}\n"
        "--- COMPONENT_MAP ---\n"
        "components: []\n"
        "--- SCHEMA_HINTS ---\n"
        "field_hints:\n"
        "  - field_description: broken\n"
        "    rat\n"
    )
    repaired = (
        "--- PROMPT ---\n"
        "# Generated prompt\n"
        "--- CONSTRAINTS ---\n"
        "constraints: []\n"
        "--- TRUST_POLICY ---\n"
        "trust: {}\n"
        "--- COMPONENT_MAP ---\n"
        "components: []\n"
        "--- SCHEMA_HINTS ---\n"
        "field_hints:\n"
        "  - field_description: broken\n"
        "    rationale: repaired\n"
    )
    backend = _Backend([invalid, repaired])
    session = Session(posture=Posture.adversarial, phase=Phase.synthesize)
    session_mgr = _SessionManager()
    io = _IO()
    engine = ConversationEngine(session, session_mgr, backend=backend, io=io)

    engine._run_synthesis()

    assert len(backend.calls) == 2
    assert session.phase is Phase.complete
    assert session_mgr.saved
    assert "rationale: repaired" in session.schema_hints_yaml
    assert "    rat\n" not in session.schema_hints_yaml
    assert any("Automatic YAML repair succeeded." in message for message in io.messages)
