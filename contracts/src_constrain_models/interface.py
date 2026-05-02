# === Constrain Session Models (src_constrain_models) v1 ===
#  Dependencies: pydantic, datetime, uuid, enum
# Core data models for Constrain sessions, including problem modeling, conversation tracking, constraints, and session state management using Pydantic BaseModel.

# Module invariants:
#   - All Message timestamps are ISO 8601 formatted strings
#   - Session.schema_version is always 1
#   - ProblemModel list fields contain no duplicate strings (for apply_update string appends)
#   - Session.id is a UUID4 string
#   - All datetime fields (timestamp, created_at, updated_at) use UTC timezone

class Posture(Enum):
    """Enumeration of AI assistant postures for constrain sessions"""
    adversarial = "adversarial"
    contrarian = "contrarian"
    critic = "critic"
    skeptic = "skeptic"
    collaborator = "collaborator"

class Phase(Enum):
    """Enumeration of session phases in the constraint elicitation process"""
    understand = "understand"
    challenge = "challenge"
    synthesize = "synthesize"
    complete = "complete"

class Severity(Enum):
    """Enumeration of constraint severity levels"""
    must = "must"
    should = "should"

class Message:
    """A conversation message with role, content, and timestamp"""
    role: str                                # required, Message role: 'user' or 'assistant'
    content: str                             # required, Message content text
    timestamp: str = datetime.now(timezone.utc).isoformat() # optional, ISO format timestamp, defaults to current UTC time

class ProblemModel:
    """Structured representation of a problem domain with stakeholders, boundaries, and constraints"""
    system_description: str = None           # optional
    stakeholders: list[str] = []             # optional
    failure_modes: list[dict] = []           # optional
    dependencies: list[str] = []             # optional
    assumptions: list[str] = []              # optional
    boundaries: list[str] = []               # optional
    history: list[str] = []                  # optional
    success_shape: list[str] = []            # optional
    acceptance_criteria: list[str] = []      # optional

class Constraint:
    """A formal constraint specification with boundary, condition, and violation details"""
    id: str                                  # required
    boundary: str                            # required
    condition: str                           # required
    violation: str                           # required
    severity: Severity                       # required
    rationale: str                           # required

class Session:
    """Complete session state including conversation history, problem model, and metadata"""
    id: str = str(uuid4())                   # optional
    schema_version: int = 1                  # optional
    posture: Posture                         # required
    phase: Phase = Phase.understand          # optional
    round: int = 0                           # optional
    understand_rounds: int = 0               # optional
    challenge_rounds: int = 0                # optional
    conversation: list[Message] = []         # optional
    problem_model: ProblemModel = ProblemModel() # optional
    prompt_md: str = None                    # optional
    constraints_yaml: str = None             # optional
    created_at: str = datetime.now(timezone.utc).isoformat() # optional
    updated_at: str = datetime.now(timezone.utc).isoformat() # optional

def apply_update(
    self: ProblemModel,
    update: dict,
) -> None:
    """
    Mutates ProblemModel instance by applying updates from a dictionary. String fields are replaced; list fields are appended to (strings only if not already present, dicts always appended). Ignores keys not in model_fields.

    Preconditions:
      - update must be a dict

    Postconditions:
      - For each key in update that exists in self.model_fields: if current field is str, it is replaced with update[key]; if current field is list, update[key] items are appended (strings only if not duplicate, dicts always)
      - Fields not in self.model_fields are ignored
      - self is mutated in place

    Errors:
      - AttributeError (AttributeError): If getattr fails (should not occur under normal operation with valid Pydantic model)

    Side effects: Mutates the ProblemModel instance fields
    Idempotent: no
    """
    ...

def touch(
    self: Session,
) -> None:
    """
    Updates the updated_at timestamp of a Session instance to the current UTC time in ISO format.

    Postconditions:
      - self.updated_at is set to current UTC timestamp in ISO format
      - self is mutated in place

    Side effects: Mutates the Session instance updated_at field
    Idempotent: no
    """
    ...

# ── REQUIRED EXPORTS ──────────────────────────────────
# Your implementation module MUST export ALL of these names
# with EXACTLY these spellings. Tests import them by name.
# __all__ = ['Posture', 'Phase', 'Severity', 'Message', 'ProblemModel', 'Constraint', 'Session', 'apply_update', 'touch']
