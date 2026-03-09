"""Core data models for Constrain sessions."""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field


class Posture(str, enum.Enum):
    adversarial = "adversarial"
    contrarian = "contrarian"
    critic = "critic"
    skeptic = "skeptic"
    collaborator = "collaborator"


class Phase(str, enum.Enum):
    understand = "understand"
    challenge = "challenge"
    synthesize = "synthesize"
    complete = "complete"


class Severity(str, enum.Enum):
    must = "must"
    should = "should"


class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ProblemModel(BaseModel):
    system_description: str = ""
    stakeholders: list[str] = Field(default_factory=list)
    failure_modes: list[dict] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    boundaries: list[str] = Field(default_factory=list)
    history: list[str] = Field(default_factory=list)
    success_shape: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)

    def apply_update(self, update: dict) -> None:
        for key, value in update.items():
            if key not in self.model_fields:
                continue
            current = getattr(self, key)
            if isinstance(current, str):
                setattr(self, key, value)
            elif isinstance(current, list):
                items = value if isinstance(value, list) else [value]
                for item in items:
                    if isinstance(item, str) and item not in current:
                        current.append(item)
                    elif isinstance(item, dict):
                        current.append(item)


class Constraint(BaseModel):
    id: str
    boundary: str
    condition: str
    violation: str
    severity: Severity
    rationale: str


class Session(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    schema_version: int = 1
    posture: Posture
    phase: Phase = Phase.understand
    round: int = 0
    understand_rounds: int = 0
    challenge_rounds: int = 0
    conversation: list[Message] = Field(default_factory=list)
    problem_model: ProblemModel = Field(default_factory=ProblemModel)
    prompt_md: str = ""
    constraints_yaml: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()
