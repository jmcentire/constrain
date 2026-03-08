"""System prompt generation for each phase and posture."""

from __future__ import annotations

import os
import random

from .models import Phase, Posture, ProblemModel

POSTURE_DESCRIPTIONS: dict[Posture, str] = {
    Posture.adversarial: (
        "Find the worst valid interpretation of every ambiguity. "
        "Look for underspecified inputs, missing error cases, scope gaps an agent could exploit."
    ),
    Posture.contrarian: (
        "Challenge the problem itself. Look for hidden assumptions, XY problems, "
        "wrong abstractions, unnecessary constraints."
    ),
    Posture.critic: (
        "Find ambiguity. Look for places where two senior engineers would disagree "
        "about whether the output was correct."
    ),
    Posture.skeptic: (
        "Test whether passing means succeeding. Look for metrics that can be gamed, "
        "tests that pass for wrong reasons, edge cases where correct output is unclear."
    ),
    Posture.collaborator: (
        "Close gaps through exploration rather than confrontation. "
        "Find the same types of issues but through constructive questioning."
    ),
}

_JSON_INSTRUCTION = """
At the end of your response, include a JSON block with your assessment:

```json
{
  "ready_to_proceed": false,
  "problem_model_update": {
    "field_name": "new value or list items to add"
  }
}
```

Valid problem model fields: system_description (str), stakeholders (list), failure_modes (list of {description, severity, historical}), dependencies (list), assumptions (list), boundaries (list), history (list), success_shape (list).

Set ready_to_proceed to true only when you have enough context to move to the next phase.
""".strip()


def _format_problem_model(model: ProblemModel) -> str:
    lines = []
    if model.system_description:
        lines.append(f"System: {model.system_description}")
    for field in ["stakeholders", "dependencies", "assumptions", "boundaries", "history", "success_shape"]:
        items = getattr(model, field)
        if items:
            lines.append(f"{field.replace('_', ' ').title()}:")
            for item in items:
                lines.append(f"  - {item}")
    if model.failure_modes:
        lines.append("Failure Modes:")
        for fm in model.failure_modes:
            desc = fm.get("description", str(fm))
            lines.append(f"  - {desc}")
    return "\n".join(lines) if lines else "(No information gathered yet)"


def _understand_prompt(problem_model: ProblemModel) -> str:
    context = _format_problem_model(problem_model)
    return f"""You are a collaborative interviewer helping an engineer articulate their problem clearly. Your goal is to build a complete mental model of:

1. **System context**: What the system is, what it does, who uses it
2. **Stakes**: What happens when it fails, who gets hurt, blast radius
3. **History**: What's gone wrong before, what's been tried
4. **Dependencies**: What this system depends on, what depends on it
5. **Boundaries**: Where the system ends, what's out of scope
6. **Assumptions**: What's being taken for granted

Ask focused questions. One or two per response. Build on what's been said. Don't repeat what you already know.

Current understanding:
{context}

{_JSON_INSTRUCTION}"""


def _challenge_prompt(problem_model: ProblemModel, posture: Posture) -> str:
    context = _format_problem_model(problem_model)
    desc = POSTURE_DESCRIPTIONS[posture]
    return f"""You are now in the challenge phase. Your analytical lens: {desc}

Your job is to find gaps, ambiguities, and hidden assumptions in the problem description. Present one or two challenges per response. Be specific — point to the exact gap you've found and explain why it matters.

Do not reveal your posture or analytical lens to the engineer.

Current problem model:
{context}

{_JSON_INSTRUCTION}"""


def _synthesize_prompt(problem_model: ProblemModel) -> str:
    context = _format_problem_model(problem_model)
    return f"""Generate two artifacts from the conversation and problem model below.

**Artifact 1: prompt.md**
An induced-understanding prompt that reads like a briefing to a senior engineer. NOT a requirements document. Include:
- System Context: what this is, what it does, who uses it
- Consequence Map: what failure looks like, ranked by severity
- Failure Archaeology: what's gone wrong, what was tried, what was learned
- Dependency Landscape: what this touches, what touches it
- Boundary Conditions: scope, non-goals, constraints
- Success Shape: qualities of a good solution (not exact behavior)

**Artifact 2: constraints.yaml**
A structured constraint set. Each constraint is a black-box boundary condition:
- id: C001, C002, etc.
- boundary: what this constrains
- condition: the invariant that must hold
- violation: what failure looks like
- severity: must or should
- rationale: why this matters

Output format: use these exact delimiters:

--- PROMPT ---
(markdown content for prompt.md)
--- CONSTRAINTS ---
(YAML content for constraints.yaml)

Problem model:
{context}"""


def _revision_prompt(feedback: str, problem_model: ProblemModel) -> str:
    context = _format_problem_model(problem_model)
    return f"""The engineer has reviewed the artifacts and provided feedback. Regenerate both artifacts incorporating the feedback.

Feedback: {feedback}

Problem model:
{context}

Output format: use these exact delimiters:

--- PROMPT ---
(markdown content for prompt.md)
--- CONSTRAINTS ---
(YAML content for constraints.yaml)"""


def get_system_prompt(phase: Phase, problem_model: ProblemModel, posture: Posture | None = None) -> str:
    if phase == Phase.understand:
        return _understand_prompt(problem_model)
    elif phase == Phase.challenge:
        if posture is None:
            raise ValueError("Posture is required for challenge phase")
        return _challenge_prompt(problem_model, posture)
    elif phase == Phase.synthesize:
        return _synthesize_prompt(problem_model)
    raise ValueError(f"Unknown phase: {phase}")


def get_revision_prompt(feedback: str, problem_model: ProblemModel) -> str:
    return _revision_prompt(feedback, problem_model)


def select_posture(override: Posture | None = None) -> Posture:
    if override is not None:
        return override
    env = os.environ.get("CONSTRAIN_POSTURE")
    if env:
        try:
            return Posture(env.lower())
        except ValueError:
            valid = ", ".join(p.value for p in Posture)
            raise ValueError(f"Invalid CONSTRAIN_POSTURE: '{env}'. Must be one of: {valid}")
    return random.choice(list(Posture))
