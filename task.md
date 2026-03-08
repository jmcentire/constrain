# Constrain

A CLI tool that interviews engineers about their problem and produces two artifacts: an induced-understanding prompt and a structured constraint set. The prompt gives an AI agent the context to develop the right attractor. The constraint set gives an evaluator the boundary conditions that define acceptable behavior without specifying implementation.

## What It Does

The engineer runs `constrain` from their project directory and enters a conversation. Constrain asks questions, builds a model of the problem, challenges the engineer's description through a randomized adversarial lens, then synthesizes two output files.

### Session Flow

Every session has three phases:

**Phase 1: Understand (always collaborative)**
Build the problem model through structured questioning:
- What is the system? What does it do?
- Who depends on it? What are the downstream consequences of failure?
- What has gone wrong before? What was tried?
- What does this system depend on? What are the integration boundaries?
- What are you assuming that you haven't stated?
- What's in scope and what's out?

The tool asks 3-7 rounds of questions. Each round processes the engineer's response, updates the internal problem model, and generates the next question based on what's still unknown. The tool decides when it has enough context to move to the challenge phase.

**Phase 2: Challenge (randomized posture)**
At session start, one of five postures is randomly selected (hidden from the engineer). The posture determines what the tool looks for during the challenge phase:

- **Adversarial**: Finds the worst valid interpretation of every ambiguity. Looks for underspecified inputs, missing error cases, scope gaps an agent could exploit.
- **Contrarian**: Challenges the problem itself. Looks for hidden assumptions, XY problems, wrong abstractions, unnecessary constraints.
- **Critic**: Finds ambiguity. Looks for places where two senior engineers would disagree about whether the output was correct, subjective criteria masquerading as objective.
- **Skeptic**: Tests whether passing means succeeding. Looks for metrics that can be gamed, tests that pass for wrong reasons, edge cases where correct output is unclear.
- **Collaborator**: Closes gaps through exploration rather than confrontation. Finds the same types of issues but through constructive questioning.

The posture is randomized so the engineer cannot learn the pattern and write descriptions that satisfy a specific challenge style. Over multiple sessions, they learn to write descriptions robust to all five lenses.

2-5 rounds of challenge. Each round presents a gap or concern, the engineer responds, the problem model is updated.

**Phase 3: Synthesize**
Generate two artifacts from the accumulated conversation and problem model. Present them to the engineer for review. Allow one revision cycle.

### Output Artifacts

**prompt.md** — The induced-understanding prompt. NOT a requirements document. Reads like a briefing to a senior engineer:
- System context: what this is, what it does, who uses it
- Consequence map: what failure looks like, ranked by severity
- Failure archaeology: what's gone wrong, what was tried, what was learned
- Dependency landscape: what this touches, what touches it
- Boundary conditions: scope, non-goals, constraints
- Success shape: the qualities of a good solution (not exact behavior)

**constraints.yaml** — Structured constraint set. Each constraint is a black-box boundary condition:
```yaml
constraints:
  - id: C001
    boundary: "what this constrains (e.g., 'file upload endpoint')"
    condition: "the invariant that must hold (e.g., 'files > 10MB rejected before processing begins')"
    violation: "what failure looks like (e.g., 'large file consumed memory and crashed worker')"
    severity: must | should
    rationale: "why this matters (e.g., 'production incident 2024-03, cost 4 hours downtime')"
```

Constraints are implementation-agnostic. They describe what must hold, not how to make it hold.

## CLI Interface

```
constrain              Start new session (or offer to resume if incomplete session exists)
constrain new          Always start a new session
constrain resume       Resume the most recent incomplete session
constrain show         Display artifacts from the most recent completed session
constrain list         List all sessions with status and date
```

### Session Persistence

Sessions are saved to `.constrain/sessions/<session-id>.json` in the current directory. Each session contains:
- Session ID (UUID)
- Posture (stored but not displayed)
- Current phase
- Conversation history (all messages)
- Problem model (accumulated understanding)
- Timestamps (created, last updated)
- Output artifacts (once synthesized)

### Terminal I/O

The conversation runs in the terminal. The tool's questions and challenges are printed. The engineer types responses. Standard readline support for editing input. Ctrl+C gracefully saves and exits (resumable). Ctrl+D ends the current phase early.

The tool should clearly indicate phase transitions (e.g., "--- Moving to challenge phase ---") but should NOT reveal the posture.

## Technical Requirements

- Python 3.12+
- Anthropic SDK (claude-sonnet-4-20250514 for conversation, claude-sonnet-4-20250514 for synthesis)
- pydantic for data models
- pyyaml for constraint output
- click for CLI
- No other dependencies

The ANTHROPIC_API_KEY is read from the environment (standard Anthropic SDK behavior).

## Architecture Guidance

The core is a multi-turn conversation with Claude, managed through system prompts that change per phase. The tool is the orchestrator; the LLM does the heavy lifting.

Key components:
1. **Session**: State management (create, save, load, resume)
2. **Engine**: Conversation loop (send message to Claude, get response, update state)
3. **Posture**: System prompt generation per posture and phase
4. **Synthesizer**: Artifact generation from conversation history
5. **CLI**: Click-based command interface

The system prompt for each phase should:
- Phase 1 (Understand): Instruct the LLM to be a collaborative interviewer building a problem model. Provide the interview structure (system, stakes, history, dependencies, boundaries, assumptions). Ask the LLM to output a JSON block at the end of each response indicating whether it has enough context to proceed.
- Phase 2 (Challenge): Inject the posture-specific lens. Instruct the LLM to find gaps through that lens. Provide the accumulated problem model as context.
- Phase 3 (Synthesize): Provide the full conversation history and problem model. Instruct the LLM to generate the two artifacts in the specified formats.

## Success Criteria

1. Running `constrain` starts an interactive session that asks meaningful, contextual questions
2. The challenge phase finds gaps the engineer didn't anticipate
3. The output prompt reads like a briefing, not a spec
4. The output constraints are black-box testable
5. Sessions persist and resume correctly
6. Different postures produce meaningfully different challenge patterns
7. The tool works end-to-end on a real engineering problem in under 15 minutes
