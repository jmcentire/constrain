# Constrain

Find the boundary between specification and intent.

Constrain is a CLI tool that interviews engineers about their problem and produces two artifacts: an **induced-understanding prompt** and a **structured constraint set**. The prompt gives an AI agent the context to develop the right solution. The constraints give an evaluator the boundary conditions that define acceptable behavior without specifying implementation.

## Install

```bash
pip install -e .
```

Requires Python 3.12+ and an `ANTHROPIC_API_KEY` environment variable.

## Usage

```bash
constrain              # Start a new session (or resume incomplete)
constrain new          # Always start fresh
constrain resume       # Resume the most recent incomplete session
constrain show         # Display artifacts from the last completed session
constrain list         # List all sessions
```

## How It Works

Run `constrain` from your project directory and have a conversation. The tool builds a model of your problem, challenges your description through a randomized adversarial lens, then synthesizes two output files.

### Phase 1: Understand

Collaborative interview that builds a problem model through structured questioning: system context, stakeholders, failure modes, dependencies, assumptions, boundaries. 2-10 rounds until the model is complete.

### Phase 2: Challenge

One of five analytical postures is randomly selected at session start (hidden from you):

| Posture | Lens |
|---------|------|
| **Adversarial** | Worst valid interpretation of every ambiguity |
| **Contrarian** | Challenges the problem itself — XY problems, wrong abstractions |
| **Critic** | Where two senior engineers would disagree about correctness |
| **Skeptic** | Tests that pass for wrong reasons, gameable metrics |
| **Collaborator** | Same gaps, constructive framing |

Randomized so you can't learn the pattern. Over multiple sessions, you learn to write descriptions robust to all five lenses.

### Phase 3: Synthesize

Generates two artifacts from the accumulated conversation, with one revision cycle.

## Output Artifacts

**`prompt.md`** — Induced-understanding prompt. Reads like a briefing to a senior engineer:
- System context
- Consequence map (ranked by severity)
- Failure archaeology
- Dependency landscape
- Boundary conditions
- Success shape

**`constraints.yaml`** — Structured constraint set. Each constraint is a black-box boundary condition:

```yaml
constraints:
  - id: C001
    boundary: "what this constrains"
    condition: "the invariant that must hold"
    violation: "what failure looks like"
    severity: must | should
    rationale: "why this matters"
```

Constraints are implementation-agnostic. They describe what must hold, not how to make it hold.

## Session Persistence

Sessions save to `.constrain/sessions/` in the current directory. Ctrl+C saves and exits (resumable). Ctrl+D ends the current phase early.

## Dependencies

- [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) (claude-sonnet-4)
- [Click](https://click.palletsprojects.com/) (CLI)
- [Pydantic](https://docs.pydantic.dev/) (data models)
- [PyYAML](https://pyyaml.org/) (constraint output)

## License

[MIT](LICENSE)
