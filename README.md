# Constrain

Find the boundary between specification and intent.

Constrain is a CLI tool that interviews engineers about their problem and produces two artifacts: an **induced-understanding prompt** and a **structured constraint set**. The prompt gives an AI agent the context to develop the right solution. The constraints give an evaluator the boundary conditions that define acceptable behavior without specifying implementation.

## Install

```bash
pip install -e ".[anthropic]"    # Anthropic backend (default)
pip install -e ".[openai]"       # OpenAI-compatible backend
pip install -e ".[mcp]"          # MCP server
pip install -e ".[all]"          # Everything
```

Requires Python 3.12+ and an API key for your chosen backend.

## Usage

```bash
constrain              # Start a new session (or resume incomplete)
constrain new          # Always start fresh
constrain resume       # Resume the most recent incomplete session
constrain show         # Display artifacts from the last completed session
constrain list         # List all sessions
constrain mcp-server   # Run MCP server (stdio transport)
```

### Backend Selection

```bash
# Anthropic (default)
export ANTHROPIC_API_KEY=sk-...
constrain

# OpenAI
constrain --backend openai --model gpt-4o

# Local model via OpenAI-compatible API
CONSTRAIN_BACKEND=openai OPENAI_BASE_URL=http://localhost:11434/v1 constrain --model llama3

# Environment variables
CONSTRAIN_BACKEND=anthropic|openai
CONSTRAIN_MODEL=<model-name>
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

## MCP Server

Constrain exposes sessions and artifacts to AI assistants via MCP:

```bash
# Standalone
constrain-mcp

# Via CLI
constrain mcp-server --project-dir /path/to/project

# Register with Claude Code
claude mcp add --scope user --transport stdio constrain -- constrain-mcp
```

**Tools**: `constrain_list_sessions`, `constrain_show_session`, `constrain_show_artifacts`, `constrain_search_sessions`
**Resources**: `constrain://sessions`, `constrain://session/{id}`, `constrain://artifacts/{id}`

## Dependencies

- [Click](https://click.palletsprojects.com/) (CLI)
- [Pydantic](https://docs.pydantic.dev/) (data models)
- [PyYAML](https://pyyaml.org/) (constraint output)
- [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) (optional, default backend)
- [OpenAI SDK](https://github.com/openai/openai-python) (optional, OpenAI-compatible backend)
- [MCP](https://github.com/modelcontextprotocol/python-sdk) (optional, MCP server)

## License

[MIT](LICENSE)
