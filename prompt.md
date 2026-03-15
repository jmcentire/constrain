# Constrain — System Context

## What It Is
Constrain is an interactive constraint elicitation tool. It conducts structured interviews via LLM to extract project boundaries, then synthesizes five machine-readable artifacts consumed by the Pact/Arbiter/Baton/Sentinel/Ledger stack.

## How It Works
Three-phase conversation loop:
1. **Understand** — curious posture, builds problem model
2. **Challenge** — adversarial posture, tests assumptions
3. **Synthesize** — separate conversation, generates artifacts

Each LLM response includes structured JSON (`ready_to_proceed`, `problem_model_update`).

## Key Constraints
- Session state survives Ctrl+C and crashes
- All 5 artifacts cross-validated before writing
- Sync-only (no async)
- Backend-agnostic (Anthropic/OpenAI swappable)
- MCP server is read-only

## Output Artifacts
1. prompt.md — induced-understanding briefing (Pact)
2. constraints.yaml — boundary conditions (Pact, Sentinel)
3. trust_policy.yaml — trust config (Arbiter)
4. component_map.yaml — topology (Pact, Baton)
5. schema_hints.yaml — storage hints (Ledger)

## Architecture
8 components: cli, engine, posture, synthesizer, models, session, mcp_server, backends (anthropic + openai)

## Done Checklist
- [ ] Session persistence tested (kill -9 during interview)
- [ ] All 5 artifacts generated and cross-validated
- [ ] Backend swap tested (Anthropic <-> OpenAI)
- [ ] MCP server tested (list, show, search)
- [ ] Round limits respected per phase
