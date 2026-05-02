# === System Prompt Generation for Phase and Posture (src_constrain_posture) v1 ===
#  Dependencies: os, random, .models
# Generates system prompts tailored to different phases (understand, challenge, synthesize) and postures (adversarial, contrarian, critic, skeptic, collaborator) for an AI-assisted problem clarification workflow. Formats problem models into human-readable text and constructs phase-specific prompts with JSON instruction blocks.

# Module invariants:
#   - POSTURE_DESCRIPTIONS contains exactly 5 entries mapping each Posture enum value to a description string
#   - _JSON_INSTRUCTION is a constant multi-line string template included in multiple prompt types
#   - All prompt functions return non-empty strings
#   - Formatted problem models return '(No information gathered yet)' when all fields are empty

class Posture(Enum):
    """Analytical postures for challenge phase (imported from .models)"""
    adversarial = "adversarial"
    contrarian = "contrarian"
    critic = "critic"
    skeptic = "skeptic"
    collaborator = "collaborator"

class Phase(Enum):
    """Workflow phases (imported from .models)"""
    understand = "understand"
    challenge = "challenge"
    synthesize = "synthesize"

def _format_problem_model(
    model: ProblemModel,
) -> str:
    """
    Formats a ProblemModel into a human-readable multi-line string representation. Iterates through model fields (system_description, stakeholders, dependencies, assumptions, boundaries, history, success_shape, acceptance_criteria, failure_modes) and formats non-empty ones with titles and bullet points.

    Postconditions:
      - Returns '(No information gathered yet)' if all model fields are empty
      - Returns formatted multi-line string with field names as headers and items as bullet points if any field has data
      - System description appears first if present
      - Failure modes are formatted by extracting 'description' key or converting to string

    Errors:
      - AttributeError (AttributeError): If model lacks expected attributes during getattr() calls

    Side effects: none
    Idempotent: no
    """
    ...

def _understand_prompt(
    problem_model: ProblemModel,
) -> str:
    """
    Generates a system prompt for the 'understand' phase. Creates an interviewer persona that asks questions to build a complete mental model of the problem across 7 dimensions (system context, stakes, history, dependencies, boundaries, assumptions, acceptance criteria).

    Postconditions:
      - Returns a multi-line string containing interviewer instructions
      - Includes formatted problem model as 'Current understanding' section
      - Includes _JSON_INSTRUCTION template at the end

    Side effects: none
    Idempotent: no
    """
    ...

def _challenge_prompt(
    problem_model: ProblemModel,
    posture: Posture,
) -> str:
    """
    Generates a system prompt for the 'challenge' phase with a specific analytical posture. Instructs the AI to find gaps, ambiguities, and hidden assumptions using the lens defined by the given posture.

    Preconditions:
      - posture must be a valid key in POSTURE_DESCRIPTIONS dictionary

    Postconditions:
      - Returns a multi-line string with challenge phase instructions
      - Includes the posture description from POSTURE_DESCRIPTIONS
      - Includes formatted problem model as 'Current problem model' section
      - Includes _JSON_INSTRUCTION template at the end

    Errors:
      - KeyError (KeyError): If posture is not found in POSTURE_DESCRIPTIONS

    Side effects: none
    Idempotent: no
    """
    ...

def _synthesize_prompt(
    problem_model: ProblemModel,
) -> str:
    """
    Generates a system prompt for the 'synthesize' phase. Instructs the AI to generate two artifacts: prompt.md (induced-understanding briefing) and constraints.yaml (structured constraint set) using specific delimiters.

    Postconditions:
      - Returns a multi-line string with synthesis instructions
      - Specifies format for two artifacts with exact delimiters (--- PROMPT ---, --- CONSTRAINTS ---)
      - Includes formatted problem model as 'Problem model' section
      - Defines structure for prompt.md (7 sections) and constraints.yaml (constraint format)

    Side effects: none
    Idempotent: no
    """
    ...

def _revision_prompt(
    feedback: str,
    problem_model: ProblemModel,
) -> str:
    """
    Generates a prompt for artifact revision based on engineer feedback. Instructs the AI to regenerate both artifacts (prompt.md and constraints.yaml) incorporating the provided feedback.

    Postconditions:
      - Returns a multi-line string with revision instructions
      - Includes the feedback text
      - Includes formatted problem model as 'Problem model' section
      - Specifies same output format with delimiters as _synthesize_prompt

    Side effects: none
    Idempotent: no
    """
    ...

def get_system_prompt(
    phase: Phase,
    problem_model: ProblemModel,
    posture: Posture | None = None,
) -> str:
    """
    Main dispatcher function that returns the appropriate system prompt based on the current phase. Routes to _understand_prompt, _challenge_prompt, or _synthesize_prompt based on phase enum value.

    Preconditions:
      - If phase is Phase.challenge, posture must not be None

    Postconditions:
      - Returns appropriate prompt string for the given phase

    Errors:
      - ValueError_MissingPosture (ValueError): When phase is Phase.challenge and posture is None
          message: Posture is required for challenge phase
      - ValueError_UnknownPhase (ValueError): When phase is not one of Phase.understand, Phase.challenge, or Phase.synthesize
          message: Unknown phase: {phase}

    Side effects: none
    Idempotent: no
    """
    ...

def get_revision_prompt(
    feedback: str,
    problem_model: ProblemModel,
) -> str:
    """
    Public wrapper for _revision_prompt. Returns a prompt for revising artifacts based on feedback.

    Postconditions:
      - Returns revision prompt string identical to _revision_prompt output

    Side effects: none
    Idempotent: no
    """
    ...

def select_posture(
    override: Posture | None = None,
) -> Posture:
    """
    Selects an analytical posture based on priority: explicit override parameter, CONSTRAIN_POSTURE environment variable, or random selection from all Posture enum values. Validates environment variable value if present.

    Postconditions:
      - If override is not None, returns override value
      - If override is None and CONSTRAIN_POSTURE env var is set to valid posture, returns that posture
      - If override is None and CONSTRAIN_POSTURE is not set, returns a randomly selected Posture
      - Random selection uses random.choice() with uniform distribution across all Posture values

    Errors:
      - ValueError_InvalidEnvironmentPosture (ValueError): When CONSTRAIN_POSTURE environment variable is set but contains invalid posture value
          message: Invalid CONSTRAIN_POSTURE: '{env}'. Must be one of: {valid}

    Side effects: Reads CONSTRAIN_POSTURE environment variable, Uses random number generator for posture selection
    Idempotent: no
    """
    ...

# ── REQUIRED EXPORTS ──────────────────────────────────
# Your implementation module MUST export ALL of these names
# with EXACTLY these spellings. Tests import them by name.
# __all__ = ['Posture', 'Phase', '_format_problem_model', '_understand_prompt', '_challenge_prompt', '_synthesize_prompt', '_revision_prompt', 'get_system_prompt', 'get_revision_prompt', 'select_posture']
