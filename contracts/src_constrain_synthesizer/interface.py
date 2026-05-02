# === Synthesizer Artifact Generator (src_constrain_synthesizer) v1 ===
#  Dependencies: re, pathlib
# Artifact generation module for parsing LLM synthesis output and writing prompt and constraint files to disk. Parses delimited text sections and manages file I/O with overwrite protection.

# Module invariants:
#   - prompt_marker constant: '--- PROMPT ---'
#   - constraints_marker constant: '--- CONSTRAINTS ---'
#   - Output filenames are fixed: 'prompt.md' and 'constraints.yaml'
#   - File encoding is always UTF-8

Path = primitive  # pathlib.Path type for filesystem paths

def parse_synthesis_output(
    raw: str,
) -> tuple[str, str]:
    """
    Parse '--- PROMPT ---' and '--- CONSTRAINTS ---' delimiters from LLM output. Extracts the content between the two markers, returning prompt content and constraints content as separate strings.

    Preconditions:
      - raw must contain '--- PROMPT ---' marker
      - raw must contain '--- CONSTRAINTS ---' marker

    Postconditions:
      - Returns tuple of (prompt_md, constraints_yaml)
      - prompt_md is the stripped text between '--- PROMPT ---' and '--- CONSTRAINTS ---'
      - constraints_yaml is the stripped text after '--- CONSTRAINTS ---'

    Errors:
      - missing_delimiters (ValueError): prompt_idx == -1 or constraints_idx == -1
          message: Synthesis output missing required delimiters ('--- PROMPT ---' and/or '--- CONSTRAINTS ---').

    Side effects: none
    Idempotent: no
    """
    ...

def write_artifacts(
    prompt_md: str,
    constraints_yaml: str,
    output_dir: str | Path,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """
    Write prompt.md and constraints.yaml to output_dir. Creates the output directory if it doesn't exist and writes two files with UTF-8 encoding. Provides overwrite protection by default.

    Preconditions:
      - output_dir must be a valid path string or Path object
      - If overwrite=False, prompt.md and constraints.yaml must not exist in output_dir

    Postconditions:
      - output_dir exists as a directory
      - prompt.md is written to output_dir with prompt_md content in UTF-8 encoding
      - constraints.yaml is written to output_dir with constraints_yaml content in UTF-8 encoding
      - Returns tuple of (prompt_path, constraints_path) as Path objects

    Errors:
      - file_exists (FileExistsError): existing files found and overwrite=False
          message: Artifact(s) already exist: {names}. Use overwrite=True to replace.

    Side effects: Creates output directory if it doesn't exist (including parent directories), Writes prompt.md file to disk, Writes constraints.yaml file to disk, May overwrite existing files if overwrite=True
    Idempotent: no
    """
    ...

# ── REQUIRED EXPORTS ──────────────────────────────────
# Your implementation module MUST export ALL of these names
# with EXACTLY these spellings. Tests import them by name.
# __all__ = ['parse_synthesis_output', 'write_artifacts', 'FileExistsError']
