"""Artifact generation: parsing LLM synthesis output and writing files."""

from __future__ import annotations

import re
from pathlib import Path


def parse_synthesis_output(raw: str) -> tuple[str, str]:
    """Parse '--- PROMPT ---' and '--- CONSTRAINTS ---' delimiters from LLM output.

    Returns (prompt_md, constraints_yaml).
    """
    prompt_marker = "--- PROMPT ---"
    constraints_marker = "--- CONSTRAINTS ---"

    prompt_idx = raw.find(prompt_marker)
    constraints_idx = raw.find(constraints_marker)

    if prompt_idx == -1 or constraints_idx == -1:
        # Fallback: return raw as prompt, empty constraints
        raise ValueError(
            "Synthesis output missing required delimiters "
            f"('{prompt_marker}' and/or '{constraints_marker}')."
        )

    prompt_start = prompt_idx + len(prompt_marker)
    prompt_md = raw[prompt_start:constraints_idx].strip()

    constraints_start = constraints_idx + len(constraints_marker)
    constraints_yaml = raw[constraints_start:].strip()

    return prompt_md, constraints_yaml


def write_artifacts(
    prompt_md: str,
    constraints_yaml: str,
    output_dir: str | Path,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write prompt.md and constraints.yaml to output_dir.

    Returns (prompt_path, constraints_path).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    prompt_path = out / "prompt.md"
    constraints_path = out / "constraints.yaml"

    existing = []
    if prompt_path.exists():
        existing.append(prompt_path)
    if constraints_path.exists():
        existing.append(constraints_path)

    if existing and not overwrite:
        names = ", ".join(p.name for p in existing)
        raise FileExistsError(
            f"Artifact(s) already exist: {names}. Use overwrite=True to replace."
        )

    prompt_path.write_text(prompt_md, encoding="utf-8")
    constraints_path.write_text(constraints_yaml, encoding="utf-8")

    return prompt_path, constraints_path
