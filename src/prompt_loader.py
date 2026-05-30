"""Loads and validates versioned prompt YAML files from the /prompts directory.

Design rationale: Prompts are treated as first-class versioned artifacts, not
buried in code. The loader enforces the PromptConfig schema at read time so
a malformed prompt file fails fast with a clear error instead of producing
garbage LLM outputs 20 minutes into an eval run.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from src.models import PromptConfig


# Default location — relative to the project root
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def get_prompts_dir() -> Path:
    """Return the prompts directory, allowing override via env var."""
    override = os.environ.get("MRD_PROMPTS_DIR")
    if override:
        return Path(override)
    return _PROMPTS_DIR


def load_prompt(filename: str, prompts_dir: Path | None = None) -> PromptConfig:
    """Load a single prompt YAML file and validate it against PromptConfig.

    Parameters
    ----------
    filename:
        Name of the YAML file (e.g. ``"v1.0.0.yaml"``).
    prompts_dir:
        Optional override for the prompts directory.

    Returns
    -------
    PromptConfig
        A fully validated prompt configuration.

    Raises
    ------
    FileNotFoundError
        If the YAML file doesn't exist.
    pydantic.ValidationError
        If the YAML content doesn't match the PromptConfig schema.
    """
    directory = prompts_dir or get_prompts_dir()
    filepath = directory / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {filepath}\n"
            f"Available files: {list(directory.glob('*.yaml'))}"
        )

    with open(filepath, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return PromptConfig(**raw)


def load_latest_prompt(prompts_dir: Path | None = None) -> PromptConfig:
    """Load the most recently created prompt file (by filename sort).

    Convention: prompt files are named ``v<semver>.yaml`` so lexicographic
    sort gives us the latest version.
    """
    directory = prompts_dir or get_prompts_dir()
    yaml_files = sorted(directory.glob("*.yaml"))

    if not yaml_files:
        raise FileNotFoundError(f"No prompt YAML files found in {directory}")

    return load_prompt(yaml_files[-1].name, directory)


def list_prompt_versions(prompts_dir: Path | None = None) -> list[str]:
    """Return all available prompt version filenames, sorted."""
    directory = prompts_dir or get_prompts_dir()
    return sorted([f.name for f in directory.glob("*.yaml")])
