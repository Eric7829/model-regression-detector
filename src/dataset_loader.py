"""Loads and validates the versioned golden dataset JSON files.

Works with the GoldenDataset Pydantic model to ensure every test case
conforms to the schema at load time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.models import GoldenDataset


# Default location — relative to the project root
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def get_data_dir() -> Path:
    """Return the data directory, allowing override via env var."""
    override = os.environ.get("MRD_DATA_DIR")
    if override:
        return Path(override)
    return _DATA_DIR


def load_golden_dataset(filename: str, data_dir: Path | None = None) -> GoldenDataset:
    """Load a golden dataset JSON file and validate against the schema.

    Parameters
    ----------
    filename:
        Name of the JSON file (e.g. ``"golden_dataset_v1.0.0.json"``).
    data_dir:
        Optional override for the data directory.

    Returns
    -------
    GoldenDataset
        A fully validated golden dataset.

    Raises
    ------
    FileNotFoundError
        If the JSON file doesn't exist.
    pydantic.ValidationError
        If the JSON content doesn't match the GoldenDataset schema.
    """
    directory = data_dir or get_data_dir()
    filepath = directory / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Golden dataset file not found: {filepath}\n"
            f"Available files: {list(directory.glob('*.json'))}"
        )

    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return GoldenDataset(**raw)


def load_latest_golden_dataset(data_dir: Path | None = None) -> GoldenDataset:
    """Load the most recent golden dataset (by filename sort)."""
    directory = data_dir or get_data_dir()
    json_files = sorted(directory.glob("golden_dataset_*.json"))

    if not json_files:
        raise FileNotFoundError(f"No golden dataset files found in {directory}")

    return load_golden_dataset(json_files[-1].name, directory)


def list_dataset_versions(data_dir: Path | None = None) -> list[str]:
    """Return all available dataset version filenames, sorted."""
    directory = data_dir or get_data_dir()
    return sorted([f.name for f in directory.glob("golden_dataset_*.json")])
