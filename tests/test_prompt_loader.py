"""Tests for the prompt loader module."""

import textwrap
from pathlib import Path

import pytest

from src.prompt_loader import list_prompt_versions, load_latest_prompt, load_prompt


# Use the real prompts directory for integration-style tests
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class TestLoadPrompt:
    def test_load_v1_0_0(self):
        config = load_prompt("v1.0.0.yaml", PROMPTS_DIR)
        assert config.version == "v1.0.0"
        assert config.model == "gemini-flash-lite-latest"
        assert config.temperature == 0.0
        assert len(config.few_shot_examples) == 0
        assert "email classifier" in config.system_prompt.lower()

    def test_load_v1_1_0(self):
        config = load_prompt("v1.1.0.yaml", PROMPTS_DIR)
        assert config.version == "v1.1.0"
        assert len(config.few_shot_examples) == 4
        assert config.few_shot_examples[0].category == "billing"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_prompt("nonexistent.yaml", PROMPTS_DIR)


class TestLoadLatestPrompt:
    def test_returns_latest_version(self):
        config = load_latest_prompt(PROMPTS_DIR)
        # v1.1.0 sorts after v1.0.0
        assert config.version == "v1.1.0"


class TestListPromptVersions:
    def test_lists_all_versions(self):
        versions = list_prompt_versions(PROMPTS_DIR)
        assert "v1.0.0.yaml" in versions
        assert "v1.1.0.yaml" in versions
        assert len(versions) >= 2


class TestLoadFromTempDir:
    """Test loading from a custom directory with a minimal YAML file."""

    def test_load_minimal_prompt(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            version: "v0.0.1"
            system_prompt: "Test prompt"
        """)
        prompt_file = tmp_path / "v0.0.1.yaml"
        prompt_file.write_text(yaml_content, encoding="utf-8")

        config = load_prompt("v0.0.1.yaml", tmp_path)
        assert config.version == "v0.0.1"
        assert config.system_prompt == "Test prompt"

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_latest_prompt(tmp_path)
