"""Tests for the golden dataset and dataset loader."""

from pathlib import Path

import pytest

from src.dataset_loader import list_dataset_versions, load_golden_dataset, load_latest_golden_dataset
from src.models import EmailCategory, ExpectedDifficulty, GoldenDataset


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class TestLoadGoldenDataset:
    def test_load_v1(self):
        ds = load_golden_dataset("golden_dataset_v1.0.0.json", DATA_DIR)
        assert ds.version == "v1.0.0"
        assert ds.case_count >= 50  # Spec says 50-100

    def test_all_cases_have_required_fields(self):
        ds = load_golden_dataset("golden_dataset_v1.0.0.json", DATA_DIR)
        for case in ds.cases:
            assert case.id.startswith("TC-")
            assert len(case.input_email) > 0
            assert case.expected_category in EmailCategory
            assert len(case.expected_summary) > 0
            assert case.expected_difficulty in ExpectedDifficulty

    def test_all_categories_represented(self):
        ds = load_golden_dataset("golden_dataset_v1.0.0.json", DATA_DIR)
        categories = {c.expected_category for c in ds.cases}
        assert categories == {
            EmailCategory.BILLING,
            EmailCategory.TECHNICAL,
            EmailCategory.ACCOUNT,
            EmailCategory.GENERAL,
        }

    def test_all_difficulties_represented(self):
        ds = load_golden_dataset("golden_dataset_v1.0.0.json", DATA_DIR)
        difficulties = {c.expected_difficulty for c in ds.cases}
        assert ExpectedDifficulty.EASY in difficulties
        assert ExpectedDifficulty.MEDIUM in difficulties
        assert ExpectedDifficulty.HARD in difficulties
        assert ExpectedDifficulty.ADVERSARIAL in difficulties

    def test_unique_ids(self):
        ds = load_golden_dataset("golden_dataset_v1.0.0.json", DATA_DIR)
        ids = [c.id for c in ds.cases]
        assert len(ids) == len(set(ids)), "Duplicate test case IDs found"

    def test_category_balance(self):
        """No category should have fewer than 10 cases."""
        ds = load_golden_dataset("golden_dataset_v1.0.0.json", DATA_DIR)
        for cat in EmailCategory:
            count = len(ds.cases_by_category(cat))
            assert count >= 10, f"Category {cat.value} has only {count} cases (need >= 10)"

    def test_edge_cases_present(self):
        """At least some cases should be tagged as edge cases."""
        ds = load_golden_dataset("golden_dataset_v1.0.0.json", DATA_DIR)
        edge_cases = [c for c in ds.cases if "edge-case" in c.tags]
        assert len(edge_cases) >= 5, "Need at least 5 edge cases"

    def test_has_notes(self):
        """Every case should have a notes field explaining why it exists."""
        ds = load_golden_dataset("golden_dataset_v1.0.0.json", DATA_DIR)
        cases_with_notes = [c for c in ds.cases if c.notes.strip()]
        assert len(cases_with_notes) == ds.case_count, "All cases should have notes"


class TestLoadLatest:
    def test_loads_latest(self):
        ds = load_latest_golden_dataset(DATA_DIR)
        assert ds.version == "v1.0.0"

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_latest_golden_dataset(tmp_path)


class TestListVersions:
    def test_lists_versions(self):
        versions = list_dataset_versions(DATA_DIR)
        assert "golden_dataset_v1.0.0.json" in versions


class TestFileNotFound:
    def test_missing_file(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_golden_dataset("nonexistent.json", DATA_DIR)
