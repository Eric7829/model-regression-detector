"""Tests for the Pydantic interface models.

These are fast, offline tests — no LLM calls. They verify that the data
contracts enforce the constraints we care about.
"""

import pytest
from pydantic import ValidationError

from src.models import (
    ClassifierOutput,
    EmailCategory,
    ExpectedDifficulty,
    FewShotExample,
    GoldenCase,
    GoldenDataset,
    PromptConfig,
    RunMetadata,
)


# ---------------------------------------------------------------------------
# PromptConfig
# ---------------------------------------------------------------------------


class TestPromptConfig:
    def test_minimal_valid(self):
        config = PromptConfig(
            version="v1.0.0",
            system_prompt="You are an email classifier.",
        )
        assert config.version == "v1.0.0"
        assert config.model == "gpt-4o-mini"  # default
        assert config.temperature == 0.0  # default
        assert config.few_shot_examples == []

    def test_with_few_shot_examples(self):
        config = PromptConfig(
            version="v1.1.0",
            system_prompt="Classify emails.",
            few_shot_examples=[
                FewShotExample(
                    email="I was double-charged",
                    category="billing",
                    summary="Customer reports a double charge.",
                )
            ],
        )
        assert len(config.few_shot_examples) == 1
        assert config.few_shot_examples[0].category == "billing"

    def test_temperature_out_of_range(self):
        with pytest.raises(ValidationError):
            PromptConfig(
                version="v1.0.0",
                system_prompt="test",
                temperature=3.0,
            )

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            PromptConfig()  # type: ignore


# ---------------------------------------------------------------------------
# ClassifierOutput
# ---------------------------------------------------------------------------


class TestClassifierOutput:
    def test_valid_output(self):
        output = ClassifierOutput(
            category="billing",
            summary="Customer was charged twice for their subscription.",
        )
        assert output.category == EmailCategory.BILLING
        assert output.confidence == 1.0  # default

    def test_all_categories(self):
        for cat in ["billing", "technical", "account", "general"]:
            output = ClassifierOutput(category=cat, summary="A valid summary for testing purposes.")
            assert output.category.value == cat

    def test_invalid_category(self):
        with pytest.raises(ValidationError):
            ClassifierOutput(
                category="refund",  # not a valid category
                summary="Some summary text here.",
            )

    def test_summary_too_short(self):
        with pytest.raises(ValidationError):
            ClassifierOutput(category="billing", summary="Short")

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            ClassifierOutput(
                category="billing",
                summary="A valid summary for testing.",
                confidence=1.5,
            )


# ---------------------------------------------------------------------------
# GoldenCase / GoldenDataset
# ---------------------------------------------------------------------------


class TestGoldenDataset:
    def _make_case(self, id: str, category: str = "billing", **kwargs) -> GoldenCase:
        return GoldenCase(
            id=id,
            input_email="Test email content for the classifier.",
            expected_category=category,
            expected_summary="Expected summary of the test email.",
            **kwargs,
        )

    def test_dataset_creation(self):
        ds = GoldenDataset(
            version="v1.0.0",
            cases=[self._make_case("TC-001"), self._make_case("TC-002", "technical")],
        )
        assert ds.case_count == 2

    def test_filter_by_category(self):
        ds = GoldenDataset(
            version="v1.0.0",
            cases=[
                self._make_case("TC-001", "billing"),
                self._make_case("TC-002", "technical"),
                self._make_case("TC-003", "billing"),
            ],
        )
        billing = ds.cases_by_category(EmailCategory.BILLING)
        assert len(billing) == 2

    def test_filter_by_difficulty(self):
        ds = GoldenDataset(
            version="v1.0.0",
            cases=[
                self._make_case("TC-001", expected_difficulty="hard"),
                self._make_case("TC-002", expected_difficulty="easy"),
                self._make_case("TC-003", expected_difficulty="hard"),
            ],
        )
        hard = ds.cases_by_difficulty(ExpectedDifficulty.HARD)
        assert len(hard) == 2

    def test_difficulty_values(self):
        for diff in ["easy", "medium", "hard", "adversarial"]:
            case = self._make_case("TC-X", expected_difficulty=diff)
            assert case.expected_difficulty.value == diff


# ---------------------------------------------------------------------------
# RunMetadata
# ---------------------------------------------------------------------------


class TestRunMetadata:
    def test_creation(self):
        meta = RunMetadata(
            run_id="test-run-001",
            prompt_version="v1.0.0",
            model="gpt-4o-mini",
            dataset_version="v1.0.0",
        )
        assert meta.total_cases == 0
        assert meta.finished_at is None
