"""Pydantic models defining the interface contract for the eval pipeline.

The key design choice here is strict typing everywhere. The eval engine never
deals with raw dicts — everything flows through these models so a schema
mismatch surfaces as a validation error, not a silent wrong answer.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Prompt configuration — the "code" that CI runs against
# ---------------------------------------------------------------------------


class FewShotExample(BaseModel):
    """A single few-shot demonstration embedded in a prompt version."""

    email: str = Field(..., description="Example customer email text")
    category: str = Field(..., description="Correct category label for this email")
    summary: str = Field(..., description="Ideal one-sentence summary")


class PromptConfig(BaseModel):
    """Fully describes one version of the classification prompt.

    Loaded from a YAML file in ``/prompts``. The eval pipeline treats this as
    the immutable input for a single run — swap the file, get a new run.
    """

    version: str = Field(..., description="Semantic version string, e.g. 'v1.0.0'")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this prompt version was authored",
    )
    model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model identifier to use with this prompt",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature — 0 for deterministic evals",
    )
    system_prompt: str = Field(..., description="The full system prompt sent to the LLM")
    few_shot_examples: list[FewShotExample] = Field(
        default_factory=list,
        description="Optional few-shot examples appended after the system prompt",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form metadata (author, change rationale, ticket link, etc.)",
    )


# ---------------------------------------------------------------------------
# Classifier output — the structured JSON the LLM must return
# ---------------------------------------------------------------------------


class EmailCategory(str, Enum):
    """The fixed set of categories the classifier can assign."""

    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"


class ClassifierOutput(BaseModel):
    """Structured output expected from the email classifier.

    The LLM is instructed to respond with JSON that validates against this
    schema. If it doesn't, the test case is marked as a parse failure.
    """

    category: EmailCategory = Field(..., description="Predicted email category")
    summary: str = Field(
        ...,
        min_length=10,
        max_length=300,
        description="One-sentence summary of the customer's issue",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Model's self-reported confidence (0-1)",
    )


# ---------------------------------------------------------------------------
# Eval run metadata — ties a run to its inputs
# ---------------------------------------------------------------------------


class RunMetadata(BaseModel):
    """Captures everything needed to reproduce or compare an eval run."""

    run_id: str = Field(..., description="Unique run identifier (UUID)")
    prompt_version: str = Field(..., description="Version string from the PromptConfig used")
    model: str = Field(..., description="Model identifier used for this run")
    dataset_version: str = Field(..., description="Version of the golden dataset")
    started_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: datetime | None = Field(default=None)
    total_cases: int = Field(default=0)
    passed_cases: int = Field(default=0)
    failed_cases: int = Field(default=0)
    error_cases: int = Field(default=0)


# ---------------------------------------------------------------------------
# Golden dataset case — one row of the test suite
# ---------------------------------------------------------------------------


class ExpectedDifficulty(str, Enum):
    """How hard we expect this test case to be for the classifier."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    ADVERSARIAL = "adversarial"


class GoldenCase(BaseModel):
    """A single test case in the golden dataset.

    The ``notes`` field is critical — it captures *why* this case exists,
    which matters when someone debates removing a flaky test.
    """

    id: str = Field(..., description="Stable unique identifier, e.g. 'TC-001'")
    input_email: str = Field(..., description="The raw customer email text")
    expected_category: EmailCategory = Field(..., description="Ground-truth category")
    expected_summary: str = Field(
        ..., description="Reference summary (used for LLM-as-judge scoring)"
    )
    expected_difficulty: ExpectedDifficulty = Field(
        default=ExpectedDifficulty.MEDIUM,
        description="How hard this case should be",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Freeform tags for filtering (e.g. 'edge-case', 'multilingual')",
    )
    notes: str = Field(
        default="",
        description="Why this test case exists and what it's designed to catch",
    )


class GoldenDataset(BaseModel):
    """The full versioned golden dataset consumed by the eval runner."""

    version: str = Field(..., description="Dataset version, e.g. 'v1.0.0'")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    description: str = Field(default="", description="What changed in this version")
    cases: list[GoldenCase] = Field(..., description="All test cases")

    @property
    def case_count(self) -> int:
        return len(self.cases)

    def cases_by_category(self, category: EmailCategory) -> list[GoldenCase]:
        return [c for c in self.cases if c.expected_category == category]

    def cases_by_difficulty(self, difficulty: ExpectedDifficulty) -> list[GoldenCase]:
        return [c for c in self.cases if c.expected_difficulty == difficulty]


# ---------------------------------------------------------------------------
# Eval results — output of the evaluation engine
# ---------------------------------------------------------------------------


class CaseResult(BaseModel):
    """Result of evaluating a single test case."""

    case_id: str = Field(..., description="The test case ID from the golden dataset")
    input_email: str = Field(..., description="The email text that was classified")
    expected_category: EmailCategory = Field(..., description="Ground truth category")
    predicted_category: EmailCategory | None = Field(
        default=None, description="Category predicted by the LLM"
    )
    category_match: bool = Field(default=False, description="Whether predicted matches expected")
    expected_summary: str = Field(..., description="Reference summary")
    predicted_summary: str = Field(default="", description="Summary generated by the LLM")
    summary_relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description="LLM-as-judge relevance score (1-5), 0 if not scored",
    )
    latency_ms: float = Field(default=0.0, description="Time to get LLM response in ms")
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)
    raw_response: str = Field(default="", description="Raw LLM response text")
    error: str | None = Field(default=None, description="Error message if the case failed")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def passed(self) -> bool:
        """A case passes if category matches and no error occurred."""
        return self.category_match and self.error is None


class CategoryAccuracy(BaseModel):
    """Accuracy stats for a single category."""

    category: EmailCategory
    total: int = 0
    correct: int = 0
    accuracy: float = 0.0


class EvalRunResult(BaseModel):
    """Complete result of an evaluation run."""

    metadata: RunMetadata
    case_results: list[CaseResult] = Field(default_factory=list)
    overall_accuracy: float = Field(default=0.0, description="Fraction of cases with correct category")
    per_category_accuracy: list[CategoryAccuracy] = Field(default_factory=list)
    avg_summary_score: float = Field(default=0.0, description="Mean LLM-as-judge score across cases")
    avg_latency_ms: float = Field(default=0.0)
    total_tokens_used: int = Field(default=0)
    total_cost_estimate_usd: float = Field(default=0.0)


class CaseFlip(BaseModel):
    """A test case that changed status between two runs."""

    case_id: str
    input_email_preview: str = Field(description="First 100 chars of the email")
    expected_category: EmailCategory
    old_predicted: EmailCategory | None = None
    new_predicted: EmailCategory | None = None
    old_summary: str = ""
    new_summary: str = ""
    flip_type: str = Field(description="'regression' or 'improvement'")


class RunComparison(BaseModel):
    """Diff between two evaluation runs."""

    baseline_run_id: str
    current_run_id: str
    baseline_accuracy: float
    current_accuracy: float
    accuracy_delta: float = Field(description="Positive = improvement, negative = regression")
    per_category_deltas: dict[str, float] = Field(default_factory=dict)
    regressions: list[CaseFlip] = Field(default_factory=list)
    improvements: list[CaseFlip] = Field(default_factory=list)
    status: str = Field(
        default="pass",
        description="'pass', 'warning', or 'critical' based on thresholds",
    )
    summary_score_delta: float = 0.0
    latency_delta_ms: float = 0.0


class EvalConfig(BaseModel):
    """Configuration for the evaluation engine."""

    warning_threshold_pct: float = Field(default=3.0, description="Flag warning if accuracy drops by this %")
    critical_threshold_pct: float = Field(default=8.0, description="Flag critical if accuracy drops by this %")
    max_concurrency: int = Field(default=5, description="Max concurrent LLM calls")
    judge_model: str = Field(default="gemini-2.0-flash-lite", description="Model for LLM-as-judge scoring")
    retry_count: int = Field(default=2, description="Retries per failed LLM call")
    retry_delay_s: float = Field(default=1.0, description="Delay between retries")
