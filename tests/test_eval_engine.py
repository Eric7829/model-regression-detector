"""Tests for the Phase 3 Evaluation Engine.

Covers the SQLite/JSON storage, LLM-as-a-judge parsing, async concurrency runner, 
retry logic, cost calculation, and regression diff/comparison engine.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.eval_diff import compare_runs
from src.eval_runner import EvalRunner, estimate_token_cost
from src.models import (
    CaseResult,
    CategoryAccuracy,
    EmailCategory,
    EvalConfig,
    EvalRunResult,
    ExpectedDifficulty,
    GoldenCase,
    GoldenDataset,
    PromptConfig,
    RunMetadata,
)
from src.storage import EvaluationStorage


import gc

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test storage."""
    d = tempfile.mkdtemp()
    yield Path(d)
    # Force close any open sqlite connections by running garbage collection
    gc.collect()
    shutil.rmtree(d)


@pytest.fixture
def mock_run_result() -> EvalRunResult:
    """Provide a mock EvalRunResult for testing storage and comparisons."""
    run_id = "test-run-123"
    metadata = RunMetadata(
        run_id=run_id,
        prompt_version="v1.0.0",
        model="gpt-4o-mini",
        dataset_version="v1.0.0",
        started_at=datetime(2026, 5, 30, 10, 0, 0),
        finished_at=datetime(2026, 5, 30, 10, 1, 0),
        total_cases=3,
        passed_cases=2,
        failed_cases=1,
        error_cases=0,
    )
    
    case_results = [
        CaseResult(
            case_id="TC-001",
            input_email="Hello, I need help with resetting my password.",
            expected_category=EmailCategory.ACCOUNT,
            predicted_category=EmailCategory.ACCOUNT,
            category_match=True,
            expected_summary="Customer requested a password reset.",
            predicted_summary="Customer requested a password reset.",
            summary_relevance_score=5.0,
            latency_ms=450.0,
            prompt_tokens=150,
            completion_tokens=25,
            total_tokens=175,
            raw_response=json.dumps({"category": "account", "summary": "Customer requested a password reset."}),
            confidence=0.95,
        ),
        CaseResult(
            case_id="TC-002",
            input_email="I was double billed for my monthly subscription.",
            expected_category=EmailCategory.BILLING,
            predicted_category=EmailCategory.BILLING,
            category_match=True,
            expected_summary="Customer disputing double billing.",
            predicted_summary="Customer disputing double billing.",
            summary_relevance_score=4.8,
            latency_ms=520.0,
            prompt_tokens=180,
            completion_tokens=30,
            total_tokens=210,
            raw_response=json.dumps({"category": "billing", "summary": "Customer disputing double billing."}),
            confidence=0.90,
        ),
        CaseResult(
            case_id="TC-003",
            input_email="The API endpoint `/v1/deploy` is returning 500 error codes.",
            expected_category=EmailCategory.TECHNICAL,
            predicted_category=EmailCategory.GENERAL,  # Failure!
            category_match=False,
            expected_summary="API returning 500 internal server error.",
            predicted_summary="API returned something.",
            summary_relevance_score=2.0,
            latency_ms=610.0,
            prompt_tokens=200,
            completion_tokens=20,
            total_tokens=220,
            raw_response=json.dumps({"category": "general", "summary": "API returned something."}),
            confidence=0.60,
        ),
    ]
    
    per_cat = [
        CategoryAccuracy(category=EmailCategory.ACCOUNT, total=1, correct=1, accuracy=1.0),
        CategoryAccuracy(category=EmailCategory.BILLING, total=1, correct=1, accuracy=1.0),
        CategoryAccuracy(category=EmailCategory.TECHNICAL, total=1, correct=0, accuracy=0.0),
    ]
    
    return EvalRunResult(
        metadata=metadata,
        case_results=case_results,
        overall_accuracy=2/3,
        per_category_accuracy=per_cat,
        avg_summary_score=3.933,
        avg_latency_ms=526.67,
        total_tokens_used=605,
        total_cost_estimate_usd=0.0001,
    )


# ===========================================================================
# 1. Storage Layer Tests
# ===========================================================================

class TestStorageLayer:
    """Verify robust SQLite and JSON file persistence."""

    def test_init_and_save_load(self, temp_dir, mock_run_result):
        db_path = temp_dir / "test.db"
        results_dir = temp_dir / "results"
        
        storage = EvaluationStorage(db_path=db_path, results_dir=results_dir)
        
        # Save run
        storage.save_run(mock_run_result)
        
        # Verify JSON file exists
        json_file = results_dir / f"run_{mock_run_result.metadata.run_id}.json"
        assert json_file.exists()
        
        # Load run and verify equivalence
        loaded = storage.load_run(mock_run_result.metadata.run_id)
        assert loaded is not None
        assert loaded.metadata.run_id == mock_run_result.metadata.run_id
        assert loaded.overall_accuracy == pytest.approx(2/3)
        assert len(loaded.case_results) == 3
        assert loaded.case_results[0].case_id == "TC-001"
        assert loaded.case_results[0].expected_category == EmailCategory.ACCOUNT
        assert loaded.case_results[2].category_match is False
        assert loaded.case_results[2].predicted_category == EmailCategory.GENERAL

    def test_list_runs_and_delete(self, temp_dir, mock_run_result):
        db_path = temp_dir / "test.db"
        storage = EvaluationStorage(db_path=db_path, results_dir=temp_dir / "results")
        
        storage.save_run(mock_run_result)
        
        runs = storage.list_runs()
        assert len(runs) == 1
        assert runs[0]["run_id"] == mock_run_result.metadata.run_id
        assert runs[0]["prompt_version"] == "v1.0.0"
        
        # Test deletion
        storage.delete_run(mock_run_result.metadata.run_id)
        assert len(storage.list_runs()) == 0
        assert storage.load_run(mock_run_result.metadata.run_id) is None

    def test_get_latest_run(self, temp_dir, mock_run_result):
        db_path = temp_dir / "test.db"
        storage = EvaluationStorage(db_path=db_path, results_dir=temp_dir / "results")
        
        storage.save_run(mock_run_result)
        
        # Match latest
        latest = storage.get_latest_run(dataset_version="v1.0.0")
        assert latest is not None
        assert latest.metadata.run_id == mock_run_result.metadata.run_id
        
        # Mismatch version returns None
        assert storage.get_latest_run(dataset_version="v2.0.0") is None


# ===========================================================================
# 2. Token Cost Estimation Tests
# ===========================================================================

class TestCostEstimation:
    """Verify accurate pricing computations."""

    def test_token_costs(self):
        # GPT-4o-mini: input = 0.15/1M, output = 0.60/1M
        assert estimate_token_cost("gpt-4o-mini", 1_000_000, 2_000_000) == pytest.approx(0.15 + 1.20)
        
        # GPT-4o: input = 5.00/1M, output = 15.00/1M
        assert estimate_token_cost("gpt-4o", 100_000, 50_000) == pytest.approx(0.50 + 0.75)
        
        # Gemini-2.0-flash-lite: input = 0.075/1M, output = 0.30/1M
        assert estimate_token_cost("gemini-2.0-flash-lite", 1_000_000, 1_000_000) == pytest.approx(0.075 + 0.30)
        
        # Default fallback
        assert estimate_token_cost("unknown-model", 1_000_000, 1_000_000) == pytest.approx(0.15 + 0.60)


# ===========================================================================
# 3. Run Comparison Engine Tests
# ===========================================================================

class TestComparisonEngine:
    """Verify correct categorization of regressions, improvements, and status."""

    def test_run_comparisons(self, mock_run_result):
        # We will create a mock "current" run to compare against the baseline mock_run_result
        # Baseline mock_run_result accuracy: 2/3 (66.67%)
        # Let's construct a current run that has:
        # - TC-001: still passed
        # - TC-002: flipped to FAIL (regression!)
        # - TC-003: flipped to PASS (improvement!)
        # Overall accuracy remains 2/3, but we have exactly 1 regression and 1 improvement.
        
        current_metadata = RunMetadata(
            run_id="test-run-456",
            prompt_version="v1.1.0",
            model="gpt-4o-mini",
            dataset_version="v1.0.0",
            started_at=datetime.utcnow(),
            finished_at=datetime.utcnow(),
            total_cases=3,
            passed_cases=2,
            failed_cases=1,
            error_cases=0,
        )
        
        current_case_results = [
            # TC-001 stays passed
            CaseResult(
                case_id="TC-001",
                input_email="Hello, I need help with resetting my password.",
                expected_category=EmailCategory.ACCOUNT,
                predicted_category=EmailCategory.ACCOUNT,
                category_match=True,
                expected_summary="Customer requested a password reset.",
                predicted_summary="Customer requested a password reset.",
                summary_relevance_score=5.0,
                latency_ms=410.0,
            ),
            # TC-002 flips to fail (predicted general instead of billing)
            CaseResult(
                case_id="TC-002",
                input_email="I was double billed for my monthly subscription.",
                expected_category=EmailCategory.BILLING,
                predicted_category=EmailCategory.GENERAL,
                category_match=False,
                expected_summary="Customer disputing double billing.",
                predicted_summary="Billing complaint.",
                summary_relevance_score=4.0,
                latency_ms=480.0,
            ),
            # TC-003 flips to pass (predicted technical correctly!)
            CaseResult(
                case_id="TC-003",
                input_email="The API endpoint `/v1/deploy` is returning 500 error codes.",
                expected_category=EmailCategory.TECHNICAL,
                predicted_category=EmailCategory.TECHNICAL,
                category_match=True,
                expected_summary="API returning 500 internal server error.",
                predicted_summary="API returning 500 error.",
                summary_relevance_score=4.8,
                latency_ms=500.0,
            ),
        ]
        
        current_run = EvalRunResult(
            metadata=current_metadata,
            case_results=current_case_results,
            overall_accuracy=2/3,
            per_category_accuracy=[
                CategoryAccuracy(category=EmailCategory.ACCOUNT, total=1, correct=1, accuracy=1.0),
                CategoryAccuracy(category=EmailCategory.BILLING, total=1, correct=0, accuracy=0.0),
                CategoryAccuracy(category=EmailCategory.TECHNICAL, total=1, correct=1, accuracy=1.0),
            ],
            avg_summary_score=4.6,
            avg_latency_ms=463.33,
        )
        
        comp = compare_runs(baseline=mock_run_result, current=current_run)
        
        assert comp.baseline_run_id == mock_run_result.metadata.run_id
        assert comp.current_run_id == current_run.metadata.run_id
        assert comp.accuracy_delta == pytest.approx(0.0)
        assert comp.status == "pass"  # No accuracy drop overall
        
        # Verify flips
        assert len(comp.regressions) == 1
        assert comp.regressions[0].case_id == "TC-002"
        assert comp.regressions[0].old_predicted == EmailCategory.BILLING
        assert comp.regressions[0].new_predicted == EmailCategory.GENERAL
        
        assert len(comp.improvements) == 1
        assert comp.improvements[0].case_id == "TC-003"
        assert comp.improvements[0].old_predicted == EmailCategory.GENERAL
        assert comp.improvements[0].new_predicted == EmailCategory.TECHNICAL

    def test_run_comparison_regression_thresholds(self, mock_run_result):
        # If accuracy drops significantly, it should trigger warning/critical
        # Baseline accuracy: 2/3 (66.67%)
        # Let's make current run accuracy 0.0% (all failed) -> Drop of 66.67%
        
        current_metadata = RunMetadata(
            run_id="test-run-789",
            prompt_version="v1.1.0",
            model="gpt-4o-mini",
            dataset_version="v1.0.0",
            total_cases=3,
            passed_cases=0,
            failed_cases=3,
        )
        current_run = EvalRunResult(
            metadata=current_metadata,
            case_results=[
                CaseResult(case_id="TC-001", input_email="a", expected_category=EmailCategory.ACCOUNT, expected_summary="a", category_match=False),
                CaseResult(case_id="TC-002", input_email="b", expected_category=EmailCategory.BILLING, expected_summary="b", category_match=False),
                CaseResult(case_id="TC-003", input_email="c", expected_category=EmailCategory.TECHNICAL, expected_summary="c", category_match=False),
            ],
            overall_accuracy=0.0,
        )
        
        config = EvalConfig(warning_threshold_pct=5.0, critical_threshold_pct=15.0)
        comp = compare_runs(baseline=mock_run_result, current=current_run, config=config)
        
        assert comp.accuracy_delta == pytest.approx(-66.6666, abs=1e-2)
        assert comp.status == "critical"  # Drop of 66% exceeds critical threshold of 15%


# ===========================================================================
# 4. Evaluation Runner Mock Integration Tests
# ===========================================================================

class TestEvalRunnerMock:
    """Verify parallel processing semaphore and retry behavior using mocks."""

    @pytest.mark.asyncio
    @patch("src.eval_runner.classify_email", new_callable=AsyncMock)
    @patch("src.eval_runner.score_summary_relevance", new_callable=AsyncMock)
    async def test_runner_concurrency_and_scoring(self, mock_judge, mock_classify):
        # Setup mocks
        from src.classifier import ClassificationResult
        from src.models import ClassifierOutput
        
        # Classifier returns successful classification
        mock_classify.return_value = ClassificationResult(
            output=ClassifierOutput(category=EmailCategory.ACCOUNT, summary="Test summary reset password", confidence=0.9),
            raw_response="{\"category\": \"account\", \"summary\": \"Test summary reset password\"}",
            latency_ms=150.0,
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
        )
        
        # Judge returns score of 5.0, reasoning, and mock token usage
        mock_judge.return_value = (5.0, "Great summary!", 800, 80)
        
        dataset = GoldenDataset(
            version="v1.0.0",
            cases=[
                GoldenCase(
                    id="TC-001",
                    input_email="Please reset my password.",
                    expected_category=EmailCategory.ACCOUNT,
                    expected_summary="Password reset request",
                    expected_difficulty=ExpectedDifficulty.EASY,
                    notes="Test easy case",
                )
            ]
        )
        
        prompt_config = PromptConfig(
            version="v1.0.0",
            system_prompt="Classify this email.",
            model="gemini-2.0-flash-lite",
            temperature=0.0,
        )
        
        config = EvalConfig(max_concurrency=2, retry_count=1)
        runner = EvalRunner(config)
        
        # Mock API clients
        runner.classifier_client = MagicMock()
        runner.judge_client = MagicMock()
        
        run_res = await runner.run_evaluation(
            dataset=dataset,
            prompt_config=prompt_config,
            skip_judge=False,
        )
        
        assert run_res.metadata.total_cases == 1
        assert run_res.metadata.passed_cases == 1
        assert run_res.overall_accuracy == 1.0
        assert run_res.avg_summary_score == 5.0
        assert run_res.case_results[0].category_match is True
        assert run_res.case_results[0].summary_relevance_score == 5.0
        
        # Verify classification and judge were called
        mock_classify.assert_called_once()
        mock_judge.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.eval_runner.classify_email", new_callable=AsyncMock)
    async def test_runner_retries_on_failure(self, mock_classify):
        from src.classifier import ClassificationResult
        
        # Fail on first call, succeed on second call
        mock_classify.side_effect = [
            Exception("API Rate Limit exceeded!"),
            ClassificationResult(
                output=None,  # Or successfully parsed
                raw_response="",
                latency_ms=0,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                error="Some transient classification failure"
            )
        ]
        
        # Let's mock a second scenario where it eventually fails
        # Actually let's test retry success:
        mock_classify.side_effect = [
            Exception("API Rate Limit exceeded!"),
            ClassificationResult(
                output=None,
                raw_response="",
                latency_ms=100.0,
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
                error="Transient parse error"
            )
        ]
        
        dataset = GoldenDataset(
            version="v1.0.0",
            cases=[
                GoldenCase(
                    id="TC-001",
                    input_email="Please reset my password.",
                    expected_category=EmailCategory.ACCOUNT,
                    expected_summary="Password reset request",
                    expected_difficulty=ExpectedDifficulty.EASY,
                    notes="Test easy case",
                )
            ]
        )
        
        prompt_config = PromptConfig(
            version="v1.0.0",
            system_prompt="Classify this email.",
            model="gemini-2.0-flash-lite",
        )
        
        config = EvalConfig(max_concurrency=2, retry_count=1, retry_delay_s=0.01)
        runner = EvalRunner(config)
        runner.classifier_client = MagicMock()
        
        run_res = await runner.run_evaluation(
            dataset=dataset,
            prompt_config=prompt_config,
            skip_judge=True,
        )
        
        # Should complete and call classify twice due to retry count = 1 (1 try + 1 retry)
        assert mock_classify.call_count == 2
        assert run_res.metadata.error_cases == 1 or run_res.metadata.failed_cases == 1
