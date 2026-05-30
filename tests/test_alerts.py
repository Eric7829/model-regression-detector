"""Tests for the Phase 4 Slack Alerting and Formatting module.

Verifies the Block Kit layout structures, dynamic color formatting, delta highlights,
and HTTP webhook delivery handlers.
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.alerts import format_slack_blocks, send_slack_alert
from src.models import (
    CaseFlip,
    CaseResult,
    CategoryAccuracy,
    EmailCategory,
    EvalRunResult,
    RunComparison,
    RunMetadata,
)


@pytest.fixture
def sample_run_result() -> EvalRunResult:
    """Provide a baseline EvalRunResult for formatting tests."""
    metadata = RunMetadata(
        run_id="run-999",
        prompt_version="v1.1.0",
        model="gemini-flash-lite-latest",
        dataset_version="mini_v1.0.0",
        started_at=datetime(2026, 5, 30, 12, 0, 0),
        finished_at=datetime(2026, 5, 30, 12, 0, 10),
        total_cases=2,
        passed_cases=2,
        failed_cases=0,
        error_cases=0,
    )
    
    case_results = [
        CaseResult(
            case_id="TC-001",
            input_email="hello",
            expected_category=EmailCategory.ACCOUNT,
            predicted_category=EmailCategory.ACCOUNT,
            category_match=True,
            expected_summary="Summary 1",
            predicted_summary="Summary 1",
            latency_ms=120.0,
        ),
        CaseResult(
            case_id="TC-002",
            input_email="world",
            expected_category=EmailCategory.BILLING,
            predicted_category=EmailCategory.BILLING,
            category_match=True,
            expected_summary="Summary 2",
            predicted_summary="Summary 2",
            latency_ms=150.0,
        ),
    ]
    
    per_cat = [
        CategoryAccuracy(category=EmailCategory.ACCOUNT, total=1, correct=1, accuracy=1.0),
        CategoryAccuracy(category=EmailCategory.BILLING, total=1, correct=1, accuracy=1.0),
    ]
    
    return EvalRunResult(
        metadata=metadata,
        case_results=case_results,
        overall_accuracy=1.0,
        per_category_accuracy=per_cat,
        avg_summary_score=5.0,
        avg_latency_ms=135.0,
        total_tokens_used=1200,
        total_cost_estimate_usd=0.00015,
    )


class TestSlackAlerts:
    """Validate Block Kit structures and delivery mocks."""

    def test_format_slack_blocks_pass(self, sample_run_result):
        # 1. No baseline comparison (first run)
        blocks = format_slack_blocks(sample_run_result, comparison=None)
        
        # Verify basic structure
        assert len(blocks) >= 5
        assert blocks[0]["type"] == "header"
        assert "🟢" in blocks[0]["text"]["text"]
        
        meta_section = blocks[1]
        assert meta_section["type"] == "section"
        assert "v1.1.0" in meta_section["text"]["text"]
        assert "mini_v1.0.0" in meta_section["text"]["text"]
        assert "VERDICT: PASS" in meta_section["text"]["text"]
        
        metrics_section = blocks[3]
        assert "100.0%" in metrics_section["text"]["text"]
        assert "135.0 ms" in metrics_section["text"]["text"]
        
        category_section = blocks[5]
        assert "account" in category_section["text"]["text"]
        assert "billing" in category_section["text"]["text"]

    def test_format_slack_blocks_critical_regression(self, sample_run_result):
        # 2. Comparison with regressions (critical failure)
        comp = RunComparison(
            baseline_run_id="run-888",
            current_run_id="run-999",
            baseline_accuracy=1.0,
            current_accuracy=0.5,
            accuracy_delta=-50.0,
            per_category_deltas={"account": -100.0},
            regressions=[
                CaseFlip(
                    case_id="TC-001",
                    input_email_preview="hello email",
                    expected_category=EmailCategory.ACCOUNT,
                    old_predicted=EmailCategory.ACCOUNT,
                    new_predicted=EmailCategory.TECHNICAL,
                    flip_type="regression",
                )
            ],
            improvements=[],
            status="critical",
            summary_score_delta=-1.0,
            latency_delta_ms=50.0,
        )
        
        blocks = format_slack_blocks(sample_run_result, comp)
        
        # Verify header is Red fail
        assert "🔴" in blocks[0]["text"]["text"]
        
        # Verify metadata indicates critical regression verdict
        assert "VERDICT: CRITICAL FAILURE" in blocks[1]["text"]["text"]
        
        # Verify regression section exists and lists details
        flips_section = [b for b in blocks if b["type"] == "section" and "Regressions" in b["text"]["text"]]
        assert len(flips_section) == 1
        assert "TC-001" in flips_section[0]["text"]["text"]
        assert "hello email" in flips_section[0]["text"]["text"]
        assert "technical" in flips_section[0]["text"]["text"]

    @patch("urllib.request.urlopen")
    def test_send_slack_alert_success(self, mock_urlopen, sample_run_result):
        # Mock connection success (returns 200)
        mock_response = MagicMock()
        mock_response.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        webhook_url = "https://hooks.slack.com/services/T00/B00/X00"
        success = send_slack_alert(sample_run_result, comparison=None, webhook_url=webhook_url)
        
        assert success is True
        mock_urlopen.assert_called_once()
        
        # Check that the request payload is correct JSON
        args, kwargs = mock_urlopen.call_args
        req = args[0]
        assert req.full_url == webhook_url
        assert req.get_header("Content-type") == "application/json"
        
        # Decode body
        payload = json.loads(req.data.decode("utf-8"))
        assert "blocks" in payload
        assert len(payload["blocks"]) > 0

    @patch("urllib.request.urlopen")
    def test_send_slack_alert_http_error(self, mock_urlopen, sample_run_result):
        # Mock connection failure (raises Exception)
        mock_urlopen.side_effect = Exception("HTTP Timeout/Error")
        
        webhook_url = "https://hooks.slack.com/services/T00/B00/X00"
        success = send_slack_alert(sample_run_result, comparison=None, webhook_url=webhook_url)
        
        # Should catch gracefully and return False
        assert success is False


def test_build_pr_comment_markdown(sample_run_result):
    from scripts.post_pr_comment import build_pr_comment_markdown
    
    # Test pass comment without comparison
    comment = build_pr_comment_markdown(sample_run_result, comparison=None)
    assert "PROMPT EVALUATION: PASSED" in comment
    assert "v1.1.0" in comment
    assert "gemini-flash-lite-latest" in comment
    assert "Accuracy by customer Intent Category" in comment

    # Test failure comment with regressions comparison
    comp = RunComparison(
        baseline_run_id="run-888",
        current_run_id="run-999",
        baseline_accuracy=1.0,
        current_accuracy=0.5,
        accuracy_delta=-50.0,
        per_category_deltas={"account": -100.0},
        regressions=[
            CaseFlip(
                case_id="TC-001",
                input_email_preview="hello email",
                expected_category=EmailCategory.ACCOUNT,
                old_predicted=EmailCategory.ACCOUNT,
                new_predicted=EmailCategory.TECHNICAL,
                flip_type="regression",
            )
        ],
        improvements=[],
        status="critical",
        summary_score_delta=-1.0,
        latency_delta_ms=50.0,
    )
    
    comment_fail = build_pr_comment_markdown(sample_run_result, comp)
    assert "PROMPT EVALUATION: CRITICAL QUALITY FAILURE" in comment_fail
    assert "TC-001" in comment_fail
    assert "Flipped From" in comment_fail
    assert "hello email" in comment_fail

