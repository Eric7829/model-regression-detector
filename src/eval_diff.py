"""Diff and regression comparison engine.

Compares two evaluation runs (baseline and current) to compute accuracy deltas,
category shifts, latency/score deltas, and list individual case flips.
"""

from __future__ import annotations

import logging
from src.models import CaseFlip, EvalConfig, EvalRunResult, RunComparison

logger = logging.getLogger(__name__)


def compare_runs(
    baseline: EvalRunResult,
    current: EvalRunResult,
    config: EvalConfig | None = None,
) -> RunComparison:
    """Compare a current evaluation run against a baseline run.

    Computes overall deltas and flags regressions and improvements at the individual
    case level.
    """
    config = config or EvalConfig()
    
    baseline_cases = {c.case_id: c for c in baseline.case_results}
    current_cases = {c.case_id: c for c in current.case_results}
    
    regressions: list[CaseFlip] = []
    improvements: list[CaseFlip] = []
    
    # Identify case status flips
    for case_id, curr_res in current_cases.items():
        base_res = baseline_cases.get(case_id)
        if not base_res:
            # Case is new in this run, cannot compare flip status
            continue
            
        curr_passed = curr_res.passed
        base_passed = base_res.passed
        
        email_preview = curr_res.input_email[:100] + "..." if len(curr_res.input_email) > 100 else curr_res.input_email
        
        if base_passed and not curr_passed:
            # Case flipped from PASS to FAIL/ERROR -> REGRESSION
            regressions.append(
                CaseFlip(
                    case_id=case_id,
                    input_email_preview=email_preview,
                    expected_category=curr_res.expected_category,
                    old_predicted=base_res.predicted_category,
                    new_predicted=curr_res.predicted_category,
                    old_summary=base_res.predicted_summary,
                    new_summary=curr_res.predicted_summary,
                    flip_type="regression",
                )
            )
        elif not base_passed and curr_passed:
            # Case flipped from FAIL/ERROR to PASS -> IMPROVEMENT
            improvements.append(
                CaseFlip(
                    case_id=case_id,
                    input_email_preview=email_preview,
                    expected_category=curr_res.expected_category,
                    old_predicted=base_res.predicted_category,
                    new_predicted=curr_res.predicted_category,
                    old_summary=base_res.predicted_summary,
                    new_summary=curr_res.predicted_summary,
                    flip_type="improvement",
                )
            )

    # Convert accuracy to percentage deltas (e.g. 0.85 -> 85%)
    baseline_acc_pct = baseline.overall_accuracy * 100
    current_acc_pct = current.overall_accuracy * 100
    accuracy_delta = current_acc_pct - baseline_acc_pct
    
    # Per-category deltas
    baseline_cat_acc = {c.category.value: c.accuracy * 100 for c in baseline.per_category_accuracy}
    current_cat_acc = {c.category.value: c.accuracy * 100 for c in current.per_category_accuracy}
    
    per_category_deltas = {}
    all_categories = set(baseline_cat_acc.keys()) | set(current_cat_acc.keys())
    for cat in all_categories:
        base_cat = baseline_cat_acc.get(cat, 0.0)
        curr_cat = current_cat_acc.get(cat, 0.0)
        per_category_deltas[cat] = curr_cat - base_cat

    # Delta for latency and summary scores
    summary_score_delta = current.avg_summary_score - baseline.avg_summary_score
    latency_delta_ms = current.avg_latency_ms - baseline.avg_latency_ms

    # Determine PASS/WARNING/CRITICAL status
    # Standard regression threshold checks
    status = "pass"
    
    # If accuracy has dropped
    if accuracy_delta < 0:
        drop_magnitude = abs(accuracy_delta)
        if drop_magnitude >= config.critical_threshold_pct:
            status = "critical"
        elif drop_magnitude >= config.warning_threshold_pct:
            status = "warning"
            
    # Also evaluate critical regressions on specific edge-cases if designated
    # (e.g. if we have a regression in high-priority tags, but for now we rely on accuracy drop)

    return RunComparison(
        baseline_run_id=baseline.metadata.run_id,
        current_run_id=current.metadata.run_id,
        baseline_accuracy=baseline.overall_accuracy,
        current_accuracy=current.overall_accuracy,
        accuracy_delta=accuracy_delta,
        per_category_deltas=per_category_deltas,
        regressions=regressions,
        improvements=improvements,
        status=status,
        summary_score_delta=summary_score_delta,
        latency_delta_ms=latency_delta_ms,
    )
