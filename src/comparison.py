"""Comparison and diff logic between evaluation runs.

The core value proposition: for every eval run, compare against the
previous run and surface regressions, improvements, and drift.
"""

from __future__ import annotations

import logging

from src.models import (
    CaseFlip,
    EvalConfig,
    EvalRunResult,
    RunComparison,
)

logger = logging.getLogger(__name__)


def compare_runs(
    baseline: EvalRunResult,
    current: EvalRunResult,
    config: EvalConfig | None = None,
) -> RunComparison:
    """Compare two eval runs and produce a diff.

    Parameters
    ----------
    baseline:
        The previous run to compare against.
    current:
        The new run to evaluate.
    config:
        Threshold configuration for warning/critical status.

    Returns
    -------
    RunComparison
        Contains accuracy deltas, lists of regressions and improvements,
        and an overall pass/warning/critical status.
    """
    cfg = config or EvalConfig()

    # Build lookup maps by case_id
    baseline_map = {cr.case_id: cr for cr in baseline.case_results}
    current_map = {cr.case_id: cr for cr in current.case_results}

    # Find cases present in both runs
    common_ids = set(baseline_map.keys()) & set(current_map.keys())

    regressions = []
    improvements = []

    for case_id in sorted(common_ids):
        old = baseline_map[case_id]
        new = current_map[case_id]

        old_passed = old.passed
        new_passed = new.passed

        if old_passed and not new_passed:
            regressions.append(
                CaseFlip(
                    case_id=case_id,
                    input_email_preview=old.input_email[:100],
                    expected_category=old.expected_category,
                    old_predicted=old.predicted_category,
                    new_predicted=new.predicted_category,
                    old_summary=old.predicted_summary,
                    new_summary=new.predicted_summary,
                    flip_type="regression",
                )
            )
        elif not old_passed and new_passed:
            improvements.append(
                CaseFlip(
                    case_id=case_id,
                    input_email_preview=new.input_email[:100],
                    expected_category=new.expected_category,
                    old_predicted=old.predicted_category,
                    new_predicted=new.predicted_category,
                    old_summary=old.predicted_summary,
                    new_summary=new.predicted_summary,
                    flip_type="improvement",
                )
            )

    # Per-category accuracy deltas
    per_category_deltas = {}
    baseline_cat_map = {ca.category.value: ca.accuracy for ca in baseline.per_category_accuracy}
    current_cat_map = {ca.category.value: ca.accuracy for ca in current.per_category_accuracy}

    for cat in set(list(baseline_cat_map.keys()) + list(current_cat_map.keys())):
        old_acc = baseline_cat_map.get(cat, 0.0)
        new_acc = current_cat_map.get(cat, 0.0)
        per_category_deltas[cat] = (new_acc - old_acc) * 100  # as percentage points

    # Overall accuracy delta
    accuracy_delta = (current.overall_accuracy - baseline.overall_accuracy) * 100

    # Summary score delta
    summary_delta = current.avg_summary_score - baseline.avg_summary_score

    # Latency delta
    latency_delta = current.avg_latency_ms - baseline.avg_latency_ms

    # Determine status based on thresholds
    abs_delta = abs(accuracy_delta)
    if accuracy_delta < 0:  # regression
        if abs_delta >= cfg.critical_threshold_pct:
            status = "critical"
        elif abs_delta >= cfg.warning_threshold_pct:
            status = "warning"
        else:
            status = "pass"
    else:
        status = "pass"

    comparison = RunComparison(
        baseline_run_id=baseline.metadata.run_id,
        current_run_id=current.metadata.run_id,
        baseline_accuracy=baseline.overall_accuracy,
        current_accuracy=current.overall_accuracy,
        accuracy_delta=accuracy_delta,
        per_category_deltas=per_category_deltas,
        regressions=regressions,
        improvements=improvements,
        status=status,
        summary_score_delta=summary_delta,
        latency_delta_ms=latency_delta,
    )

    logger.info(
        "Comparison: %s -> %s | accuracy delta: %+.1f%% | regressions: %d | improvements: %d | status: %s",
        baseline.metadata.run_id,
        current.metadata.run_id,
        accuracy_delta,
        len(regressions),
        len(improvements),
        status,
    )

    return comparison


def format_comparison_summary(comp: RunComparison) -> str:
    """Format a human-readable comparison summary."""
    lines = [
        f"═══ Eval Comparison: {comp.baseline_run_id} → {comp.current_run_id} ═══",
        f"",
        f"Status: {comp.status.upper()}",
        f"Overall Accuracy: {comp.baseline_accuracy:.1%} → {comp.current_accuracy:.1%} ({comp.accuracy_delta:+.1f}%)",
        f"Summary Score Delta: {comp.summary_score_delta:+.2f}",
        f"Latency Delta: {comp.latency_delta_ms:+.0f}ms",
        f"",
    ]

    if comp.per_category_deltas:
        lines.append("Per-Category Accuracy Deltas:")
        for cat, delta in sorted(comp.per_category_deltas.items()):
            indicator = "✅" if delta >= 0 else "❌"
            lines.append(f"  {indicator} {cat}: {delta:+.1f}%")
        lines.append("")

    if comp.regressions:
        lines.append(f"🔴 Regressions ({len(comp.regressions)}):")
        for r in comp.regressions:
            old_cat = r.old_predicted.value if r.old_predicted else "none"
            new_cat = r.new_predicted.value if r.new_predicted else "none"
            lines.append(
                f"  {r.case_id}: expected={r.expected_category.value} "
                f"was={old_cat} now={new_cat}"
            )
            lines.append(f"    email: {r.input_email_preview}...")
        lines.append("")

    if comp.improvements:
        lines.append(f"🟢 Improvements ({len(comp.improvements)}):")
        for i in comp.improvements:
            old_cat = i.old_predicted.value if i.old_predicted else "none"
            new_cat = i.new_predicted.value if i.new_predicted else "none"
            lines.append(
                f"  {i.case_id}: expected={i.expected_category.value} "
                f"was={old_cat} now={new_cat}"
            )
        lines.append("")

    return "\n".join(lines)
