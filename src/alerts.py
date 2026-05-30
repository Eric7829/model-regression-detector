"""Slack alerting layer using Block Kit formatting.

Formats evaluation run results and comparative differences into visually premium,
color-coded Slack messages, and delivers them asynchronously.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any

from src.models import EvalRunResult, RunComparison

logger = logging.getLogger(__name__)


def format_slack_blocks(result: EvalRunResult, comparison: RunComparison | None) -> list[dict[str, Any]]:
    """Format evaluation run results and baseline comparison into Slack Block Kit payload."""
    m = result.metadata
    
    # 1. Determine Verdict Status and Visual Emojis
    status = comparison.status if comparison else "pass"
    if status == "pass":
        title_emoji = "🟢"
        verdict_text = "*VERDICT: PASS (BUILD GREEN)*"
    elif status == "warning":
        title_emoji = "🟡"
        verdict_text = "*VERDICT: WARNING (QUALITY DEGRADATION)*"
    else:
        title_emoji = "🔴"
        verdict_text = "*VERDICT: CRITICAL FAILURE (REGRESSION DETECTED)*"

    blocks: list[dict[str, Any]] = []

    # 2. Header Block
    blocks.append(
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{title_emoji} Model Regression Detection Alert",
                "emoji": True,
            },
        }
    )

    # 3. Main Info Section
    meta_text = (
        f"*Prompt Version:* `{m.prompt_version}` | *Dataset Version:* `{m.dataset_version}`\n"
        f"*Model:* `{m.model}` | *Cost:* `${result.total_cost_estimate_usd:.5f}`\n"
        f"*Verdict:* {verdict_text}"
    )
    blocks.append(
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": meta_text},
        }
    )

    blocks.append({"type": "divider"})

    # 4. Core Metrics Section
    acc_val = result.overall_accuracy * 100
    acc_text = f"*{acc_val:.1f}%*"
    if comparison:
        base_acc = comparison.baseline_accuracy * 100
        curr_acc = comparison.current_accuracy * 100
        delta = comparison.accuracy_delta
        sign = "+" if delta > 0 else ""
        delta_emoji = "🔼" if delta > 0 else ("🔻" if delta < 0 else "")
        acc_text = f"*{curr_acc:.1f}%* (Baseline: {base_acc:.1f}% | Diff: *{sign}{delta:.1f}% {delta_emoji}*)"

    lat_text = f"*{result.avg_latency_ms:.1f} ms*"
    if comparison:
        l_delta = comparison.latency_delta_ms
        l_sign = "+" if l_delta > 0 else ""
        lat_text = f"*{result.avg_latency_ms:.1f} ms* (Delta: *{l_sign}{l_delta:.1f} ms*)"

    metrics_md = (
        f"• *Overall Accuracy:* {acc_text}\n"
        f"• *Average Latency:* {lat_text}\n"
        f"• *Cases Passed:* `{m.passed_cases}/{m.total_cases}` (`{m.failed_cases}` failed, `{m.error_cases}` errors)"
    )
    blocks.append(
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": metrics_md},
        }
    )

    blocks.append({"type": "divider"})

    # 5. Category Performance Block
    cat_lines = []
    for cat_acc in result.per_category_accuracy:
        cat_val = cat_acc.category.value
        curr_cat_acc = cat_acc.accuracy * 100
        cases_info = f"({cat_acc.correct}/{cat_acc.total})"
        
        if comparison and cat_val in comparison.per_category_deltas:
            c_delta = comparison.per_category_deltas[cat_val]
            c_sign = "+" if c_delta > 0 else ""
            c_emoji = "🔼" if c_delta > 0 else ("🔻" if c_delta < 0 else "")
            cat_lines.append(f"• *{cat_val}*: `{curr_cat_acc:.1f}%` {cases_info} | Diff: *{c_sign}{c_delta:.1f}% {c_emoji}*")
        else:
            cat_lines.append(f"• *{cat_val}*: `{curr_cat_acc:.1f}%` {cases_info}")
            
    blocks.append(
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Accuracy by Category:*\n" + "\n".join(cat_lines),
            },
        }
    )

    # 6. Flips / Regressions Section
    if comparison:
        regressions_blocks = []
        improvements_blocks = []
        
        for reg in comparison.regressions:
            old_pred = reg.old_predicted.value if reg.old_predicted else "error/fail"
            new_pred = reg.new_predicted.value if reg.new_predicted else "error/fail"
            regressions_blocks.append(
                f"• *[{reg.case_id}]* Flipped from *{old_pred}* (expected) to *{new_pred}* (predicted)\n"
                f"  _Preview:_ \"{reg.input_email_preview}\""
            )
            
        for imp in comparison.improvements:
            old_pred = imp.old_predicted.value if imp.old_predicted else "error/fail"
            new_pred = imp.new_predicted.value if imp.new_predicted else "error/fail"
            improvements_blocks.append(
                f"• *[{imp.case_id}]* Flipped from fail/error *{old_pred}* to correct *{new_pred}*"
            )

        if regressions_blocks or improvements_blocks:
            blocks.append({"type": "divider"})
            
            flips_text = []
            if regressions_blocks:
                flips_text.append(f"🔴 *Regressions ({len(comparison.regressions)}):*\n" + "\n".join(regressions_blocks))
            if improvements_blocks:
                flips_text.append(f"🟢 *Improvements ({len(comparison.improvements)}):*\n" + "\n".join(improvements_blocks))
                
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "\n\n".join(flips_text),
                    },
                }
            )

    # 7. Context Footer Block
    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Run ID: `{m.run_id}` | Finished: `{m.finished_at or ''}`",
                }
            ],
        }
    )

    return blocks


def send_slack_alert(
    result: EvalRunResult,
    comparison: RunComparison | None,
    webhook_url: str | None = None,
) -> bool:
    """Deliver a visual Slack Block Kit report to a Slack webhook URL.

    Parameters
    ----------
    webhook_url:
        The URL to post the notification to. If None, retrieves from the
        `SLACK_WEBHOOK_URL` environment variable.

    Returns
    -------
    bool
        True if the post succeeded, False otherwise.
    """
    url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL", "")
    if not url or url.startswith("https://hooks.slack.com/services/XXX"):
        logger.info("No active Slack Webhook URL found. Skipping Slack alerting.")
        return False

    blocks = format_slack_blocks(result, comparison)
    payload = {"blocks": blocks}

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            status_code = response.status
            if status_code in (200, 201):
                logger.info("Slack alert delivered successfully (status: %d)", status_code)
                return True
            else:
                logger.warning("Slack webhook returned error code: %d", status_code)
                return False
    except Exception as err:
        logger.error("Failed to deliver Slack webhook alert: %s", err)
        return False
