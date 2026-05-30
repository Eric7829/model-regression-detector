#!/usr/bin/env python
"""Evaluation CLI runner script.

Executes an evaluation run of the classifier on the golden dataset, persist results,
compares with the latest baseline run, and outputs a formatted ASCII report.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Ensure the root project directory is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset_loader import (
    list_dataset_versions,
    load_golden_dataset,
    load_latest_golden_dataset,
)
from src.eval_diff import compare_runs
from src.eval_runner import EvalRunner
from src.models import EvalConfig
from src.prompt_loader import (
    list_prompt_versions,
    load_latest_prompt,
    load_prompt,
)
from src.storage import EvaluationStorage

# ANSI Colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def setup_cli_logging():
    """Configure minimal clean CLI logging."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.basicConfig(level=logging.WARNING, handlers=[handler])
    # Set classifier and runner log levels to INFO so we see status
    logging.getLogger("src.eval_runner").setLevel(logging.INFO)


def format_delta_pct(delta: float) -> str:
    """Format a percentage delta with arrow indicators and colors."""
    if delta > 0:
        return f"{GREEN}+{delta:.1f}% (+){RESET}"
    elif delta < 0:
        return f"{RED}{delta:.1f}% (-){RESET}"
    else:
        return f"{YELLOW}0.0% (no change){RESET}"


def format_delta_value(delta: float, inverted: bool = False) -> str:
    """Format a raw float delta (e.g. latency, score) with colors.

    Parameters
    ----------
    inverted:
        If True, negative is better (e.g. latency, tokens).
    """
    if delta == 0:
        return "no change"
        
    better = delta < 0 if inverted else delta > 0
    color = GREEN if better else RED
    sign = "+" if delta > 0 else ""
    
    return f"{color}{sign}{delta:.2f}{RESET}"


def print_report(run_res, comparison=None, skip_judge: bool = False):
    """Print a highly polished, detailed CLI evaluation report."""
    m = run_res.metadata
    
    print("\n" + "=" * 80)
    print(f" {BOLD}{BLUE}LLM FEATURE EVALUATION REPORT — RUN {m.run_id}{RESET} ")
    print("=" * 80)
    
    # Metadata
    print(f" {BOLD}Prompt Version:{RESET}  {m.prompt_version:<20} | {BOLD}Dataset Version:{RESET} {m.dataset_version}")
    print(f" {BOLD}Classifier Model:{RESET} {m.model:<20} | {BOLD}Started At:{RESET}      {m.started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f" {BOLD}Total Cases:{RESET}     {m.total_cases:<20} | {BOLD}Execution Cost:{RESET}  ${run_res.total_cost_estimate_usd:.5f}")
    print("-" * 80)
    
    # Cases stats
    pass_col = GREEN if m.passed_cases > 0 else RESET
    err_col = RED if m.error_cases > 0 else RESET
    print(f" {BOLD}Passed Cases:{RESET} {pass_col}{m.passed_cases:<5}{RESET} | "
          f"{BOLD}Failed Cases:{RESET} {m.failed_cases:<5} | "
          f"{BOLD}API Errors:{RESET}   {err_col}{m.error_cases:<5}{RESET}")
    print("-" * 80)

    # Core Metrics Block
    print(f" {BOLD}{CYAN}KEY PERFORMANCE METRICS{RESET}")
    
    # Accuracy Row
    acc_str = f"{run_res.overall_accuracy * 100:.1f}%"
    if comparison:
        base_acc = comparison.baseline_accuracy * 100
        curr_acc = comparison.current_accuracy * 100
        delta_str = format_delta_pct(comparison.accuracy_delta)
        print(f"   * {BOLD}Overall Accuracy:{RESET} {curr_acc:.1f}% (Baseline: {base_acc:.1f}% | Diff: {delta_str})")
    else:
        print(f"   * {BOLD}Overall Accuracy:{RESET} {acc_str}")

    # Latency Row
    lat_str = f"{run_res.avg_latency_ms:.1f} ms"
    if comparison:
        delta_val = format_delta_value(comparison.latency_delta_ms, inverted=True)
        print(f"   * {BOLD}Average Latency:{RESET}  {run_res.avg_latency_ms:.1f} ms (Delta: {delta_val} ms)")
    else:
        print(f"   * {BOLD}Average Latency:{RESET}  {lat_str}")

    # Summary score Row
    if not skip_judge:
        score_str = f"{run_res.avg_summary_score:.2f} / 5.0"
        if comparison:
            delta_val = format_delta_value(comparison.summary_score_delta, inverted=False)
            print(f"   * {BOLD}Summary Quality (Judge):{RESET} {score_str} (Delta: {delta_val})")
        else:
            print(f"   * {BOLD}Summary Quality (Judge):{RESET} {score_str}")
            
    print("-" * 80)

    # Per-Category Accuracy
    print(f" {BOLD}{CYAN}ACCURACY BY CUSTOMER INTENT CATEGORY{RESET}")
    for cat_acc in run_res.per_category_accuracy:
        cat_val = cat_acc.category.value
        curr_cat_acc = cat_acc.accuracy * 100
        cases_info = f"({cat_acc.correct}/{cat_acc.total})"
        
        if comparison and cat_val in comparison.per_category_deltas:
            delta_pct = comparison.per_category_deltas[cat_val]
            delta_str = format_delta_pct(delta_pct)
            print(f"   * {cat_val:<12}: {curr_cat_acc:.1f}% {cases_info:<8} | Diff: {delta_str}")
        else:
            print(f"   * {cat_val:<12}: {curr_cat_acc:.1f}% {cases_info}")
    print("-" * 80)

    # Behavioral Flips: Regressions & Improvements
    if comparison:
        reg_count = len(comparison.regressions)
        imp_count = len(comparison.improvements)
        
        if reg_count > 0 or imp_count > 0:
            print(f" {BOLD}{CYAN}BEHAVIORAL CHANGE DETECTION (FLIPS){RESET}")
            
            # Improvements
            if imp_count > 0:
                print(f"\n   {BOLD}{GREEN}[+] Improvements ({imp_count}){RESET}")
                for flip in comparison.improvements:
                    print(f"     - [{GREEN}{flip.case_id}{RESET}] Expected: {flip.expected_category.value:<10} | "
                          f"Flipped from: {flip.old_predicted} -> {GREEN}{flip.new_predicted}{RESET}")
                          
            # Regressions
            if reg_count > 0:
                print(f"\n   {BOLD}{RED}[-] Regressions ({reg_count}){RESET}")
                for flip in comparison.regressions:
                    old_pred_str = flip.old_predicted.value if flip.old_predicted else "error/fail"
                    new_pred_str = flip.new_predicted.value if flip.new_predicted else "error/fail"
                    print(f"     - [{RED}{flip.case_id}{RESET}] Expected: {flip.expected_category.value:<10} | "
                          f"Flipped from: {GREEN}{old_pred_str}{RESET} -> {RED}{new_pred_str}{RESET}")
                    print(f"       {BOLD}Email Preview:{RESET} \"{flip.input_email_preview}\"")
                    if flip.old_summary and flip.new_summary:
                        print(f"       {BOLD}Old Summary:{RESET}   {flip.old_summary}")
                        print(f"       {BOLD}New Summary:{RESET}   {flip.new_summary}")
            print("-" * 80)
        else:
            print(f" {BOLD}{GREEN}No behavioral regressions or improvements detected. Consistency matches 100%!{RESET}")
            print("-" * 80)

    # Verdict Box
    status = comparison.status if comparison else "pass"
    if status == "pass":
        print(f"\n {BOLD}{GREEN}========================================================================{RESET}")
        print(f" {BOLD}{GREEN}                     VERDICT: PASS (BUILD GREEN)                         {RESET}")
        print(f" {BOLD}{GREEN}========================================================================{RESET}\n")
    elif status == "warning":
        print(f"\n {BOLD}{YELLOW}========================================================================{RESET}")
        print(f" {BOLD}{YELLOW}                     VERDICT: WARNING (QUALITY DROPPED)                  {RESET}")
        print(f" {BOLD}{YELLOW}========================================================================{RESET}\n")
    else:
        print(f"\n {BOLD}{RED}========================================================================{RESET}")
        print(f" {BOLD}{RED}                     VERDICT: CRITICAL FAILURE (REGRESSION DETECTED)     {RESET}")
        print(f" {BOLD}{RED}========================================================================{RESET}\n")


async def main():
    # Load env vars from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Run evaluation suite on LLM customer support email classifier."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="YAML prompt configuration filename in /prompts (default: latest)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="JSON golden dataset filename in /data (default: latest)."
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Run ID of the baseline to compare against (default: latest matching dataset)."
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Override max concurrent API requests."
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip LLM-as-a-judge summary relevance grading to save API cost/time."
    )
    parser.add_argument(
        "--warning-threshold",
        type=float,
        default=3.0,
        help="Flag warning if overall accuracy drops by this percentage (default: 3.0)."
    )
    parser.add_argument(
        "--critical-threshold",
        type=float,
        default=8.0,
        help="Flag critical failure if overall accuracy drops by this percentage (default: 8.0)."
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gemini-2.0-flash-lite",
        help="Model to use for summary evaluation (default: gemini-2.0-flash-lite)."
    )
    parser.add_argument(
        "--retry-count",
        type=int,
        default=2,
        help="API retry attempts on transient failure (default: 2)."
    )

    args = parser.parse_args()
    setup_cli_logging()

    # 1. Load Configurations
    try:
        if args.prompt:
            prompt_config = load_prompt(args.prompt)
        else:
            prompt_config = load_latest_prompt()
            
        if args.dataset:
            dataset = load_golden_dataset(args.dataset)
        else:
            dataset = load_latest_golden_dataset()
    except Exception as e:
        print(f"{RED}{BOLD}Configuration Error:{RESET} {e}", file=sys.stderr)
        sys.exit(2)

    # 2. Build Eval Configurations
    eval_config = EvalConfig(
        warning_threshold_pct=args.warning_threshold,
        critical_threshold_pct=args.critical_threshold,
        max_concurrency=args.concurrency if args.concurrency is not None else 5,
        judge_model=args.judge_model,
        retry_count=args.retry_count,
    )

    # 3. Setup Storage
    storage = EvaluationStorage()

    # 4. Fetch Baseline if needed
    baseline_run = None
    if args.baseline:
        baseline_run = storage.load_run(args.baseline)
        if not baseline_run:
            print(f"{YELLOW}Warning: Specified baseline run ID '{args.baseline}' not found in database. Running without baseline comparison.{RESET}")
    else:
        # Auto-match latest previous run on the exact same dataset
        baseline_run = storage.get_latest_run(dataset_version=dataset.version)

    # 5. Run Evaluation
    print(f"\n{BOLD}{CYAN}Initializing evaluation runner...{RESET}")
    print(f"  -> Prompt version:  {BOLD}{prompt_config.version}{RESET} ({prompt_config.model})")
    print(f"  -> Dataset version: {BOLD}{dataset.version}{RESET} ({dataset.case_count} cases)")
    if baseline_run:
        print(f"  -> Baseline run:    {BOLD}{baseline_run.metadata.run_id}{RESET} (Accuracy: {baseline_run.overall_accuracy * 100:.1f}%)")
    else:
        print(f"  -> Baseline run:    {YELLOW}None found (first run of this dataset){RESET}")
        
    runner = EvalRunner(eval_config)
    
    # Progress counter for live CLI feedback
    completed_count = 0
    total_count = dataset.case_count
    
    def progress_cb(case_res):
        nonlocal completed_count
        completed_count += 1
        pct = (completed_count / total_count) * 100
        sys.stdout.write(f"\r  Progress: [{completed_count}/{total_count}] {pct:.1f}% completed... ")
        sys.stdout.flush()

    try:
        run_result = await runner.run_evaluation(
            dataset=dataset,
            prompt_config=prompt_config,
            skip_judge=args.skip_judge,
            progress_callback=progress_cb,
        )
        print(f"\n{GREEN}{BOLD}Evaluation finished successfully.{RESET}")
    except Exception as e:
        print(f"\n{RED}{BOLD}Critical Execution Error:{RESET} {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 6. Save Run
    try:
        storage.save_run(run_result)
        print(f"Saved run results to SQLite & JSON records.")
    except Exception as e:
        print(f"{YELLOW}Warning: Failed to save run results: {e}{RESET}")

    # 7. Compare with Baseline
    comparison = None
    if baseline_run:
        try:
            comparison = compare_runs(baseline_run, run_result, eval_config)
        except Exception as e:
            print(f"{YELLOW}Warning: Failed to compare runs: {e}{RESET}")

    # 7.5 Deliver Slack Alert
    import json
    from src.alerts import send_slack_alert, format_slack_blocks
    slack_webhook = os.environ.get("SLACK_WEBHOOK_URL", "")
    if slack_webhook and not slack_webhook.startswith("https://hooks.slack.com/services/XXX"):
        print(f"\n{BOLD}{BLUE}Posting evaluation results to Slack...{RESET}")
        alert_sent = send_slack_alert(run_result, comparison, slack_webhook)
        if alert_sent:
            print(f"{GREEN}Slack alert posted successfully!{RESET}")
        else:
            print(f"{RED}Failed to post Slack alert. Check error logs.{RESET}")
    else:
        print(f"\n{YELLOW}Note: No active Slack Webhook configured (SLACK_WEBHOOK_URL is empty or placeholder).{RESET}")
        print(f"{CYAN}Mock Slack Block Kit payload preview:{RESET}")
        blocks = format_slack_blocks(run_result, comparison)
        print(json.dumps({"blocks": blocks}, indent=2))

    # 8. Print Report
    print_report(run_result, comparison, args.skip_judge)

    # 9. Return Exit Code
    if comparison and comparison.status == "critical":
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
