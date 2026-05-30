#!/usr/bin/env python
"""PR Comment Poster for CI/CD pipelines.

Loads the latest evaluation run result, performs a comparison check against the latest baseline,
formats a beautifully premium Markdown report, and posts it directly onto the current
GitHub Pull Request using the GitHub REST API.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

# Ensure the root project directory is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.alerts import format_slack_blocks  # reusable blocks logic if needed, but we build Markdown
from src.eval_diff import compare_runs
from src.models import EvalConfig
from src.storage import EvaluationStorage


def build_pr_comment_markdown(run_result, comparison) -> str:
    """Build a rich, visual Markdown comment for pull requests."""
    m = run_result.metadata
    status = comparison.status if comparison else "pass"
    
    # 1. Color-coded Header Banner
    if status == "pass":
        verdict_banner = "### 🟢 **PROMPT EVALUATION: PASSED (BUILD GREEN)**"
    elif status == "warning":
        verdict_banner = "### 🟡 **PROMPT EVALUATION: WARNING (QUALITY DROPPED)**"
    else:
        verdict_banner = "### 🔴 **PROMPT EVALUATION: CRITICAL QUALITY FAILURE (REGRESSION DETECTED)**"

    # 2. Key Metadata Block
    md = [
        verdict_banner,
        "",
        "| Configuration | Value |",
        "| :--- | :--- |",
        f"| **Prompt Version** | `{m.prompt_version}` |",
        f"| **Golden Dataset** | `{m.dataset_version}` |",
        f"| **Classifier Model** | `{m.model}` |",
        f"| **Estimated Cost** | `${run_result.total_cost_estimate_usd:.5f}` |",
        f"| **Run Timestamp** | `{m.finished_at or m.started_at}` |",
        "",
        "---",
        "",
        "#### 📊 **Key Performance Metrics**",
        "",
    ]

    # 3. Metrics Table
    acc_text = f"**{run_result.overall_accuracy * 100:.1f}%**"
    if comparison:
        base_acc = comparison.baseline_accuracy * 100
        curr_acc = comparison.current_accuracy * 100
        sign = "+" if comparison.accuracy_delta > 0 else ""
        delta_emoji = "🔼" if comparison.accuracy_delta > 0 else ("🔻" if comparison.accuracy_delta < 0 else "")
        acc_text = f"**{curr_acc:.1f}%** (Baseline: {base_acc:.1f}% | Diff: **{sign}{comparison.accuracy_delta:.1f}% {delta_emoji}**)"

    lat_text = f"**{run_result.avg_latency_ms:.1f} ms**"
    if comparison:
        l_sign = "+" if comparison.latency_delta_ms > 0 else ""
        lat_text = f"**{run_result.avg_latency_ms:.1f} ms** (Delta: **{l_sign}{comparison.latency_delta_ms:.1f} ms**)"

    md.extend([
        f"- **Overall Accuracy:** {acc_text}",
        f"- **Average Latency:** {lat_text}",
        f"- **Passed Cases:** `{m.passed_cases}/{m.total_cases}` (`{m.failed_cases}` failed, `{m.error_cases}` errors)",
        "",
        "---",
        "",
        "#### 📂 **Accuracy by customer Intent Category**",
        "",
        "| Category | Accuracy | Diff vs. Baseline |",
        "| :--- | :--- | :--- |",
    ])

    # 4. Category Breakdowns Table
    for cat_acc in run_result.per_category_accuracy:
        cat_val = cat_acc.category.value
        curr_cat_acc = cat_acc.accuracy * 100
        cases_info = f"({cat_acc.correct}/{cat_acc.total})"
        
        diff_str = "no change"
        if comparison and cat_val in comparison.per_category_deltas:
            c_delta = comparison.per_category_deltas[cat_val]
            c_sign = "+" if c_delta > 0 else ""
            c_emoji = "🔼" if c_delta > 0 else ("🔻" if c_delta < 0 else "")
            diff_str = f"**{c_sign}{c_delta:.1f}% {c_emoji}**" if c_delta != 0 else "no change"

        md.append(f"| `{cat_val}` | {curr_cat_acc:.1f}% {cases_info} | {diff_str} |")

    # 5. Behavioral Flips (Regressions & Improvements)
    if comparison and (comparison.regressions or comparison.improvements):
        md.extend([
            "",
            "---",
            "",
            "#### 🔄 **Behavioral Change Detection (Flips)**",
            "",
        ])
        
        if comparison.improvements:
            md.extend([
                "##### 🟢 **Improvements**",
                "",
                "| Case ID | Expected | Flipped From | Flipped To |",
                "| :--- | :--- | :--- | :--- |",
            ])
            for imp in comparison.improvements:
                old = imp.old_predicted.value if imp.old_predicted else "error/fail"
                new = imp.new_predicted.value if imp.new_predicted else "error/fail"
                md.append(f"| **{imp.case_id}** | `{imp.expected_category.value}` | `{old}` | **`{new}`** |")
            md.append("")
            
        if comparison.regressions:
            md.extend([
                "##### 🔴 **Regressions**",
                "",
                "| Case ID | Expected | Flipped From | Flipped To | Email Preview |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ])
            for reg in comparison.regressions:
                old = reg.old_predicted.value if reg.old_predicted else "error/fail"
                new = reg.new_predicted.value if reg.new_predicted else "error/fail"
                md.append(f"| **{reg.case_id}** | `{reg.expected_category.value}` | **`{old}`** | `{new}` | \"_{reg.input_email_preview}_\" |")
            md.append("")

    # 6. Bottom Banner / Context
    md.extend([
        "---",
        f"_*Automated PromptOps pipeline evaluation report. Run ID:* `{m.run_id}`_"
    ])

    return "\n".join(md)


def main():
    print("Formatting Pull Request comment...")
    storage = EvaluationStorage()
    
    # 1. Load latest run
    latest_run = storage.get_latest_run()
    if not latest_run:
        print("Error: No evaluation runs found in storage database. Cannot post comment.")
        sys.exit(1)
        
    # 2. Get baseline comparison (latest previous run on same dataset version)
    baseline_run = storage.get_latest_run(dataset_version=latest_run.metadata.dataset_version)
    
    # If the latest run and baseline run are the same ID, query previous run
    runs = storage.list_runs(limit=5)
    matching_runs = [r for r in runs if r["dataset_version"] == latest_run.metadata.dataset_version]
    
    comparison = None
    if len(matching_runs) >= 2:
        # Load second newest run
        baseline_id = matching_runs[1]["run_id"]
        baseline_run = storage.load_run(baseline_id)
        if baseline_run:
            comparison = compare_runs(baseline_run, latest_run)
            print(f"Compared run '{latest_run.metadata.run_id}' against baseline '{baseline_run.metadata.run_id}'.")
    else:
        print("No prior baseline run found of the same dataset. Generating zero-baseline report.")

    # 3. Build Markdown content
    pr_comment = build_pr_comment_markdown(latest_run, comparison)
    
    # 4. Check CI context vs Local mock
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    
    if token and repo and event_path:
        print("GitHub Actions environment detected. Posting comment to pull request...")
        
        try:
            with open(event_path, "r", encoding="utf-8") as f:
                event_data = json.load(f)
                
            # For pull_request event, pull request number is in "number"
            pr_number = event_data.get("number")
            if not pr_number:
                # If triggered by a push event, check if there's a commit comment or fallback
                print("No active pull request number found in GitHub event. Skipping PR comment.")
                sys.exit(0)
                
            url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
            
            payload = {"body": pr_comment}
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"token {token}",
                    "Accept": "application/vnd.github.v3+json",
                    "Content-Type": "application/json",
                    "User-Agent": "Model-Regression-Detector-CI",
                },
            )
            
            with urllib.request.urlopen(req) as resp:
                status = resp.status
                if status in (200, 201):
                    print(f"Successfully posted PR comment to {repo} PR #{pr_number}!")
                else:
                    print(f"Error: GitHub API returned status code {status}")
                    
        except Exception as e:
            print(f"Error posting PR comment: {e}")
            sys.exit(1)
    else:
        print("\n=== LOCAL PREVIEW: MOCK PR COMMENT ===")
        try:
            print(pr_comment)
        except UnicodeEncodeError:
            # Fallback for older Windows terminals with restricted cp1252/ascii encodings
            safe_text = pr_comment.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8')
            print(safe_text)
        print("======================================\n")


if __name__ == "__main__":
    main()
