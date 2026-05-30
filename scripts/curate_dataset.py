"""Curate the golden dataset from the raw emails CSV.

Maps the Aetheros support email dataset to our classifier's 4 categories:
- technical: Mercury Language bugs, API errors, Cloud deployment failures
- account: IAM service issues (permissions, roles, user access)  
- billing: Billing/pricing/subscription (sparse in dataset, some synthesized)
- general: Feature requests, suggestions, general inquiries

Selects first customer email per thread for diversity, balances across
categories and difficulty levels, and outputs versioned JSON.
"""

import csv
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path


RAW_CSV = Path(__file__).resolve().parent.parent / "emails" / "dataset.csv"
OUTPUT = Path(__file__).resolve().parent.parent / "data" / "golden_dataset_v1.0.0.json"


def categorize_email(row: dict) -> tuple[str, str]:
    """Map dataset columns to our classifier category + reasoning."""
    email_type = row["email_types"].lower()
    products = row["product_types"].lower()
    criticality = row["email_criticality"].lower()
    subject = row["subject"].lower()
    body = row["message_body"].lower()

    # Account: IAM-related (permissions, roles, access, login)
    if "iam" in products and ("permission" in body or "role" in body or "access" in body or "login" in body or "user" in body):
        return "account", "IAM/access related issue"

    # Technical: bugs, errors, crashes, deployment failures, API issues
    if any(kw in body for kw in ["error", "bug", "crash", "failing", "502", "500", "timeout", "latency spike", "unreachable", "outage", "type mismatch"]):
        if "iam" not in products:
            return "technical", "Technical error or system issue"

    # Technical: Mercury Language code questions
    if "mercury" in products and ("code" in body or "function" in body or "type" in body or "documentation" in body or "data type" in body):
        if "suggestion" not in email_type:
            return "technical", "Mercury language technical question"

    # Technical: Cloud/API deployment issues
    if ("cloud" in products or "api" in products) and any(kw in body for kw in ["deploy", "server", "instance", "endpoint", "integration"]):
        if "suggestion" not in email_type:
            return "technical", "Infrastructure/API technical issue"

    # General: suggestions, feature requests, feedback
    if "suggestion" in email_type:
        return "general", "Feature suggestion or feedback"

    # General: general inquiries not about bugs
    if "inquiry" in email_type and not any(kw in body for kw in ["error", "fail", "bug", "crash", "broken"]):
        return "general", "General inquiry"

    # Fallback to technical for remaining issues
    if "issue" in email_type:
        return "technical", "Reported issue (fallback)"

    return "general", "Uncategorized (fallback)"


def estimate_difficulty(row: dict, category: str) -> str:
    """Estimate test case difficulty based on email characteristics."""
    body = row["message_body"]
    products = row["product_types"].lower()
    criticality = row["email_criticality"].lower()
    email_type = row["email_types"].lower()

    # Adversarial: very short, ambiguous, or multi-product
    if len(body.strip()) < 100:
        return "hard"
    if products.count(",") >= 2:
        return "hard"

    # Hard: ambiguous cases (e.g., IAM issues that could be technical)
    if "iam" in products and any(kw in body.lower() for kw in ["api", "endpoint", "deploy"]):
        return "hard"
    if "suggestion" in email_type and "issue" in body.lower():
        return "medium"

    # Easy: clear-cut single-product cases
    if products.count(",") == 0 and len(body) > 200:
        if category in ("technical", "general"):
            return "easy"

    return "medium"


def make_case_id(idx: int) -> str:
    return f"TC-{idx:03d}"


def build_expected_summary(row: dict) -> str:
    """Build a reference summary from subject + body context."""
    subject = row["subject"].strip()
    body = row["message_body"].strip()

    # Take first 2 sentences of the body as summary basis
    sentences = body.replace("\n", " ").split(". ")
    core = ". ".join(sentences[:2]).strip()
    if len(core) > 250:
        core = core[:247] + "..."
    if not core.endswith("."):
        core += "."
    return core


def select_first_customer_emails(rows: list[dict]) -> list[dict]:
    """Select only the first customer email per thread for diversity."""
    seen_threads = set()
    selected = []
    for row in rows:
        # Skip support replies
        if row["sender"] == "support@aetheros.com":
            continue
        thread = row["thread_id"]
        if thread not in seen_threads:
            seen_threads.add(thread)
            selected.append(row)
    return selected


def curate_golden_dataset():
    with open(RAW_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    print(f"Total rows in CSV: {len(all_rows)}")

    # Get first customer email per thread
    candidates = select_first_customer_emails(all_rows)
    print(f"Unique first-customer emails (threads): {len(candidates)}")

    # Categorize all candidates
    categorized = []
    for row in candidates:
        category, reason = categorize_email(row)
        difficulty = estimate_difficulty(row, category)
        categorized.append((row, category, difficulty, reason))

    # Count by category
    cat_counts = {}
    for _, cat, _, _ in categorized:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    print(f"Category distribution (all candidates): {cat_counts}")

    # Select balanced subset: ~25 per category, preferring diversity
    target_per_cat = 25
    selected_by_cat: dict[str, list] = {c: [] for c in ["billing", "technical", "account", "general"]}

    # Sort by diversity of difficulty
    for row, cat, diff, reason in categorized:
        if len(selected_by_cat[cat]) < target_per_cat:
            selected_by_cat[cat].append((row, cat, diff, reason))

    # We won't have enough billing cases — let's synthesize them
    billing_emails = [
        {
            "body": "Hi, I was charged $49.99 twice this month for my Pro plan subscription. Can you please look into this and refund the duplicate charge? My credit card ending in 4242 was charged on Oct 15 and Oct 16.",
            "subject": "Double charge on Pro plan",
            "difficulty": "easy",
            "notes": "Clear billing case — duplicate charge with specific details",
        },
        {
            "body": "I'd like to upgrade from the Basic plan to Enterprise. Can you walk me through the pricing differences and what additional features I'd get? Also, is there a discount for annual billing?",
            "subject": "Plan upgrade inquiry",
            "difficulty": "medium",
            "notes": "Could be general inquiry but primary intent is pricing/subscription change",
        },
        {
            "body": "Our invoice #INV-2023-0847 shows a charge for 15 API seats but we only have 12 active users. Please correct this and issue a credit for the 3 extra seats.",
            "subject": "Incorrect invoice - wrong seat count",
            "difficulty": "easy",
            "notes": "Clear billing dispute with specific invoice reference",
        },
        {
            "body": "I cancelled my subscription last month but I'm still being charged. This is the third time I've reached out about this. I need a full refund for the charges since cancellation and confirmation that no further charges will occur.",
            "subject": "Still being charged after cancellation",
            "difficulty": "easy",
            "notes": "Recurring billing issue with escalation tone",
        },
        {
            "body": "Can you explain the difference between the per-seat and per-usage pricing models? We're trying to figure out which would be more cost-effective for our team of 50 developers who use the API intermittently.",
            "subject": "Pricing model comparison",
            "difficulty": "medium",
            "notes": "Pricing inquiry that could be mistaken for general",
        },
        {
            "body": "We need to add 5 more seats to our account but the self-service portal isn't letting me. It says 'payment method expired'. Can you update our payment info and process the seat addition?",
            "subject": "Cannot add seats - payment issue",
            "difficulty": "medium",
            "notes": "Mix of billing (payment) and account (portal access) — billing is primary",
        },
        {
            "body": "hi i see a charge i dont recognize on my statement from aetheros for 199.99. i dont think i signed up for anything. pls help",
            "subject": "unknown charge",
            "difficulty": "hard",
            "notes": "Informal writing, no details, could be fraud or forgotten subscription",
        },
        {
            "body": "Our finance team needs a W-9 form and a detailed breakdown of all charges for Q3 2023 for our audit. Can you provide these documents?",
            "subject": "Tax documents and Q3 billing breakdown",
            "difficulty": "hard",
            "notes": "Financial/admin request — billing-adjacent but could be classified as general",
        },
        {
            "body": "The free trial ended but I'm locked out AND being asked to pay. I thought the trial was 30 days but it's only been 14 days. Either extend my trial or let me at least export my data before I decide whether to subscribe.",
            "subject": "Trial ended early + billing confusion",
            "difficulty": "hard",
            "notes": "Overlaps billing (trial/payment), account (locked out), and general (data export). Billing is primary.",
        },
        {
            "body": "Necesito una factura con RFC para mi empresa en México. El número de cuenta es MX-2847. También quiero saber si aceptan pagos en pesos mexicanos.",
            "subject": "Factura y método de pago",
            "difficulty": "adversarial",
            "notes": "Spanish language billing request — tests multilingual handling",
        },
        {
            "body": "refund",
            "subject": "refund",
            "difficulty": "adversarial",
            "notes": "Extremely terse — tests classifier on minimal input",
        },
        {
            "body": "I want to downgrade my plan but I'm worried I'll lose access to the analytics dashboard I set up. If I downgrade, will my saved dashboards and API monitoring configurations be preserved, or will they be deleted?",
            "subject": "Downgrade plan - data retention question",
            "difficulty": "hard",
            "notes": "Mix of billing (downgrade) and technical (data retention) — billing is primary intent",
        },
    ]

    # Build the golden cases
    cases = []
    idx = 1

    # Add billing cases (synthesized)
    for be in billing_emails:
        tags = ["synthesized"]
        if be["difficulty"] in ("hard", "adversarial"):
            tags.append("edge-case")
        if "español" in be.get("body", "").lower() or "factura" in be.get("body", "").lower():
            tags.append("multilingual")
        if len(be["body"]) < 20:
            tags.append("minimal-input")

        cases.append({
            "id": make_case_id(idx),
            "input_email": be["body"],
            "expected_category": "billing",
            "expected_summary": be["body"][:200].strip() + ("..." if len(be["body"]) > 200 else ""),
            "expected_difficulty": be["difficulty"],
            "tags": tags,
            "notes": be["notes"],
        })
        idx += 1

    # Add real emails from other categories
    for cat in ["technical", "account", "general"]:
        for row, _, diff, reason in selected_by_cat[cat]:
            body = row["message_body"].strip()
            subject = row["subject"].strip()

            email_type_clean = row['email_types'].strip('[]').replace("'", "")
            tags = [f"source:{email_type_clean}"]
            if diff in ("hard", "adversarial"):
                tags.append("edge-case")
            if row["product_types"].count(",") >= 2:
                tags.append("multi-product")
            if row["email_criticality"] == "high":
                tags.append("high-priority")
            if len(body) < 100:
                tags.append("short-email")

            summary = build_expected_summary(row)

            cases.append({
                "id": make_case_id(idx),
                "input_email": body,
                "expected_category": cat,
                "expected_summary": summary,
                "expected_difficulty": diff,
                "tags": tags,
                "notes": f"{reason}. Subject: '{subject}'. Products: {row['product_types']}",
            })
            idx += 1

    # Add some additional adversarial/edge cases
    adversarial_cases = [
        {
            "id": make_case_id(idx),
            "input_email": "This product is absolute garbage. Nothing works. I can't log in, I can't see my invoices, and the API returns errors half the time. I want a full refund and I'm cancelling everything.",
            "expected_category": "billing",
            "expected_summary": "Customer expresses extreme frustration across multiple issues (login, invoices, API errors) and demands a refund and cancellation.",
            "expected_difficulty": "adversarial",
            "tags": ["synthesized", "edge-case", "multi-category", "angry-customer"],
            "notes": "Touches billing (refund), account (login), and technical (API errors). Billing/cancellation is the primary actionable request.",
        },
        {
            "id": make_case_id(idx + 1),
            "input_email": "lol ur api is down again 😂 fix it pls thx",
            "expected_category": "technical",
            "expected_summary": "Customer reports API downtime in informal language.",
            "expected_difficulty": "adversarial",
            "tags": ["synthesized", "edge-case", "informal", "emoji"],
            "notes": "Tests handling of extremely informal language with emoji",
        },
        {
            "id": make_case_id(idx + 2),
            "input_email": "I accidentally deleted my admin account and now nobody on our team can manage permissions or access the billing portal. We need this restored ASAP - we have 200 employees affected.",
            "expected_category": "account",
            "expected_summary": "Customer accidentally deleted their admin account, leaving the entire team unable to manage permissions or access billing.",
            "expected_difficulty": "hard",
            "tags": ["synthesized", "edge-case", "multi-category"],
            "notes": "Primarily account issue (deleted admin) though it also impacts billing access. Account restoration is the critical action.",
        },
        {
            "id": make_case_id(idx + 3),
            "input_email": "Dear Support,\n\nI hope this message finds you well. I wanted to reach out regarding a matter that has been on my mind. You see, I've been using your platform for quite some time now, and while I generally find it satisfactory, there are certain aspects that I believe could benefit from refinement. Specifically, I think the onboarding experience for new team members could be smoother, and the documentation, while comprehensive, sometimes lacks practical examples that would help less technical users get started more quickly. I don't have any urgent issues — just wanted to share my thoughts as a long-time customer.\n\nBest regards,\nA satisfied customer",
            "expected_category": "general",
            "expected_summary": "Long-time customer provides unsolicited feedback about improving the onboarding experience and documentation with more practical examples.",
            "expected_difficulty": "medium",
            "tags": ["synthesized", "verbose", "feedback"],
            "notes": "Very wordy email with no specific issue — tests whether classifier correctly identifies general feedback vs. a reportable problem",
        },
        {
            "id": make_case_id(idx + 4),
            "input_email": "Subject: RE: RE: RE: FW: Original ticket #4847\n\n+1 to what Sarah said below. Also tagging @mike for visibility.\n\n---\nForwarded message not included",
            "expected_category": "general",
            "expected_summary": "Forwarded thread reply with no substantive content — customer agrees with a previous message and tags a colleague.",
            "expected_difficulty": "adversarial",
            "tags": ["synthesized", "edge-case", "no-content", "forwarded"],
            "notes": "Email with almost no actionable content — tests classifier on context-free forwarded threads",
        },
    ]
    cases.extend(adversarial_cases)

    # Build the dataset
    dataset = {
        "version": "v1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": "Initial golden dataset curated from Aetheros support emails CSV with synthesized billing cases and adversarial edge cases. 4 categories: billing, technical, account, general.",
        "cases": cases,
    }

    # Stats
    cat_final = {}
    diff_final = {}
    for c in cases:
        cat_final[c["expected_category"]] = cat_final.get(c["expected_category"], 0) + 1
        diff_final[c["expected_difficulty"]] = diff_final.get(c["expected_difficulty"], 0) + 1

    print(f"\nGolden dataset stats:")
    print(f"  Total cases: {len(cases)}")
    print(f"  By category: {cat_final}")
    print(f"  By difficulty: {diff_final}")

    # Write output
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nWritten to: {OUTPUT}")


if __name__ == "__main__":
    curate_golden_dataset()
