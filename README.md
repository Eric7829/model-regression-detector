# Model Regression Detection System (PromptOps Pipeline)

This system is an automated, continuous integration (CI/CD) evaluation and regression testing pipeline designed to protect LLM-powered features from quality degradation. It version-controls system prompts in Git like code, continuously tests model predictions against a highly curated, versioned golden dataset, scores results across multiple dimensions (exact category matching, latency, cost, and LLM-as-a-judge semantic summary evaluation), saves detailed run metrics in a dual-storage layer (SQLite + git-friendly JSON), and alerts engineers on Slack or blocks Git merges when quality declines.

---

## 🛠️ Onboarding Quickstart

Welcome to the team! This repository represents our core **PromptOps** infrastructure. As a developer shipping changes to our customer support email classifier, you must never change a prompt "blind". Every prompt alteration must run through this evaluation harness.

### 1. Local Environment Setup

Ensure you have Python 3.11+ installed. Clone the repository and run the following commands:

```bash
# Initialize virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package with all development and evaluation dependencies
pip install -e .[all]

# Create local env configuration
cp .env.example .env
```

Open `.env` and populate your `GEMINI_API_KEY`. If you want live Slack alerts, populate the `SLACK_WEBHOOK_URL` too.

### 2. Verify Operational Integrity

Ensure all automated tests are passing:

```bash
python -m pytest tests/ -v
```

### 3. Run a Local Prompt Evaluation

To evaluate the current prompt configuration against our golden test dataset:

```bash
# Run baseline evaluation on the mini rate-limit safe dataset
python scripts/run_eval.py --prompt v1.0.0.yaml --dataset golden_dataset_mini_v1.0.0.json --concurrency 1 --skip-judge
```

---

## 🏗️ Architectural Foundations & Rationale

Our design emphasizes strict type safety, zero external database overhead, and highly optimized network concurrency.

```
                  +--------------------------------+
                  |  Versioned Prompt (YAML v1.1)  |
                  +--------------------------------+
                                  |
                                  v
+------------------+     +------------------+     +------------------------+
|  Golden Dataset  | --> |   Async Runner   | --> |   LLM-as-a-Judge       |
|  (Versioned JSON)|     |   (Semaphore)    |     |   (gemini-flash-lite)  |
+------------------+     +------------------+     +------------------------+
                                  |
                                  v
                       +----------------------+
                       |  Dual Storage Layer  |
                       |  (SQLite + JSON Run) |
                       +----------------------+
                                  |
            +---------------------+---------------------+
            |                                           |
            v                                           v
+-----------------------+                   +-----------------------+
|  Slack Block Kit Alert|                   |  GitHub Actions PR    |
|  (Color-coded verdict)|                   |  Markdown Comments    |
+-----------------------+                   +-----------------------+
```

### 1. Prompts-as-Code (YAML)
We treat prompts like source code, not arbitrary database configurations. By versioning prompts as `.yaml` files in the `/prompts` folder, they are integrated into standard development cycles. Prompt edits undergo code review, require pull requests, and automatically trigger CI runs.

### 2. Dual-Storage Persistence (SQLite + JSON)
We reject heavy database infrastructure requirements. 
- **SQLite (`data/results.db`)**: Enforces relational database schemas (`eval_runs` and `case_results`) to allow immediate, lightweight SQL query capability for baseline runs and historical delta calculations.
- **Serialized JSON (`data/results/run_<run_id>.json`)**: Every run is compiled into a highly portable, human-readable, git-trackable JSON file. This decouples individual run details from physical servers.

### 3. Concurrency Orchestration (asyncio Semaphore)
Running 92+ test cases against remote LLM API endpoints sequentially is unacceptably slow. However, unrestricted concurrency leads to `429 Rate Limit` exhaustion. We use an `asyncio.Semaphore` (capped via `EvalConfig.max_concurrency`) to run tests in parallel while strictly adhering to rate limits, utilizing custom exponential backoff wrappers (`_retry_async`) for transient error recovery.

### 4. Semantic LLM-as-a-Judge
Exact text diffs (like BLEU or ROUGE) are poor indicators of summary quality. A rewritten sentence can be 100% semantically correct but return a 0% exact-match score. We employ a fast, low-cost model (`gemini-2.0-flash-lite`) as a judge, grading the predicted summary relative to the golden summary on a strict 1.0–5.0 semantic rubric.

### 5. Regression Flips vs. Slow Drift
Our engine divides quality tracking into two separate paradigms:
- **Per-Run Regressions ("Flips")**: Individual test cases that went from `PASS` to `FAIL` in a single run. These represent immediate, localized feature breakage (e.g., breaking a few-shot demonstration).
- **Average Metric Degradation ("Slow Drift")**: Incremental, statistical declines in overall accuracy or latency across multiple runs (often caused by subtle base model updates or prompt clutter). We track these via our comparative delta logic to flag builds before degradation hits users.

---

## 🗃️ golden Dataset & Curation

Our Golden Dataset resides at [data/golden_dataset_v1.0.0.json](file:///C:/Users/ericz/.gemini/antigravity/scratch/model-regression-detector/data/golden_dataset_v1.0.0.json). It is validated through strict Pydantic schemas defined in `src/models.py`.

### 1. Adding a New Test Case

To add a new customer support email to the golden suite, append it to the `cases` list in the JSON dataset:

```json
{
  "id": "TC-093",
  "input_email": "Hello, I cannot access the API monitoring dashboard since we rotated our IAM credentials. I keep receiving a 403 Forbidden error.",
  "expected_category": "account",
  "expected_summary": "Customer reports access issues to the API monitoring dashboard after rotating IAM credentials.",
  "expected_difficulty": "medium",
  "tags": ["IAM-service", "credentials-rotation", "production"],
  "notes": "Asserts role permission boundary — discussions of credential rotation should categorize as account, not technical."
}
```

*Every test case MUST include a `notes` field detailing why it exists and what edge case or regression risk it protects against.*

---

## 🎛️ Calibrating Quality Thresholds

Thresholds are defined inside `src/models.py` in the `EvalConfig` model and can be overridden via arguments on the CLI:

```python
class EvalConfig(BaseModel):
    warning_threshold_pct: float = Field(default=3.0, description="Flag warning if accuracy drops by this %")
    critical_threshold_pct: float = Field(default=8.0, description="Flag critical if accuracy drops by this %")
    max_concurrency: int = Field(default=5, description="Max concurrent LLM calls")
```

### Deciphering Build Exit Codes
The pipeline enforces standard UNIX exit codes:
- **`0`**: Evaluation successful and verdict is `pass` or `warning`. Build succeeds, PR comments post, Slack alerts deliver.
- **`1`**: Evaluation detected a **`critical` quality failure** (overall accuracy dropped by $\ge$ critical threshold, or unhandled runner exception). Build fails, merge is blocked.
- **`2`**: Invalid CLI usage or prompt configuration validation failure (invalid YAML structure). Build fails.

---

## 🐳 Containerization (Docker)

We containerize the pipeline to ensure identical, stateless execution across developer laptops, local git hooks, and cloud runner environments.

### 1. Build the Docker Image
```bash
docker build -t model-regression-detector .
```

### 2. Execute Containerized Evaluation
You can run evaluations by passing your environmental API key dynamically to the container:

```bash
docker run --env GEMINI_API_KEY="AIzaSy..." model-regression-detector --prompt v1.0.0.yaml --dataset golden_dataset_mini_v1.0.0.json --concurrency 1 --skip-judge
```
