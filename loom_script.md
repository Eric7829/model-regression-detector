# Loom Walkthrough Script: Model Regression Detection System

*A professional, 3-minute script and recording flow guide designed to showcase engineering maturity to hiring managers.*

---

## 🎥 Recording Setup Tips

- **Screen Resolution**: Standard 1080p.
- **Layout**: Split screen. VS Code on the left, Terminal (or GitHub) on the right. Have a browser tab open with the Slack block payload or GitHub repository.
- **Tone**: Energetic, articulate, engineering-focused. Avoid jargon-heavy buzzwords; focus on system architecture and product stability.

---

## ⏱️ Video Timeline

| Duration | Section | Focus |
| :--- | :--- | :--- |
| **0:00 - 0:30** | **Introduction** | State the core problem (blind prompt deployment) and present the solution. |
| **0:30 - 1:15** | **Architecture Walkthrough** | Show prompts-as-code YAMLs, Pydantic schemas, and explain the SQLite/JSON dual storage. |
| **1:15 - 2:15** | **Live Execution Demo** | Run the CLI, trigger an evaluation, show delta metric comparison, and explain "Flips". |
| **2:15 - 2:45** | **CI/CD & Alerting Gates** | Show the GitHub Actions workflow and show the beautiful Slack block JSON layout. |
| **2:45 - 3:00** | **Conclusion** | Summarize the PromptOps philosophy and wrap up. |

---

## 🎙️ Transcript Script

### 1. Introduction (0:00 - 0:30)

**[Action: Video bubble showing you, sharing your split screen of VS Code.]**

> *"Hi everyone, I’m [Your Name]. Today I’m excited to walk you through my **Model Regression Detection System**.*
> 
> *Almost every AI engineering team has a major blind spot: they tweak a system prompt, verify it against three cases, and push it to production—only to silently break 15% of their core traffic. I built this automated PromptOps pipeline to bring continuous integration (CI/CD) and statistical regression testing to LLM-powered features. Let's look at how it works."*

---

### 2. Architecture & Prompts-as-Code (0:30 - 1:15)

**[Action: Click on `prompts/v1.1.0.yaml` in VS Code.]**

> *"We treat prompts exactly like source code. Here, our system prompt and few-shot examples are stored in git-versioned YAML files, rather than hidden in database tables. If someone edits a prompt, they must go through a code review and trigger an automated evaluation run.*
> 
> *Our pipeline is strictly typed. All prompt configs, golden datasets, and model outputs are validated through rigid Pydantic models. A schema mismatch fails fast during load time, rather than producing garbage outputs in production."*

---

### 3. Live Execution Demo (1:15 - 2:15)

**[Action: Switch focus to the terminal. Type or show the command: `python scripts/run_eval.py --prompt v1.0.0.yaml --dataset golden_dataset_mini_v1.0.0.json --concurrency 1 --skip-judge` and hit enter. Let it run and display the scorecard.]**

> *"Let’s run the evaluation. Our runner executing against our versioned golden dataset runs parallel async calls under strict semaphore controls to respect rate limits, utilizing custom exponential backoff wrappers.*
> 
> *When the run finishes, it automatically queries a local SQLite database to fetch the latest baseline run, computes accuracy and latency deltas, and outputs this rich CLI report.*
> 
> *One architectural detail I’m proud of is how we separate **Slow Drift**—like a statistical 2% drop in overall accuracy—from **Behavioral Flips**, where a specific high-priority case went from passing to failing. This makes it incredibly easy for developers to pinpoint exactly *what* changed."*

---

### 4. CI/CD & Slack Alerting Gates (2:15 - 2:45)

**[Action: Click on `.github/workflows/eval.yml` in VS Code, then show the printed Mock Slack Block Kit JSON payload in the terminal.]**

> *"To protect our production environment, we wire this directly into CI. Our GitHub Actions workflow triggers on every pull request that modifies prompts. If a critical accuracy regression is detected, the run exits with code `1`, blocking the git merge.*
> 
> *We also integrate visual Slack alerting. Using zero-dependency Python scripts to maintain a lean CI container, the pipeline compiles and delivers this visually premium Block Kit alert directly to Slack—color-coded by verdict so our team gets real-time, actionable insights."*

---

### 5. Conclusion (2:45 - 3:00)

**[Action: Face the camera, smiling.]**

> *"Building this system proved to me that prompt engineering is software engineering. By putting robust regression barriers around LLMs, we can deploy prompt changes with absolute confidence. 
> 
> *Thanks for watching, and feel free to check out the repository's onboarding README.md for full architectural rationales!"*
