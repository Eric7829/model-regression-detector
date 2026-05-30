"""Multi-dimensional scoring for eval results.

Scores each test case on:
1. Category match (binary)
2. Summary relevance (LLM-as-judge, 1-5)
3. Latency
4. Token usage

The LLM-as-judge uses a cheap model (gemini-flash-lite) to rate summary
quality, avoiding circular dependency on the model being tested.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time

from src.models import CaseResult, EmailCategory, EvalConfig

logger = logging.getLogger(__name__)


JUDGE_PROMPT = """You are evaluating the quality of an email classification summary.

Given:
- The original customer email
- The reference summary (ground truth)
- The generated summary (to evaluate)

Rate the generated summary on a scale of 1 to 5:
1 = Completely irrelevant or wrong
2 = Captures some elements but misses the core issue
3 = Captures the main issue but lacks important details
4 = Good summary that captures the key issue with sufficient detail
5 = Excellent summary that matches or exceeds the reference quality

Respond with JSON only: {"score": <int 1-5>, "reasoning": "<brief explanation>"}
"""


async def score_summary_relevance(
    email_text: str,
    reference_summary: str,
    generated_summary: str,
    config: EvalConfig | None = None,
) -> tuple[float, str]:
    """Use LLM-as-judge to score summary relevance.

    Returns (score, reasoning) tuple.
    """
    from google import genai
    from google.genai import types

    if not generated_summary or not generated_summary.strip():
        return 0.0, "No summary generated"

    cfg = config or EvalConfig()
    api_key = os.environ.get("GEMINI_API_KEY", "")
    client = genai.Client(api_key=api_key)

    user_content = (
        f"ORIGINAL EMAIL:\n{email_text}\n\n"
        f"REFERENCE SUMMARY:\n{reference_summary}\n\n"
        f"GENERATED SUMMARY:\n{generated_summary}"
    )

    try:
        response = await client.aio.models.generate_content(
            model=cfg.judge_model,
            contents=[types.Content(role="user", parts=[types.Part(text=user_content)])],
            config=types.GenerateContentConfig(
                system_instruction=JUDGE_PROMPT,
                temperature=0.0,
                response_mime_type="application/json",
                max_output_tokens=150,
            ),
        )

        raw = response.text or ""
        data = json.loads(raw.strip())
        score = float(data.get("score", 0))
        reasoning = data.get("reasoning", "")
        return min(max(score, 0.0), 5.0), reasoning

    except Exception as e:
        logger.warning("LLM-as-judge scoring failed: %s", e)
        return 0.0, f"Scoring error: {e}"


async def score_case(
    case_result: CaseResult,
    config: EvalConfig | None = None,
    skip_judge: bool = False,
) -> CaseResult:
    """Score a single case result on all dimensions.

    Mutates and returns the CaseResult with updated scores.
    """
    # Dimension 1: Category match (already set by eval runner)
    # Dimension 2: Summary relevance via LLM-as-judge
    if not skip_judge and case_result.predicted_summary:
        score, _reasoning = await score_summary_relevance(
            case_result.input_email,
            case_result.expected_summary,
            case_result.predicted_summary,
            config,
        )
        case_result = case_result.model_copy(update={"summary_relevance_score": score})

    # Dimensions 3 & 4 (latency, tokens) are already captured during classification
    return case_result


async def score_all_cases(
    case_results: list[CaseResult],
    config: EvalConfig | None = None,
    skip_judge: bool = False,
) -> list[CaseResult]:
    """Score all case results, with concurrency limiting for the LLM judge."""
    cfg = config or EvalConfig()
    semaphore = asyncio.Semaphore(cfg.max_concurrency)

    async def _score_with_limit(cr: CaseResult) -> CaseResult:
        async with semaphore:
            return await score_case(cr, cfg, skip_judge)

    return await asyncio.gather(*[_score_with_limit(cr) for cr in case_results])
