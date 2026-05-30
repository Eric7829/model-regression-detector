"""LLM-as-a-judge summary relevance scorer.

Uses a fast, cost-effective Gemini model (e.g. ``gemini-2.0-flash-lite`` or ``gemini-flash-lite-latest``) 
to evaluate predicted summaries against golden summaries on a 1-5 scale.
"""

from __future__ import annotations

import json
import logging
import os
from google.genai import types

logger = logging.getLogger(__name__)


def _get_gemini_client():
    """Create a Gemini client using the API key from environment."""
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    return genai.Client(api_key=api_key)


_JUDGE_SYSTEM_PROMPT = """You are an expert AI evaluator. Your job is to grade the quality and semantic relevance of a model-generated summary of a customer support email compared to a pre-defined Ground Truth (Golden) Summary.

You will receive:
1. The raw Customer Email.
2. The expected Ground Truth Summary.
3. The model-generated Predicted Summary.

You must assign a score from 1.0 to 5.0 using the following strict rubric:

- 5.0 (Excellent): The predicted summary perfectly captures the primary customer issue, matches the ground truth's semantic intent, is highly concise, and adds no hallucinated/unhelpful info.
- 4.0 (Good): The predicted summary captures all main issues but is slightly wordy, uses poor phrasing, or omits a very minor detail.
- 3.0 (Fair): The predicted summary captures some main issues but misses important context, has minor inaccuracies, or misses the core problem while getting peripheral issues right.
- 2.0 (Poor): The predicted summary has significant omissions, is misleading, is highly repetitive, or is poorly structured.
- 1.0 (Unacceptable): The predicted summary is totally incorrect, irrelevant, empty, or describes an entirely different issue.

You MUST respond with a JSON object containing two fields:
{
  "reasoning": "A brief explanation of why this score was assigned, pointing to the specific strengths/weaknesses of the predicted summary based on the rubric.",
  "score": 4.0
}

Assign float values (e.g., 4.0, 3.5, 5.0, 1.0). Be fair, objective, and consistent. Do not grade the email, only grade the predicted summary's representation of the email relative to the ground truth summary.
"""


def _build_judge_prompt(email: str, expected_summary: str, predicted_summary: str) -> str:
    """Build the user prompt for the judge."""
    return f"""### CUSTOMER EMAIL:
{email}

### GROUND TRUTH (EXPECTED) SUMMARY:
{expected_summary}

### PREDICTED SUMMARY UNDER TEST:
{predicted_summary}

Assign a score and provide reasoning based on the criteria in your system prompt. Ensure your output is a single JSON object.
"""


async def score_summary_relevance(
    email: str,
    expected_summary: str,
    predicted_summary: str,
    model: str = "gemini-2.0-flash-lite",
    client=None,
) -> tuple[float, str, int, int]:
    """Score the relevance of a predicted summary against a ground truth summary using Gemini.

    Returns
    -------
    tuple[float, str, int, int]
        A tuple of (score, reasoning, prompt_tokens, completion_tokens). 
        If a failure occurs, returns (1.0, error_message, 0, 0).
    """
    if not predicted_summary or not predicted_summary.strip():
        return 1.0, "Predicted summary is empty", 0, 0

    if client is None:
        try:
            client = _get_gemini_client()
        except ValueError as err:
            logger.error("Failed to initialize Gemini client for judge: %s", err)
            return 1.0, f"Client initialization error: {err}", 0, 0

    user_prompt = _build_judge_prompt(email, expected_summary, predicted_summary)

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=_JUDGE_SYSTEM_PROMPT,
                temperature=0.0,
                response_mime_type="application/json",
                max_output_tokens=512,
            ),
        )

        raw_text = response.text or ""
        cleaned = raw_text.strip()
        
        usage = response.usage_metadata
        p_tokens = usage.prompt_token_count if usage else 0
        c_tokens = usage.candidates_token_count if usage else 0
        
        # Handle simple markdown code block encapsulation if it slips through
        if cleaned.startswith("```"):
            first_newline = cleaned.index("\n")
            cleaned = cleaned[first_newline + 1 :]
            if cleaned.endswith("```"):
                cleaned = cleaned[: -len("```")].rstrip()

        try:
            data = json.loads(cleaned)
            score = float(data.get("score", 1.0))
            reasoning = str(data.get("reasoning", "No reasoning provided by judge"))
            
            # Bound the score
            score = max(1.0, min(5.0, score))
            return score, reasoning, p_tokens, c_tokens
            
        except (json.JSONDecodeError, ValueError) as err:
            logger.warning("Failed to parse judge JSON: %s. Raw response: %s", err, raw_text)
            return 1.0, f"Judge parse error: {err}. Raw response: {raw_text[:150]}", p_tokens, c_tokens

    except Exception as api_err:
        logger.error("Judge API call failed: %s", api_err)
        return 1.0, f"Judge API error: {api_err}", 0, 0
