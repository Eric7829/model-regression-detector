"""Customer support email classifier — the LLM feature under test.

This module wraps a single LLM call behind a clean function signature.
The prompt is *not* hardcoded — it comes from a PromptConfig loaded from
the versioned YAML files. This decoupling is the entire point: when someone
changes a prompt, CI re-evaluates automatically.

Supports both OpenAI and Gemini backends — dispatches based on model name.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass

from src.models import ClassifierOutput, PromptConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassificationResult:
    """Wraps the parsed output with execution metadata for eval scoring."""

    output: ClassifierOutput | None
    raw_response: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.output is not None and self.error is None


def _build_messages(email_text: str, config: PromptConfig) -> list[dict]:
    """Assemble the chat messages from the prompt config."""
    messages: list[dict] = [{"role": "system", "content": config.system_prompt}]

    for ex in config.few_shot_examples:
        messages.append({"role": "user", "content": ex.email})
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "category": ex.category,
                        "summary": ex.summary,
                        "confidence": 1.0,
                    }
                ),
            }
        )

    messages.append({"role": "user", "content": email_text})
    return messages


def _parse_response(raw: str) -> ClassifierOutput:
    """Parse the LLM's raw text response into a validated ClassifierOutput."""
    cleaned = raw.strip()

    if cleaned.startswith("```"):
        first_newline = cleaned.index("\n")
        cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[: -len("```")].rstrip()

    data = json.loads(cleaned)
    return ClassifierOutput(**data)


# ---------------------------------------------------------------------------
# Gemini backend (primary)
# ---------------------------------------------------------------------------


def _get_gemini_client():
    """Create a Gemini client using the API key from environment."""
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    return genai.Client(api_key=api_key)


def _build_gemini_contents(email_text: str, config: PromptConfig) -> list:
    """Build Gemini content parts from the prompt config."""
    from google.genai import types

    contents = []
    for ex in config.few_shot_examples:
        contents.append(types.Content(role="user", parts=[types.Part(text=ex.email)]))
        contents.append(
            types.Content(
                role="model",
                parts=[
                    types.Part(
                        text=json.dumps(
                            {"category": ex.category, "summary": ex.summary, "confidence": 1.0}
                        )
                    )
                ],
            )
        )

    contents.append(types.Content(role="user", parts=[types.Part(text=email_text)]))
    return contents


async def classify_email_gemini(
    email_text: str,
    config: PromptConfig,
    client=None,
) -> ClassificationResult:
    """Classify a customer email using Gemini API (async)."""
    from google.genai import types

    if client is None:
        client = _get_gemini_client()

    contents = _build_gemini_contents(email_text, config)
    start = time.perf_counter()

    try:
        response = await client.aio.models.generate_content(
            model=config.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=config.system_prompt,
                temperature=config.temperature,
                response_mime_type="application/json",
                max_output_tokens=256,
            ),
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        raw_text = response.text or ""
        usage = response.usage_metadata

        prompt_toks = usage.prompt_token_count if usage else 0
        completion_toks = usage.candidates_token_count if usage else 0
        total_toks = usage.total_token_count if usage else 0

        try:
            parsed = _parse_response(raw_text)
            return ClassificationResult(
                output=parsed,
                raw_response=raw_text,
                latency_ms=elapsed_ms,
                prompt_tokens=prompt_toks,
                completion_tokens=completion_toks,
                total_tokens=total_toks,
            )
        except (json.JSONDecodeError, Exception) as parse_err:
            logger.warning("Failed to parse Gemini response: %s", parse_err)
            return ClassificationResult(
                output=None,
                raw_response=raw_text,
                latency_ms=elapsed_ms,
                prompt_tokens=prompt_toks,
                completion_tokens=completion_toks,
                total_tokens=total_toks,
                error=f"Parse error: {parse_err}",
            )

    except Exception as api_err:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.error("Gemini API error: %s", api_err)
        return ClassificationResult(
            output=None,
            raw_response="",
            latency_ms=elapsed_ms,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            error=f"API error: {api_err}",
        )


def classify_email_gemini_sync(
    email_text: str,
    config: PromptConfig,
    client=None,
) -> ClassificationResult:
    """Synchronous Gemini classification."""
    from google.genai import types

    if client is None:
        client = _get_gemini_client()

    contents = _build_gemini_contents(email_text, config)
    start = time.perf_counter()

    try:
        response = client.models.generate_content(
            model=config.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=config.system_prompt,
                temperature=config.temperature,
                response_mime_type="application/json",
                max_output_tokens=256,
            ),
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        raw_text = response.text or ""
        usage = response.usage_metadata

        prompt_toks = usage.prompt_token_count if usage else 0
        completion_toks = usage.candidates_token_count if usage else 0
        total_toks = usage.total_token_count if usage else 0

        try:
            parsed = _parse_response(raw_text)
            return ClassificationResult(
                output=parsed,
                raw_response=raw_text,
                latency_ms=elapsed_ms,
                prompt_tokens=prompt_toks,
                completion_tokens=completion_toks,
                total_tokens=total_toks,
            )
        except (json.JSONDecodeError, Exception) as parse_err:
            return ClassificationResult(
                output=None,
                raw_response=raw_text,
                latency_ms=elapsed_ms,
                prompt_tokens=prompt_toks,
                completion_tokens=completion_toks,
                total_tokens=total_toks,
                error=f"Parse error: {parse_err}",
            )

    except Exception as api_err:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return ClassificationResult(
            output=None,
            raw_response="",
            latency_ms=elapsed_ms,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            error=f"API error: {api_err}",
        )


# ---------------------------------------------------------------------------
# OpenAI backend (kept for compatibility)
# ---------------------------------------------------------------------------


async def classify_email_openai(
    email_text: str,
    config: PromptConfig,
    client=None,
) -> ClassificationResult:
    """Classify a customer email using OpenAI API (async)."""
    from openai import AsyncOpenAI

    if client is None:
        client = AsyncOpenAI()

    messages = _build_messages(email_text, config)
    start = time.perf_counter()

    try:
        response = await client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            response_format={"type": "json_object"},
            max_tokens=256,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        raw_text = response.choices[0].message.content or ""
        usage = response.usage

        try:
            parsed = _parse_response(raw_text)
            return ClassificationResult(
                output=parsed,
                raw_response=raw_text,
                latency_ms=elapsed_ms,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            )
        except (json.JSONDecodeError, Exception) as parse_err:
            return ClassificationResult(
                output=None,
                raw_response=raw_text,
                latency_ms=elapsed_ms,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
                error=f"Parse error: {parse_err}",
            )

    except Exception as api_err:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return ClassificationResult(
            output=None,
            raw_response="",
            latency_ms=elapsed_ms,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            error=f"API error: {api_err}",
        )


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------


async def classify_email(
    email_text: str,
    config: PromptConfig,
    client=None,
) -> ClassificationResult:
    """Classify an email, dispatching to the right backend based on model name."""
    if config.model.startswith("gemini"):
        return await classify_email_gemini(email_text, config, client)
    else:
        return await classify_email_openai(email_text, config, client)


def classify_email_sync(
    email_text: str,
    config: PromptConfig,
    client=None,
) -> ClassificationResult:
    """Synchronous dispatcher."""
    if config.model.startswith("gemini"):
        return classify_email_gemini_sync(email_text, config, client)
    else:
        from openai import OpenAI
        if client is None:
            client = OpenAI()
        messages = _build_messages(email_text, config)
        start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                response_format={"type": "json_object"},
                max_tokens=256,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            raw_text = response.choices[0].message.content or ""
            usage = response.usage
            try:
                parsed = _parse_response(raw_text)
                return ClassificationResult(
                    output=parsed, raw_response=raw_text, latency_ms=elapsed_ms,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                )
            except Exception as parse_err:
                return ClassificationResult(
                    output=None, raw_response=raw_text, latency_ms=elapsed_ms,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                    error=f"Parse error: {parse_err}",
                )
        except Exception as api_err:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return ClassificationResult(
                output=None, raw_response="", latency_ms=elapsed_ms,
                prompt_tokens=0, completion_tokens=0, total_tokens=0,
                error=f"API error: {api_err}",
            )
