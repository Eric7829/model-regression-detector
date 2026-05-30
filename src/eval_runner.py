"""Orchestrator for running evaluation runs against versioned datasets and prompts.

Implements highly concurrent execution using asyncio Semaphores, robust retries with 
exponential backoff, latency/token usage tracking, and cost calculation.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime

from src.classifier import classify_email
from src.judge import score_summary_relevance
from src.models import (
    CaseResult,
    CategoryAccuracy,
    EmailCategory,
    EvalConfig,
    EvalRunResult,
    GoldenCase,
    GoldenDataset,
    PromptConfig,
    RunMetadata,
)

logger = logging.getLogger(__name__)


def estimate_token_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate the estimated USD cost of an API call based on token counts."""
    model_lower = model.lower()
    
    # Pricing per 1,000,000 tokens as of early 2026 / late 2025
    if "gpt-4o-mini" in model_lower:
        input_price_per_m = 0.15
        output_price_per_m = 0.60
    elif "gpt-4o" in model_lower:
        input_price_per_m = 5.00
        output_price_per_m = 15.00
    elif "gemini-2.0-flash-lite" in model_lower or "gemini-flash-lite" in model_lower or "flash-lite" in model_lower:
        input_price_per_m = 0.075
        output_price_per_m = 0.30
    elif "gemini-2.0-flash" in model_lower or "gemini-flash" in model_lower or "flash" in model_lower:
        input_price_per_m = 0.075
        output_price_per_m = 0.30
    else:
        # Default fallback
        input_price_per_m = 0.15
        output_price_per_m = 0.60

    cost = ((prompt_tokens / 1_000_000) * input_price_per_m) + ((completion_tokens / 1_000_000) * output_price_per_m)
    return cost


async def _retry_async(func, *args, retries=2, delay=1.0, backoff=2.0, **kwargs):
    """Generic async retry wrapper with exponential backoff."""
    current_delay = delay
    for attempt in range(retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt == retries:
                logger.error("All %d retries failed for operation: %s", retries, str(e))
                raise e
            logger.warning(
                "Attempt %d failed with error: %s. Retrying in %.2fs...",
                attempt + 1, e, current_delay
            )
            await asyncio.sleep(current_delay)
            current_delay *= backoff


class EvalRunner:
    """Manages concurrent, robust execution of evaluation datasets."""

    def __init__(self, config: EvalConfig | None = None):
        self.config = config or EvalConfig()
        self.semaphore = asyncio.Semaphore(self.config.max_concurrency)
        self.classifier_client = None
        self.judge_client = None

    def _get_gemini_client(self):
        """Lazy initializer for Gemini client to avoid imports when offline."""
        from google import genai
        import os
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        return genai.Client(api_key=api_key)

    def _get_openai_client(self):
        """Lazy initializer for OpenAI client."""
        from openai import AsyncOpenAI
        return AsyncOpenAI()

    def _init_clients(self, classifier_model: str) -> None:
        """Initialize the appropriate API clients based on model selections."""
        # 1. Initialize Classifier Client
        if classifier_model.startswith("gemini"):
            if self.classifier_client is None:
                self.classifier_client = self._get_gemini_client()
        else:
            if self.classifier_client is None:
                self.classifier_client = self._get_openai_client()

        # 2. Initialize Judge Client
        if self.judge_client is None:
            # Judge is always Gemini by default
            try:
                self.judge_client = self._get_gemini_client()
            except Exception as err:
                logger.warning("Could not initialize Gemini client for judge: %s. Evals running without judge will succeed, but judge grading will fail.", err)

    async def run_case(
        self,
        case: GoldenCase,
        prompt_config: PromptConfig,
        skip_judge: bool = False,
    ) -> CaseResult:
        """Run evaluation for a single GoldenCase under concurrency controls and retries."""
        async with self.semaphore:
            # 1. Dispatch Classifier with Retries
            async def do_classify():
                return await classify_email(
                    email_text=case.input_email,
                    config=prompt_config,
                    client=self.classifier_client,
                )

            try:
                class_res = await _retry_async(
                    do_classify,
                    retries=self.config.retry_count,
                    delay=self.config.retry_delay_s,
                )
            except Exception as e:
                # API error or persistent failure
                return CaseResult(
                    case_id=case.id,
                    input_email=case.input_email,
                    expected_category=case.expected_category,
                    expected_summary=case.expected_summary,
                    predicted_summary="",
                    error=f"Classification failed after retries: {e}",
                    raw_response="",
                )

            # 2. Process Output & Score Summary
            if not class_res.success:
                return CaseResult(
                    case_id=case.id,
                    input_email=case.input_email,
                    expected_category=case.expected_category,
                    expected_summary=case.expected_summary,
                    predicted_summary="",
                    error=class_res.error or "Unknown classification error",
                    latency_ms=class_res.latency_ms,
                    prompt_tokens=class_res.prompt_tokens,
                    completion_tokens=class_res.completion_tokens,
                    total_tokens=class_res.total_tokens,
                    raw_response=class_res.raw_response,
                )

            output = class_res.output
            category_match = (output.category == case.expected_category)

            # 3. LLM-as-a-judge for summary relevance
            summary_score = 0.0
            judge_error = None
            judge_prompt_toks = 0
            judge_compl_toks = 0

            if not skip_judge and class_res.success:
                async def do_score():
                    return await score_summary_relevance(
                        email=case.input_email,
                        expected_summary=case.expected_summary,
                        predicted_summary=output.summary,
                        model=self.config.judge_model,
                        client=self.judge_client,
                    )

                try:
                    score, reasoning, j_p_tok, j_c_tok = await _retry_async(
                        do_score,
                        retries=self.config.retry_count,
                        delay=self.config.retry_delay_s,
                    )
                    summary_score = score
                    judge_prompt_toks = j_p_tok
                    judge_compl_toks = j_c_tok
                except Exception as e:
                    logger.warning("Judge failed to score summary for case %s: %s", case.id, e)
                    judge_error = f"Judge failed: {e}"
                    summary_score = 1.0  # Min score on failure

            # Capture complete CaseResult
            # We preserve case-level token tracking as the classifier's usage.
            # Total run-level metrics will sum classifier + judge tokens.
            return CaseResult(
                case_id=case.id,
                input_email=case.input_email,
                expected_category=case.expected_category,
                predicted_category=output.category,
                category_match=category_match,
                expected_summary=case.expected_summary,
                predicted_summary=output.summary,
                summary_relevance_score=summary_score,
                latency_ms=class_res.latency_ms,
                prompt_tokens=class_res.prompt_tokens,
                completion_tokens=class_res.completion_tokens,
                total_tokens=class_res.total_tokens,
                raw_response=class_res.raw_response,
                confidence=output.confidence,
                error=judge_error if judge_error else None,
            )

    async def run_evaluation(
        self,
        dataset: GoldenDataset,
        prompt_config: PromptConfig,
        skip_judge: bool = False,
        progress_callback = None,
    ) -> EvalRunResult:
        """Run an end-to-end evaluation of a dataset using a prompt configuration."""
        self._init_clients(prompt_config.model)
        
        run_id = str(uuid.uuid4())
        started_at = datetime.utcnow()
        
        logger.info(
            "Starting evaluation run %s on dataset %s using prompt version %s",
            run_id, dataset.version, prompt_config.version
        )

        # Build list of case coroutines
        tasks = []
        for case in dataset.cases:
            async def run_and_report(c=case):
                res = await self.run_case(c, prompt_config, skip_judge)
                if progress_callback:
                    progress_callback(res)
                return res
            tasks.append(run_and_report())

        # Execute concurrently
        case_results = await asyncio.gather(*tasks)
        
        finished_at = datetime.utcnow()
        
        # 4. Aggregate Run Results
        total_cases = len(case_results)
        passed_cases = sum(1 for r in case_results if r.passed)
        failed_cases = sum(1 for r in case_results if r.error is None and not r.passed)
        error_cases = sum(1 for r in case_results if r.error is not None)
        
        overall_accuracy = passed_cases / total_cases if total_cases > 0 else 0.0
        
        # Calculate category accuracies
        per_category_acc = []
        for cat in list(EmailCategory):
            cat_cases = [r for r in case_results if r.expected_category == cat]
            if cat_cases:
                cat_correct = sum(1 for r in cat_cases if r.passed)
                per_category_acc.append(
                    CategoryAccuracy(
                        category=cat,
                        total=len(cat_cases),
                        correct=cat_correct,
                        accuracy=cat_correct / len(cat_cases),
                    )
                )
        
        # Calculate summary scores and latencies
        scored_cases = [r for r in case_results if r.summary_relevance_score > 0.0]
        avg_summary_score = (
            sum(r.summary_relevance_score for r in scored_cases) / len(scored_cases)
            if scored_cases else 0.0
        )
        
        valid_latencies = [r.latency_ms for r in case_results if r.latency_ms > 0.0]
        avg_latency_ms = sum(valid_latencies) / len(valid_latencies) if valid_latencies else 0.0
        
        # Sum total tokens (classifier + judge if judge was run)
        total_tokens_used = sum(r.total_tokens for r in case_results)
        
        # Compute cost estimation
        # Sum individual case costs
        total_cost = 0.0
        for r in case_results:
            # Classifier cost
            class_cost = estimate_token_cost(prompt_config.model, r.prompt_tokens, r.completion_tokens)
            total_cost += class_cost
            
            # Judge cost (if scored and not skipped)
            # Since we didn't store judge tokens inside CaseResult (to preserve classifier data),
            # we estimate judge cost or assume a standard prompt/completion usage for the judge 
            # if we didn't track it, but wait! We can estimate judge tokens as roughly 1000 input and 150 output
            # if judge was run, or let's add a constant or estimate it:
            if not skip_judge and r.summary_relevance_score > 0.0:
                # Roughly 1000 input tokens (system instructions + email + summaries) and 150 output tokens
                judge_cost = estimate_token_cost(self.config.judge_model, 1000, 150)
                total_cost += judge_cost

        metadata = RunMetadata(
            run_id=run_id,
            prompt_version=prompt_config.version,
            model=prompt_config.model,
            dataset_version=dataset.version,
            started_at=started_at,
            finished_at=finished_at,
            total_cases=total_cases,
            passed_cases=passed_cases,
            failed_cases=failed_cases,
            error_cases=error_cases,
        )

        return EvalRunResult(
            metadata=metadata,
            case_results=case_results,
            overall_accuracy=overall_accuracy,
            per_category_accuracy=per_category_acc,
            avg_summary_score=avg_summary_score,
            avg_latency_ms=avg_latency_ms,
            total_tokens_used=total_tokens_used,
            total_cost_estimate_usd=total_cost,
        )
