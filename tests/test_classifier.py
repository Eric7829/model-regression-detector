"""Tests for the classifier module.

These tests mock the OpenAI client so they run offline and free.
They verify the message assembly, response parsing, and error handling
— everything except the actual LLM quality (that's what the eval pipeline is for).
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.classifier import (
    ClassificationResult,
    _build_messages,
    _parse_response,
    classify_email,
)
from src.models import ClassifierOutput, EmailCategory, FewShotExample, PromptConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def basic_config() -> PromptConfig:
    return PromptConfig(
        version="v1.0.0",
        system_prompt="You are an email classifier.",
    )


@pytest.fixture
def config_with_examples() -> PromptConfig:
    return PromptConfig(
        version="v1.1.0",
        system_prompt="You are an email classifier.",
        few_shot_examples=[
            FewShotExample(
                email="I was charged twice",
                category="billing",
                summary="Customer reports a double charge.",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------


class TestBuildMessages:
    def test_basic_messages(self, basic_config):
        messages = _build_messages("Help me", basic_config)
        assert len(messages) == 2  # system + user
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Help me"

    def test_with_few_shot(self, config_with_examples):
        messages = _build_messages("Help me", config_with_examples)
        # system + (user + assistant) * 1 example + user
        assert len(messages) == 4
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["content"] == "Help me"

    def test_few_shot_content_is_json(self, config_with_examples):
        messages = _build_messages("Help me", config_with_examples)
        assistant_msg = json.loads(messages[2]["content"])
        assert assistant_msg["category"] == "billing"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_valid_json(self):
        raw = json.dumps(
            {
                "category": "technical",
                "summary": "Customer reports a bug in the export feature.",
                "confidence": 0.95,
            }
        )
        result = _parse_response(raw)
        assert result.category == EmailCategory.TECHNICAL
        assert result.confidence == 0.95

    def test_json_with_code_fences(self):
        raw = '```json\n{"category": "billing", "summary": "Customer was double-charged for their subscription plan.", "confidence": 0.9}\n```'
        result = _parse_response(raw)
        assert result.category == EmailCategory.BILLING

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_response("not json at all")

    def test_invalid_category_raises(self):
        raw = json.dumps(
            {"category": "invalid", "summary": "Some summary text for validation purposes."}
        )
        with pytest.raises(Exception):  # Pydantic ValidationError
            _parse_response(raw)


# ---------------------------------------------------------------------------
# Full classify_email (mocked)
# ---------------------------------------------------------------------------


class TestClassifyEmail:
    @pytest.mark.asyncio
    async def test_successful_classification(self, basic_config):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "category": "account",
                            "summary": "Customer needs to reset their password after being locked out.",
                            "confidence": 0.88,
                        }
                    )
                )
            )
        ]
        mock_response.usage = MagicMock(
            prompt_tokens=100, completion_tokens=30, total_tokens=130
        )

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await classify_email("I can't log in", basic_config, client=mock_client)

        assert result.success
        assert result.output.category == EmailCategory.ACCOUNT
        assert result.prompt_tokens == 100
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_parse_failure(self, basic_config):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="This is not JSON"))
        ]
        mock_response.usage = MagicMock(
            prompt_tokens=50, completion_tokens=10, total_tokens=60
        )

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await classify_email("Test email", basic_config, client=mock_client)

        assert not result.success
        assert result.output is None
        assert "Parse error" in result.error

    @pytest.mark.asyncio
    async def test_api_error(self, basic_config):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("Rate limit exceeded")
        )

        result = await classify_email("Test email", basic_config, client=mock_client)

        assert not result.success
        assert "API error" in result.error
        assert result.raw_response == ""
