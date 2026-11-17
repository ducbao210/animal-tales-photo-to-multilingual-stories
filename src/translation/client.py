"""Groq client for structured story translations and reviews."""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from src.contracts.schema import StageError
from .rate_limit import GroqRateLimiter

LOGGER = logging.getLogger(__name__)


class GroqTranslationClient:
    def __init__(
        self, model_id: str, timeout_seconds: float = 60,
        reasoning_effort: str | None = None,
        *, requests_per_minute: int, tokens_per_minute: int,
    ) -> None:
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise StageError("D", "missing_credentials", False, "GROQ_API_KEY is required for translation")
        try:
            from groq import Groq
        except ImportError as error:
            raise StageError("D", "dependency_missing", False, "Install the groq dependency") from error
        self.client = Groq(api_key=key, timeout=timeout_seconds, max_retries=0)
        self.model_id = model_id
        self.reasoning_effort = reasoning_effort
        self.rate_limiter = GroqRateLimiter(requests_per_minute, tokens_per_minute)

    def generate(self, prompt: str, *, schema_name: str, schema: dict[str, Any], max_tokens: int, temperature: float = 0.2) -> dict[str, Any]:
        try:
            request = dict(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_schema", "json_schema": {"name": schema_name, "strict": True, "schema": schema}},
            )
            if self.reasoning_effort is not None:
                request["reasoning_effort"] = self.reasoning_effort
            try:
                self.rate_limiter.wait_for_request(prompt, max_tokens)
                response = self.client.chat.completions.create(**request)
            except Exception as error:
                if getattr(error, "status_code", None) != 400 or "json_validate_failed" not in str(error):
                    raise
                LOGGER.warning(
                    "Groq strict JSON generation failed for %s; retrying in JSON object mode",
                    schema_name,
                )
                self.rate_limiter.wait_for_request(prompt, max_tokens)
                response = self.client.chat.completions.create(
                    **{**request, "response_format": {"type": "json_object"}}
                )
            choice = response.choices[0]
            if getattr(choice.message, "refusal", None):
                raise StageError("D", "provider_refusal", False, "Groq refused the translation request")
            content = choice.message.content
            if not content:
                raise StageError("D", "empty_response", False, "Groq returned no translation content")
            try:
                data = json.loads(content)
            except json.JSONDecodeError as error:
                raise StageError("D", "malformed_json", False, "Groq returned malformed JSON") from error
            if not isinstance(data, dict):
                raise StageError("D", "invalid_response", False, "Groq response must be a JSON object")
            missing = [key for key in schema.get("required", []) if key not in data]
            if missing:
                raise StageError("D", "invalid_response", True, f"Groq response is missing required fields: {', '.join(missing)}")
            return {"data": data, "request_id": getattr(response, "_request_id", None)}
        except StageError:
            raise
        except Exception as error:
            status = getattr(error, "status_code", None)
            retryable = status in (408, 409, 425, 429) or (isinstance(status, int) and status >= 500)
            name = type(error).__name__.lower()
            retryable = retryable or "timeout" in name or "connection" in name
            invalid_generation = status == 400 and "json_validate_failed" in str(error)
            retryable = retryable or invalid_generation
            category = (
                "credentials" if status in (401, 403)
                else "rate_limit" if status == 429
                else "invalid_generation" if invalid_generation
                else "provider_transient" if retryable
                else "provider_error"
            )
            raise StageError("D", category, retryable, f"{type(error).__name__}: {error}") from error
