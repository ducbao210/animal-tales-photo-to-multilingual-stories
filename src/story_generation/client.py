"""Google Gemini client returning locally parsed strict JSON objects."""
from __future__ import annotations

import json
import os
from typing import Any

from src.contracts.schema import StageError


def _parse_json_object(content: str) -> Any:
    candidate = content.strip()
    if candidate.startswith("```"):
        candidate = candidate[3:].lstrip()
        if candidate[:4].casefold() == "json":
            candidate = candidate[4:].lstrip()
        if candidate.endswith("```"):
            candidate = candidate[:-3].rstrip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as original_error:
        # Be tolerant of a JSON object wrapped in prose or a Markdown fence.
        object_start = candidate.find("{")
        if object_start >= 0:
            try:
                parsed, _ = json.JSONDecoder().raw_decode(candidate, object_start)
                return parsed
            except json.JSONDecodeError:
                pass
        raise original_error


def _gemini_api_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove JSON Schema fields unsupported by the Gemini Developer API."""
    def strip_unsupported(value: Any) -> Any:
        if isinstance(value, dict):
            schema: dict[str, Any] = {}
            for key, item in value.items():
                if key in {"additionalProperties", "additional_properties"}:
                    continue
                if (
                    key == "type"
                    and isinstance(item, list)
                    and "null" in item
                    and len(item) == 2
                ):
                    schema["type"] = next(type_name for type_name in item if type_name != "null")
                    schema["nullable"] = True
                else:
                    schema[key] = strip_unsupported(item)
            return schema
        if isinstance(value, list):
            return [strip_unsupported(item) for item in value]
        return value

    return strip_unsupported(schema)


class GeminiStoryClient:
    def __init__(self, model_id: str, timeout_seconds: float = 60) -> None:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise StageError("C", "missing_credentials", False, "GEMINI_API_KEY is required for story generation")
        try:
            from google import genai
            from google.genai import types
        except ImportError as error:
            raise StageError("C", "dependency_missing", False, "Install the google-genai dependency") from error
        self.client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
        )
        self.model_id = model_id

    def generate(
        self, prompt: str, *, schema_name: str, schema: dict[str, Any],
        max_tokens: int, temperature: float = 0.4,
    ) -> dict[str, Any]:
        del schema_name  # Gemini takes the JSON schema directly rather than a named response format.
        try:
            from google.genai import types

            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json",
                    response_schema=_gemini_api_schema(schema),
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.LOW
                    ),
                ),
            )
            content = getattr(response, "text", None)
            if not content:
                raise StageError("C", "empty_response", False, "Gemini returned no story content")
            data = getattr(response, "parsed", None)
            if not isinstance(data, dict):
                candidates = getattr(response, "candidates", None) or []
                finish_reason = (
                    getattr(candidates[0], "finish_reason", None) if candidates else None
                )
                try:
                    data = _parse_json_object(content)
                except json.JSONDecodeError as error:
                    raise StageError(
                        "C",
                        "malformed_json",
                        True,
                        "Gemini returned malformed JSON at "
                        f"line {error.lineno}, column {error.colno} "
                        f"(response_chars={len(content)}, finish_reason={finish_reason})",
                    ) from error
            if not isinstance(data, dict):
                raise StageError("C", "invalid_response", False, "Gemini response must be a JSON object")
            return {"data": data, "request_id": getattr(response, "response_id", None)}
        except StageError:
            raise
        except Exception as error:
            status = getattr(error, "status_code", None) or getattr(error, "code", None)
            retryable = status in (408, 409, 425, 429) or (isinstance(status, int) and status >= 500)
            name = type(error).__name__.lower()
            retryable = retryable or "timeout" in name or "connection" in name
            category = "credentials" if status in (401, 403) else "rate_limit" if status == 429 else "provider_transient" if retryable else "provider_error"
            raise StageError("C", category, retryable, f"{type(error).__name__}: {error}") from error
