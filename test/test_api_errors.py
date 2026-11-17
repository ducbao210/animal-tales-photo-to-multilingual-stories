from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.contracts.schema import StageError
from src.translation.client import GroqTranslationClient
from src.utils.retry import is_transient_provider_error, provider_retry_after_seconds, retry_provider_call


class HttpError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class TestProviderErrors(unittest.TestCase):
    def test_groq_json_generation_error_is_retryable(self) -> None:
        class JsonValidationError(RuntimeError):
            status_code = 400

            def __init__(self) -> None:
                super().__init__("{'code': 'json_validate_failed'}")

        def fail_generation(**_kwargs: object) -> None:
            raise JsonValidationError()

        client = object.__new__(GroqTranslationClient)
        client.model_id = "openai/gpt-oss-120b"
        client.reasoning_effort = None
        client.rate_limiter = SimpleNamespace(wait_for_request=lambda _prompt, _tokens: None)
        client.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fail_generation))
        )
        with self.assertRaises(StageError) as raised:
            client.generate("prompt", schema_name="translation", schema={}, max_tokens=100)
        self.assertEqual(raised.exception.category, "invalid_generation")
        self.assertTrue(raised.exception.retryable)

    def test_groq_falls_back_to_json_object_after_strict_generation_error(self) -> None:
        class JsonValidationError(RuntimeError):
            status_code = 400

            def __init__(self) -> None:
                super().__init__("{'code': 'json_validate_failed'}")

        requests: list[dict] = []
        def create(**kwargs: object) -> SimpleNamespace:
            requests.append(kwargs)
            if len(requests) == 1:
                raise JsonValidationError()
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(refusal=None, content='{"pages":[]}'))],
                _request_id="request-1",
            )

        client = object.__new__(GroqTranslationClient)
        client.model_id = "openai/gpt-oss-120b"
        client.reasoning_effort = "low"
        class RecordingLimiter:
            def __init__(self) -> None:
                self.reservations: list[tuple[str, int]] = []

            def wait_for_request(self, prompt: str, max_completion_tokens: int) -> None:
                self.reservations.append((prompt, max_completion_tokens))

        limiter = RecordingLimiter()
        client.rate_limiter = limiter
        client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = client.generate("JSON prompt", schema_name="translation", schema={"required": ["pages"]}, max_tokens=100)
        self.assertEqual(result["data"], {"pages": []})
        self.assertTrue(requests[0]["response_format"]["json_schema"]["strict"])
        self.assertEqual(requests[1]["response_format"], {"type": "json_object"})
        self.assertEqual(requests[1]["reasoning_effort"], "low")
        self.assertEqual(limiter.reservations, [("JSON prompt", 100), ("JSON prompt", 100)])

    def test_groq_generation_reserves_budget_for_each_call(self) -> None:
        class RecordingLimiter:
            def __init__(self) -> None:
                self.reservations: list[tuple[str, int]] = []

            def wait_for_request(self, prompt: str, max_completion_tokens: int) -> None:
                self.reservations.append((prompt, max_completion_tokens))

        client = object.__new__(GroqTranslationClient)
        client.model_id = "openai/gpt-oss-120b"
        client.reasoning_effort = None
        client.rate_limiter = RecordingLimiter()
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(refusal=None, content='{"pages":[]}'))],
            _request_id="request-1",
        )
        client.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_kwargs: response))
        )

        for _ in range(2):
            client.generate("translation prompt", schema_name="translation", schema={"required": ["pages"]}, max_tokens=80)

        self.assertEqual(
            client.rate_limiter.reservations,
            [("translation prompt", 80), ("translation prompt", 80)],
        )

    def test_retry_delay_uses_each_provider_response(self) -> None:
        self.assertEqual(
            provider_retry_after_seconds(RuntimeError("Please retry in 28.5s.")), 28.5
        )
        self.assertEqual(
            provider_retry_after_seconds(RuntimeError("Please try again in 17.085s.")), 17.085
        )
        self.assertEqual(
            provider_retry_after_seconds(RuntimeError("Your rate limit resets in ~30s.")), 30
        )
        self.assertEqual(
            provider_retry_after_seconds(RuntimeError("Expected available in 1 second.")), 1
        )
        self.assertAlmostEqual(
            provider_retry_after_seconds(RuntimeError("Please retry in 2h37m7.745s.")),
            9427.745,
        )
        provider_error = HttpError(429)
        provider_error.response = SimpleNamespace(headers={"retry-after": "12"})
        wrapped = StageError("D", "rate_limit", True, "Quota exceeded")
        wrapped.__cause__ = provider_error
        self.assertEqual(provider_retry_after_seconds(wrapped), 12)

    def test_retry_classifier_retries_transient_but_not_auth_errors(self) -> None:
        self.assertTrue(is_transient_provider_error(HttpError(429)))
        self.assertTrue(is_transient_provider_error(HttpError(503)))
        self.assertFalse(is_transient_provider_error(HttpError(401)))
        self.assertFalse(is_transient_provider_error(HttpError(400)))

    def test_provider_retries_are_bounded(self) -> None:
        attempts = 0
        def operation() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise HttpError(429)
            return "ok"
        self.assertEqual(retry_provider_call(operation, max_attempts=3, base_delay_seconds=0), "ok")
        self.assertEqual(attempts, 3)
        attempts = 0
        with self.assertRaises(HttpError):
            retry_provider_call(lambda: (_ for _ in ()).throw(HttpError(401)), max_attempts=3, base_delay_seconds=0)


if __name__ == "__main__":
    unittest.main()
