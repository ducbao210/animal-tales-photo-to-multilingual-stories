from __future__ import annotations

import threading
import unittest

from src.translation.rate_limit import GroqRateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []
        self.lock = threading.Lock()

    def monotonic(self) -> float:
        with self.lock:
            return self.now

    def sleep(self, seconds: float) -> None:
        with self.lock:
            self.sleeps.append(seconds)
            self.now += seconds


class TestGroqRateLimiter(unittest.TestCase):
    def test_waits_when_request_budget_is_full(self) -> None:
        clock = FakeClock()
        limiter = GroqRateLimiter(
            2, 100, clock=clock.monotonic, sleep=clock.sleep
        )

        for _ in range(3):
            limiter.wait_for_request("", 1)

        self.assertEqual(len(clock.sleeps), 1)
        self.assertGreaterEqual(clock.sleeps[0], 60)

    def test_waits_when_token_budget_is_full(self) -> None:
        clock = FakeClock()
        limiter = GroqRateLimiter(
            30, 10, clock=clock.monotonic, sleep=clock.sleep
        )

        limiter.wait_for_request("x" * 15, 5)  # 5 estimated prompt + 5 output tokens
        limiter.wait_for_request("", 1)

        self.assertEqual(len(clock.sleeps), 1)
        self.assertGreaterEqual(clock.sleeps[0], 60)

    def test_rejects_a_single_request_larger_than_tpm(self) -> None:
        limiter = GroqRateLimiter(30, 10, clock=lambda: 0, sleep=lambda _delay: None)
        with self.assertRaisesRegex(ValueError, "exceeds.*tokens_per_minute"):
            limiter.wait_for_request("x" * 30, 1)

    def test_rejects_non_positive_limits(self) -> None:
        for requests_per_minute, tokens_per_minute in ((0, 100), (30, 0), (-1, 100)):
            with self.subTest(requests_per_minute=requests_per_minute, tokens_per_minute=tokens_per_minute):
                with self.assertRaises(ValueError):
                    GroqRateLimiter(requests_per_minute, tokens_per_minute)

    def test_concurrent_calls_reserve_distinct_request_slots(self) -> None:
        clock = FakeClock()
        limiter = GroqRateLimiter(
            1, 100, clock=clock.monotonic, sleep=clock.sleep
        )
        workers = [
            threading.Thread(target=limiter.wait_for_request, args=("", 1))
            for _ in range(3)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()

        self.assertEqual(len(clock.sleeps), 2)
        self.assertGreaterEqual(sum(clock.sleeps), 120)


if __name__ == "__main__":
    unittest.main()
