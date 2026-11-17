"""Per-process rolling request and token limits for the Groq translation client."""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable


class GroqRateLimiter:
    """Reserve request and estimated token capacity over a rolling minute."""

    WINDOW_SECONDS = 60.0

    def __init__(
        self,
        requests_per_minute: int,
        tokens_per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(requests_per_minute, int) or isinstance(requests_per_minute, bool) or requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be a positive integer")
        if not isinstance(tokens_per_minute, int) or isinstance(tokens_per_minute, bool) or tokens_per_minute <= 0:
            raise ValueError("tokens_per_minute must be a positive integer")
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self._clock = clock
        self._sleep = sleep
        self._reservations: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    @staticmethod
    def estimate_tokens(prompt: str, max_completion_tokens: int) -> int:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not isinstance(max_completion_tokens, int) or isinstance(max_completion_tokens, bool) or max_completion_tokens < 0:
            raise ValueError("max_completion_tokens must be a non-negative integer")
        return math.ceil(len(prompt) / 3) + max_completion_tokens

    def wait_for_request(self, prompt: str, max_completion_tokens: int) -> None:
        estimated_tokens = self.estimate_tokens(prompt, max_completion_tokens)
        if estimated_tokens > self.tokens_per_minute:
            raise ValueError(
                "Estimated request token budget exceeds configured tokens_per_minute"
            )

        # Hold the lock while waiting so concurrent callers cannot reserve the
        # same available request or token capacity.
        with self._lock:
            while True:
                now = self._clock()
                while self._reservations and now - self._reservations[0][0] >= self.WINDOW_SECONDS:
                    self._reservations.popleft()

                used_tokens = sum(tokens for _, tokens in self._reservations)
                request_available = len(self._reservations) < self.requests_per_minute
                tokens_available = used_tokens + estimated_tokens <= self.tokens_per_minute
                if request_available and tokens_available:
                    self._reservations.append((now, estimated_tokens))
                    return

                request_wait = 0.0
                if not request_available:
                    request_wait = self._reservations[0][0] + self.WINDOW_SECONDS - now

                token_wait = 0.0
                excess_tokens = used_tokens + estimated_tokens - self.tokens_per_minute
                if excess_tokens > 0:
                    for timestamp, tokens in self._reservations:
                        excess_tokens -= tokens
                        if excess_tokens <= 0:
                            token_wait = timestamp + self.WINDOW_SECONDS - now
                            break

                self._sleep(max(request_wait, token_wait, 0.001))
