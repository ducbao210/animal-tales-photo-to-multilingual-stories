from __future__ import annotations

import time
import random
import re
from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")


def provider_retry_after_seconds(error: Exception) -> float | None:
    """Read a provider's retry delay from headers or its error message."""
    current: BaseException | None = error
    for _ in range(3):
        if current is None:
            break
        value = getattr(current, "retry_after", None)
        if value is None:
            headers = getattr(getattr(current, "response", None), "headers", None)
            if headers is not None:
                value = headers.get("retry-after")
        if value is not None:
            try:
                return max(float(value), 0)
            except (TypeError, ValueError):
                pass
        message = str(current)
        compound = re.search(
            r"(?:retry|try\s+again)\s+in\s+((?:\d+(?:\.\d+)?\s*(?:h|m|s)\s*)+)",
            message,
            re.IGNORECASE,
        )
        if compound:
            duration = compound.group(1)
            parts = re.findall(r"(\d+(?:\.\d+)?)\s*(h|m|s)", duration, re.IGNORECASE)
            multipliers = {"h": 3600, "m": 60, "s": 1}
            return sum(float(amount) * multipliers[unit.lower()] for amount, unit in parts)
        for pattern in (
            r"(?:retry|try\s+again)\s+in\s+([0-9]+(?:\.[0-9]+)?)\s*s\b",
            r"rate\s+limit\s+resets\s+in\s+~?\s*([0-9]+(?:\.[0-9]+)?)\s*s\b",
            r"expected\s+available\s+in\s+([0-9]+(?:\.[0-9]+)?)\s+seconds?\b",
        ):
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return float(match.group(1))
        current = current.__cause__
    return None


def retry_call(
    operation: Callable[[], T],
    *,
    max_attempts: int,
    delay_seconds: float = 0,
    should_retry: Callable[[Exception], bool] | None = None,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> T:
    """Retry a synchronous operation a bounded number of times."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if delay_seconds < 0:
        raise ValueError("delay_seconds cannot be negative")

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as error:
            if attempt == max_attempts or (
                should_retry is not None and not should_retry(error)
            ):
                raise
            if on_retry is not None:
                on_retry(attempt, error)
            if delay_seconds:
                time.sleep(delay_seconds)
    raise RuntimeError("Retry loop exited without a result.")


def is_transient_provider_error(error: Exception) -> bool:
    """Return true only for common retryable provider and transport failures."""
    if getattr(error, "retryable", False) is True:
        return True
    status = getattr(error, "status_code", None)
    if status is None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
    if status in (408, 409, 425, 429) or (isinstance(status, int) and status >= 500):
        return True
    name = type(error).__name__.lower()
    return any(token in name for token in ("timeout", "connection", "connect", "ssl", "temporar"))


def retry_provider_call(
    operation: Callable[[], T], *, max_attempts: int = 3,
    base_delay_seconds: float = 1, on_retry: Callable[[int, Exception], None] | None = None,
) -> T:
    """Retry transient provider failures with capped exponential jitter."""
    if max_attempts < 1 or base_delay_seconds < 0:
        raise ValueError("Invalid provider retry settings")
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as error:
            if attempt >= max_attempts or not is_transient_provider_error(error):
                raise
            if on_retry:
                on_retry(attempt, error)
            retry_after = provider_retry_after_seconds(error)
            if retry_after is not None and retry_after > 120:
                raise
            delay = (
                retry_after
                if retry_after is not None
                else min(base_delay_seconds * (2 ** (attempt - 1)), 30)
            )
            time.sleep(delay + random.uniform(0, min(0.25, delay / 4)))
    raise RuntimeError("Provider retry loop exited without a result")
