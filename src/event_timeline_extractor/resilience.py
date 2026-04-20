"""Small retry helpers for transient external failures."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def retry_call(
    func: Callable[[], T],
    *,
    attempts: int,
    should_retry: Callable[[Exception], bool],
    delay_seconds: float = 0.0,
) -> T:
    """Run ``func`` up to ``attempts`` times for retryable exceptions."""
    if attempts <= 0:
        raise ValueError("attempts must be positive")

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= attempts or not should_retry(exc):
                raise
            if delay_seconds > 0:
                time.sleep(delay_seconds)

    assert last_error is not None
    raise last_error
