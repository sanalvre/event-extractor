from event_timeline_extractor.resilience import retry_call


def test_retry_call_retries_until_success() -> None:
    seen = {"count": 0}

    def _run() -> str:
        seen["count"] += 1
        if seen["count"] < 3:
            raise TimeoutError("slow")
        return "ok"

    result = retry_call(
        _run,
        attempts=3,
        delay_seconds=0.0,
        should_retry=lambda exc: isinstance(exc, TimeoutError),
    )

    assert result == "ok"
    assert seen["count"] == 3


def test_retry_call_stops_on_non_retryable_error() -> None:
    seen = {"count": 0}

    def _run() -> None:
        seen["count"] += 1
        raise ValueError("bad input")

    try:
        retry_call(
            _run,
            attempts=3,
            delay_seconds=0.0,
            should_retry=lambda exc: isinstance(exc, TimeoutError),
        )
    except ValueError as exc:
        assert str(exc) == "bad input"
    else:
        raise AssertionError("ValueError was not raised")

    assert seen["count"] == 1
