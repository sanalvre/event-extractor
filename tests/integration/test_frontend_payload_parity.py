"""Ensure POST bodies match ``web/static/app.js`` and invoke ``run_pipeline`` like production."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from event_timeline_extractor.schema import TimelineEvent, TimelineResult

_YT = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def _frontend_payload(*, max_seconds: float | None) -> dict:
    """Mirror ``app.js``: ``payload = { url };`` then ``payload.max_seconds = …`` if set."""
    payload: dict = {"url": _YT}
    if max_seconds is not None:
        payload["max_seconds"] = max_seconds
    return payload


@patch("event_timeline_extractor.web.app.run_pipeline")
def test_post_timeline_matches_ui_full_video(mock_run, web_client: TestClient) -> None:
    """UI preset \"Full video\" sends only ``url`` (no ``max_seconds``)."""
    mock_run.return_value = TimelineResult(
        events=[TimelineEvent(time="00:00", event="x")],
        meta={},
    )
    r = web_client.post("/api/timeline/dry-run", json=_frontend_payload(max_seconds=None))
    assert r.status_code == 200
    inp = mock_run.call_args.args[0]
    kw = mock_run.call_args.kwargs
    assert inp.youtube_url == _YT
    assert kw["max_seconds"] is None
    assert kw["window_sec"] == 20.0
    assert kw["max_minutes"] == 30.0
    assert kw["dry_run"] is True


@patch("event_timeline_extractor.web.app.run_pipeline")
def test_post_timeline_matches_ui_first_20_seconds(mock_run, web_client: TestClient) -> None:
    """UI preset \"First 20 seconds\" → ``max_seconds: 20``."""
    mock_run.return_value = TimelineResult(events=[], meta={})
    r = web_client.post("/api/timeline/dry-run", json=_frontend_payload(max_seconds=20))
    assert r.status_code == 200
    assert mock_run.call_args.kwargs["max_seconds"] == 20


@patch("event_timeline_extractor.web.app.run_pipeline")
def test_post_timeline_matches_ui_custom_seconds(mock_run, web_client: TestClient) -> None:
    """UI custom duration: same JSON shape as ``resolveMaxSeconds()`` → ``max_seconds``."""
    mock_run.return_value = TimelineResult(events=[], meta={})
    r = web_client.post("/api/timeline/dry-run", json=_frontend_payload(max_seconds=45))
    assert r.status_code == 200
    assert mock_run.call_args.kwargs["max_seconds"] == 45


@patch("event_timeline_extractor.web.app.run_pipeline")
def test_post_timeline_live_endpoint_same_kwargs(mock_run, web_client: TestClient) -> None:
    """Non-dry ``/api/timeline`` uses the same ``run_pipeline`` kwargs except ``dry_run``."""
    mock_run.return_value = TimelineResult(events=[], meta={})
    r = web_client.post("/api/timeline", json=_frontend_payload(max_seconds=120))
    assert r.status_code == 200
    kw = mock_run.call_args.kwargs
    assert kw["max_seconds"] == 120
    assert kw["dry_run"] is False
