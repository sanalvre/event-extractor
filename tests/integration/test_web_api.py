"""FastAPI routes with mocked heavy pipeline."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from event_timeline_extractor.schema import TimelineEvent, TimelineResult


def test_get_index(web_client: TestClient) -> None:
    r = web_client.get("/")
    assert r.status_code == 200
    assert b"Event Timeline" in r.content
    assert b"OPENROUTER_API_KEY" not in r.content


def test_health(web_client: TestClient) -> None:
    r = web_client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_favicon_returns_no_content(web_client: TestClient) -> None:
    r = web_client.get("/favicon.ico")
    assert r.status_code == 204


@patch("event_timeline_extractor.web.app.run_pipeline")
def test_post_timeline_dry_run_mocked(mock_run, web_client: TestClient) -> None:
    mock_run.return_value = TimelineResult(
        events=[TimelineEvent(time="00:00", event="Mock event")],
        meta={"test": True},
    )
    r = web_client.post(
        "/api/timeline/dry-run",
        json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["events"][0]["event"] == "Mock event"


def test_post_timeline_bad_url(web_client: TestClient) -> None:
    r = web_client.post("/api/timeline/dry-run", json={"url": "http://example.com"})
    assert r.status_code == 400
