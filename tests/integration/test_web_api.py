"""FastAPI routes with mocked heavy pipeline."""

import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from event_timeline_extractor.schema import TimelineEvent, TimelineResult
from event_timeline_extractor.web.app import _job_progress_update


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


@patch("event_timeline_extractor.web.app.run_pipeline")
def test_submit_job_and_fetch_result(mock_run, web_client: TestClient) -> None:
    def _fake_run_pipeline(*args, **kwargs):
        progress_callback = kwargs.get("progress_callback")
        if progress_callback is not None:
            progress_callback("preparing_media", {})
            progress_callback("transcribing", {"reused": False})
            progress_callback("extracting", {"batch_index": 0, "total_batches": 1})
        return TimelineResult(
            events=[TimelineEvent(time="00:00", event="Job event")],
            meta={"test": True},
        )

    mock_run.side_effect = _fake_run_pipeline

    r = web_client.post(
        "/api/jobs/dry-run",
        json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
    )
    assert r.status_code == 200
    job = r.json()
    assert job["status"] == "queued"
    job_id = job["job_id"]

    status_r = web_client.get(f"/api/jobs/{job_id}")
    assert status_r.status_code == 200
    status_payload = status_r.json()
    assert status_payload["status"] == "completed"
    assert status_payload["stage"] == "completed"
    assert status_payload["message"].startswith("Timeline extraction completed.")
    assert status_payload["progress_percent"] == 100
    assert status_payload["progress"]["events_count"] == 1
    assert status_payload["events_count"] == 1
    assert status_payload["stage_history"]
    assert status_payload["stage_history"][0]["stage"] == "queued"
    assert status_payload["stage_history"][-1]["stage"] == "completed"

    result_r = web_client.get(f"/api/jobs/{job_id}/result")
    assert result_r.status_code == 200
    result_payload = result_r.json()
    assert result_payload["events"][0]["event"] == "Job event"


@patch("event_timeline_extractor.web.app.run_pipeline")
def test_failed_job_status_includes_failure_details(mock_run, web_client: TestClient) -> None:
    def _fake_run_pipeline(*args, **kwargs):
        progress_callback = kwargs.get("progress_callback")
        if progress_callback is not None:
            progress_callback("preparing_media", {})
            progress_callback("transcribing", {"reused": False})
        raise RuntimeError("network went away")

    mock_run.side_effect = _fake_run_pipeline

    r = web_client.post(
        "/api/jobs",
        json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    status_r = web_client.get(f"/api/jobs/{job_id}")
    assert status_r.status_code == 200
    payload = status_r.json()
    assert payload["status"] == "failed"
    assert payload["stage"] == "failed"
    assert "transcribing" in payload["message"]
    assert payload["last_error"]["stage"] == "transcribing"
    assert payload["last_error"]["message"] == "network went away"
    assert payload["artifacts"]["transcript_exists"] is False
    assert payload["stage_history"][-1]["stage"] == "failed"


def test_job_status_not_found(web_client: TestClient) -> None:
    r = web_client.get("/api/jobs/does-not-exist")
    assert r.status_code == 404


def test_job_result_not_ready_returns_conflict(web_client: TestClient) -> None:
    job_store = web_client.app.state.job_store
    job_id, _paths = job_store.create_job(
        url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        max_seconds=None,
        dry_run=True,
    )
    job_store.write_status(job_id, status="running")

    r = web_client.get(f"/api/jobs/{job_id}/result")
    assert r.status_code == 409


def test_job_status_includes_human_readable_progress() -> None:
    payload = _job_progress_update("extracting", {"batch_index": 1, "total_batches": 4})
    assert payload["message"] == "Extracting events: batch 2 of 4."
    assert payload["progress_percent"] == 75


def test_job_result_reads_saved_timeline_file(web_client: TestClient) -> None:
    job_store = web_client.app.state.job_store
    job_id, paths = job_store.create_job(
        url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        max_seconds=None,
        dry_run=True,
    )
    artifacts_dir = paths.work_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "timeline.json").write_text(
        json.dumps({"events": [{"time": "00:00", "event": "Saved"}], "meta": {"ok": True}}),
        encoding="utf-8",
    )
    job_store.write_status(job_id, status="completed")

    r = web_client.get(f"/api/jobs/{job_id}/result")
    assert r.status_code == 200
    assert r.json()["events"][0]["event"] == "Saved"
