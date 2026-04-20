"""Local web UI and JSON API for YouTube timeline extraction."""

from __future__ import annotations

import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from event_timeline_extractor.artifacts import ArtifactStore
from event_timeline_extractor.config import Settings, load_settings
from event_timeline_extractor.fetch import FetchError, is_probably_youtube_url
from event_timeline_extractor.ffmpeg_tools import FFmpegError
from event_timeline_extractor.jobs import JobStore, default_jobs_root
from event_timeline_extractor.pipeline import PipelineInput, run_pipeline
from event_timeline_extractor.schema import TimelineResult

logger = logging.getLogger(__name__)

_WEB_WINDOW_SEC = 20.0
_WEB_MAX_MINUTES = 30.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_progress_update(stage: str, payload: dict) -> dict:
    if stage == "queued":
        return {
            "stage": "queued",
            "progress": payload,
            "message": "Job queued.",
            "progress_percent": 0,
        }
    if stage == "preparing_media":
        return {
            "stage": stage,
            "progress": payload,
            "message": "Preparing YouTube media.",
            "progress_percent": 10,
        }
    if stage == "transcribing":
        reused = bool(payload.get("reused"))
        return {
            "stage": stage,
            "progress": payload,
            "message": "Reusing transcript artifacts." if reused else "Transcribing audio.",
            "progress_percent": 45 if reused else 30,
        }
    if stage == "windowing":
        reused = bool(payload.get("reused"))
        return {
            "stage": stage,
            "progress": payload,
            "message": "Reusing transcript windows." if reused else "Building transcript windows.",
            "progress_percent": 60 if reused else 50,
        }
    if stage == "extracting":
        batch_index = int(payload.get("batch_index", 0))
        total_batches = max(int(payload.get("total_batches", 1)), 1)
        current = min(batch_index + 1, total_batches)
        percent = 60 + round((current / total_batches) * 30)
        return {
            "stage": stage,
            "progress": payload,
            "message": f"Extracting events: batch {current} of {total_batches}.",
            "progress_percent": percent,
        }
    if stage == "validating":
        return {
            "stage": stage,
            "progress": payload,
            "message": "Validating extracted evidence.",
            "progress_percent": 95,
        }
    if stage == "completed":
        events_count = payload.get("events_count")
        suffix = "" if events_count is None else f" {events_count} event(s) ready."
        return {
            "stage": stage,
            "progress": payload,
            "message": f"Timeline extraction completed.{suffix}",
            "progress_percent": 100,
        }
    if stage == "failed":
        return {
            "stage": stage,
            "progress": payload,
            "message": "Timeline extraction failed.",
            "progress_percent": 100,
        }
    return {
        "stage": stage,
        "progress": payload,
        "message": f"Job stage: {stage}.",
        "progress_percent": None,
    }


def _pipeline_error_message(exc: Exception) -> str:
    """User-visible string for HTTP 500 responses."""
    if isinstance(exc, FetchError):
        return str(exc)
    if isinstance(exc, FFmpegError):
        return f"ffmpeg/ffprobe: {exc}"
    msg = str(exc).strip()
    if "OPENROUTER_API_KEY" in msg:
        return (
            f"{msg} Set OPENROUTER_API_KEY in `.env` or the environment, "
            "then restart the server or use Dry run."
        )
    return msg


def _job_failure_payload(
    *,
    stage: str | None,
    exc: Exception,
    artifacts_dir: Path,
) -> dict:
    error_message = _pipeline_error_message(exc)
    payload = _job_progress_update("failed", {"failed_stage": stage})
    payload["message"] = (
        f"Timeline extraction failed during {stage.replace('_', ' ')}."
        if stage
        else "Timeline extraction failed."
    )
    payload["last_error"] = {
        "stage": stage,
        "message": error_message,
        "type": type(exc).__name__,
        "recorded_at": _utc_now(),
    }
    payload["error"] = error_message
    payload["artifacts"] = {
        "artifacts_dir": str(artifacts_dir),
        "status_path": "status.json",
        "timeline_exists": (artifacts_dir / "timeline.json").is_file(),
        "transcript_exists": (artifacts_dir / "transcript.json").is_file(),
        "windows_exists": (artifacts_dir / "windows.json").is_file(),
        "run_summary_exists": (artifacts_dir / "run_summary.json").is_file(),
    }
    return payload


def _run_youtube_timeline(
    *,
    settings: Settings,
    url: str,
    max_seconds: float | None,
    dry_run: bool,
) -> TimelineResult:
    with tempfile.TemporaryDirectory(prefix="ete_web_") as tmp:
        work = Path(tmp)
        try:
            return run_pipeline(
                PipelineInput(youtube_url=url),
                work_dir=work,
                settings=settings,
                window_sec=_WEB_WINDOW_SEC,
                max_minutes=_WEB_MAX_MINUTES,
                max_seconds=max_seconds,
                dry_run=dry_run,
            )
        except Exception as exc:
            logger.exception("Pipeline failed (youtube_url=%r dry_run=%s)", url, dry_run)
            raise HTTPException(
                status_code=500,
                detail=_pipeline_error_message(exc),
            ) from exc


class TimelineRequest(BaseModel):
    url: str = Field(..., min_length=8, description="YouTube https URL")
    max_seconds: float | None = Field(
        None,
        ge=1.0,
        description="Optional: only first N seconds for a smaller YouTube processing run.",
    )

    def https_youtube_only(self) -> str:
        url = self.url.strip()
        if not url.startswith("https://"):
            raise ValueError("Only https:// URLs are accepted.")
        if not is_probably_youtube_url(url):
            raise ValueError("Only YouTube URLs are supported.")
        return url


class JobResponse(BaseModel):
    job_id: str
    status: str


def _require_https_youtube(body: TimelineRequest) -> str:
    try:
        return body.https_youtube_only()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def get_settings() -> Settings:
    return load_settings()


def get_job_store() -> JobStore:
    return JobStore(default_jobs_root())


def _run_job(
    *,
    job_store: JobStore,
    settings: Settings,
    job_id: str,
    url: str,
    max_seconds: float | None,
    dry_run: bool,
) -> None:
    job_store.write_status(job_id, status="running", **_job_progress_update("queued", {}))
    paths = job_store.paths_for(job_id)
    last_stage: str | None = "queued"

    def _progress(stage: str, payload: dict) -> None:
        nonlocal last_stage
        last_stage = stage
        job_store.write_status(
            job_id,
            status="running",
            **_job_progress_update(stage, payload),
        )

    try:
        result = run_pipeline(
            PipelineInput(youtube_url=url),
            work_dir=paths.work_dir,
            settings=settings,
            window_sec=_WEB_WINDOW_SEC,
            max_minutes=_WEB_MAX_MINUTES,
            max_seconds=max_seconds,
            dry_run=dry_run,
            progress_callback=_progress,
        )
        ArtifactStore(paths.work_dir).write_timeline_result(result)
        job_store.write_status(
            job_id,
            status="completed",
            artifacts_dir=str(paths.work_dir / "artifacts"),
            events_count=len(result.events),
            **_job_progress_update("completed", {"events_count": len(result.events)}),
        )
    except Exception as exc:
        logger.exception("Job failed (job_id=%s youtube_url=%r dry_run=%s)", job_id, url, dry_run)
        job_store.write_status(
            job_id,
            status="failed",
            artifacts_dir=str(paths.work_dir / "artifacts"),
            **_job_failure_payload(
                stage=last_stage,
                exc=exc,
                artifacts_dir=paths.work_dir / "artifacts",
            ),
        )


def _load_job_result(job_store: JobStore, job_id: str) -> TimelineResult:
    paths = job_store.paths_for(job_id)
    artifacts = ArtifactStore(paths.work_dir)
    payload = artifacts.read_json("timeline.json")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=404, detail="Timeline result not found.")
    return TimelineResult.model_validate(payload)


def create_app(*, static_dir: Path | None = None, jobs_root: Path | None = None) -> FastAPI:
    """Factory for the localhost YouTube timeline app."""
    if static_dir is None:
        static_dir = Path(__file__).resolve().parent / "static"

    app = FastAPI(
        title="Event Timeline Extractor",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
    )
    app.state.job_store = JobStore(jobs_root or default_jobs_root())

    @app.get("/")
    def index() -> FileResponse:
        index_path = static_dir / "index.html"
        if not index_path.is_file():
            raise HTTPException(status_code=500, detail="UI bundle missing.")
        return FileResponse(index_path)

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        """Browsers request this automatically; avoid noisy 404s."""
        return Response(status_code=204)

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/api/health")
    def health() -> dict[str, bool]:
        """Lightweight check that the localhost server is responding."""
        return {"ok": True}

    @app.post("/api/jobs", response_model=JobResponse)
    def post_job(
        body: TimelineRequest,
        background_tasks: BackgroundTasks,
        settings: Annotated[Settings, Depends(get_settings)],
    ) -> JobResponse:
        url = _require_https_youtube(body)
        job_store = app.state.job_store
        job_id, _paths = job_store.create_job(
            url=url,
            max_seconds=body.max_seconds,
            dry_run=False,
        )
        background_tasks.add_task(
            _run_job,
            job_store=job_store,
            settings=settings,
            job_id=job_id,
            url=url,
            max_seconds=body.max_seconds,
            dry_run=False,
        )
        return JobResponse(job_id=job_id, status="queued")

    @app.post("/api/jobs/dry-run", response_model=JobResponse)
    def post_job_dry_run(
        body: TimelineRequest,
        background_tasks: BackgroundTasks,
        settings: Annotated[Settings, Depends(get_settings)],
    ) -> JobResponse:
        url = _require_https_youtube(body)
        job_store = app.state.job_store
        job_id, _paths = job_store.create_job(
            url=url,
            max_seconds=body.max_seconds,
            dry_run=True,
        )
        background_tasks.add_task(
            _run_job,
            job_store=job_store,
            settings=settings,
            job_id=job_id,
            url=url,
            max_seconds=body.max_seconds,
            dry_run=True,
        )
        return JobResponse(job_id=job_id, status="queued")

    @app.get("/api/jobs/{job_id}")
    def get_job_status(job_id: str) -> JSONResponse:
        payload = app.state.job_store.read_status(job_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return JSONResponse(content=payload)

    @app.get("/api/jobs/{job_id}/result")
    def get_job_result(job_id: str) -> JSONResponse:
        payload = app.state.job_store.read_status(job_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        if payload.get("status") != "completed":
            raise HTTPException(status_code=409, detail="Job is not completed yet.")
        result = _load_job_result(app.state.job_store, job_id)
        return JSONResponse(content=result.model_dump())

    @app.post("/api/timeline")
    def post_timeline(
        body: TimelineRequest,
        settings: Annotated[Settings, Depends(get_settings)],
    ) -> JSONResponse:
        url = _require_https_youtube(body)
        result = _run_youtube_timeline(
            settings=settings,
            url=url,
            max_seconds=body.max_seconds,
            dry_run=False,
        )
        return JSONResponse(content=result.model_dump())

    @app.post("/api/timeline/dry-run")
    def post_timeline_dry_run(
        body: TimelineRequest,
        settings: Annotated[Settings, Depends(get_settings)],
    ) -> JSONResponse:
        url = _require_https_youtube(body)
        result = _run_youtube_timeline(
            settings=settings,
            url=url,
            max_seconds=body.max_seconds,
            dry_run=True,
        )
        return JSONResponse(content=result.model_dump())

    return app


app = create_app()
