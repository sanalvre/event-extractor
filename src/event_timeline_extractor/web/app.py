"""Local web UI + JSON API."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from event_timeline_extractor.config import Settings, load_settings
from event_timeline_extractor.fetch import FetchError, is_probably_youtube_url
from event_timeline_extractor.ffmpeg_tools import FFmpegError
from event_timeline_extractor.pipeline import PipelineInput, run_pipeline
from event_timeline_extractor.schema import TimelineResult

logger = logging.getLogger(__name__)

# Kept in one place so CLI defaults and docs stay aligned with the web UI.
_WEB_WINDOW_SEC = 20.0
_WEB_MAX_MINUTES = 30.0


def _pipeline_error_message(exc: Exception) -> str:
    """User-visible string for HTTP 500 (also logged with traceback)."""
    if isinstance(exc, FetchError):
        return str(exc)
    if isinstance(exc, FFmpegError):
        return f"ffmpeg/ffprobe: {exc}"
    msg = str(exc).strip()
    if "OPENROUTER_API_KEY" in msg:
        return (
            f"{msg} Set OPENROUTER_API_KEY in `.env` or the environment, "
            "then restart the server — or use Dry run."
        )
    return msg


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
        except Exception as e:
            logger.exception("Pipeline failed (youtube_url=%r dry_run=%s)", url, dry_run)
            raise HTTPException(
                status_code=500,
                detail=_pipeline_error_message(e),
            ) from e


class TimelineRequest(BaseModel):
    url: str = Field(..., min_length=8, description="YouTube https URL")
    max_seconds: float | None = Field(
        None,
        ge=1.0,
        description="Optional: only first N seconds (smaller download via yt-dlp sections).",
    )

    def https_youtube_only(self) -> str:
        u = self.url.strip()
        if not u.startswith("https://"):
            raise ValueError("Only https:// URLs are accepted.")
        if not is_probably_youtube_url(u):
            raise ValueError("Only YouTube URLs are supported.")
        return u


def _require_https_youtube(body: TimelineRequest) -> str:
    try:
        return body.https_youtube_only()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def get_settings() -> Settings:
    return load_settings()


def create_app(*, static_dir: Path | None = None) -> FastAPI:
    """Factory for FastAPI app. `static_dir` defaults to package `web/static`."""
    if static_dir is None:
        static_dir = Path(__file__).resolve().parent / "static"

    app = FastAPI(
        title="Event Timeline Extractor",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
    )

    @app.get("/")
    def index() -> FileResponse:
        index_path = static_dir / "index.html"
        if not index_path.is_file():
            raise HTTPException(status_code=500, detail="UI bundle missing.")
        return FileResponse(index_path)

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        """Browsers request this by default; avoid noisy 404s in DevTools."""
        return Response(status_code=204)

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/api/health")
    def health() -> dict[str, bool]:
        """Lightweight check that the server is up (use before long timeline runs)."""
        return {"ok": True}

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


# Uvicorn: `uvicorn event_timeline_extractor.web.app:app --host 127.0.0.1 --port 8765`
app = create_app()
