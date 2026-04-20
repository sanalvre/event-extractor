"""Lightweight filesystem-backed job tracking for the web API."""

from __future__ import annotations

import json
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class JobPaths:
    root: Path
    work_dir: Path
    status_path: Path


def default_jobs_root() -> Path:
    root = Path(tempfile.gettempdir()) / "event_timeline_extractor_jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


class JobStore:
    """Creates and updates job directories on disk."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or default_jobs_root()).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def create_job(
        self,
        *,
        url: str,
        max_seconds: float | None,
        dry_run: bool,
    ) -> tuple[str, JobPaths]:
        job_id = uuid.uuid4().hex
        job_root = self.root / job_id
        work_dir = job_root / "work"
        work_dir.mkdir(parents=True, exist_ok=True)
        paths = JobPaths(root=job_root, work_dir=work_dir, status_path=job_root / "status.json")
        self.write_status(
            job_id,
            status="queued",
            url=url,
            max_seconds=max_seconds,
            dry_run=dry_run,
        )
        return job_id, paths

    def paths_for(self, job_id: str) -> JobPaths:
        job_root = self.root / job_id
        return JobPaths(
            root=job_root,
            work_dir=job_root / "work",
            status_path=job_root / "status.json",
        )

    def has_job(self, job_id: str) -> bool:
        return self.paths_for(job_id).status_path.is_file()

    def read_status(self, job_id: str) -> dict[str, Any] | None:
        path = self.paths_for(job_id).status_path
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def write_status(self, job_id: str, **payload: Any) -> Path:
        existing = self.read_status(job_id) or {}
        stage_history = self._merged_stage_history(existing, payload)
        merged = {
            **existing,
            **payload,
            "job_id": job_id,
            "updated_at": _utc_now(),
            "stage_history": stage_history,
        }
        if "created_at" not in merged:
            merged["created_at"] = merged["updated_at"]
        path = self.paths_for(job_id).status_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def _merged_stage_history(
        self,
        existing: dict[str, Any],
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        raw_existing = existing.get("stage_history")
        history = list(raw_existing) if isinstance(raw_existing, list) else []

        stage = payload.get("stage")
        if not isinstance(stage, str) or not stage:
            return history

        progress = payload.get("progress")
        status = payload.get("status")
        entry = {
            "stage": stage,
            "status": status if isinstance(status, str) else existing.get("status"),
            "message": payload.get("message"),
            "progress": progress if isinstance(progress, dict) else {},
            "recorded_at": _utc_now(),
        }

        if history:
            last = history[-1]
            if (
                isinstance(last, dict)
                and last.get("stage") == entry["stage"]
                and last.get("status") == entry["status"]
                and last.get("message") == entry["message"]
                and last.get("progress") == entry["progress"]
            ):
                return history

        history.append(entry)
        return history
