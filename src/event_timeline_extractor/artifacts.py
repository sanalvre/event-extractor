"""Persistent artifact helpers for pipeline runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from event_timeline_extractor.chunking import TimeWindow
from event_timeline_extractor.schema import TimelineResult
from event_timeline_extractor.transcription.base import TranscriptSegment

if TYPE_CHECKING:
    from event_timeline_extractor.pipeline import (
        MediaStage,
        PipelineInput,
        TranscriptStage,
        WindowStage,
    )


class ArtifactStore:
    """Writes and reads resume-friendly pipeline artifacts under ``work_dir/artifacts``."""

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir
        self.root = work_dir / "artifacts"
        self.root.mkdir(parents=True, exist_ok=True)
        self.batch_root = self.root / "batches"
        self.batch_root.mkdir(parents=True, exist_ok=True)

    def write_input_manifest(self, inp: PipelineInput) -> Path:
        payload = {
            "youtube_url": inp.youtube_url,
            "file_path": str(inp.file_path) if inp.file_path is not None else None,
        }
        return self._write_json("input.json", payload)

    def load_input_manifest(self) -> dict[str, Any] | None:
        return self.read_json("input.json")

    def input_matches(self, inp: PipelineInput) -> bool:
        payload = self.load_input_manifest()
        if payload is None:
            return False
        return payload == {
            "youtube_url": inp.youtube_url,
            "file_path": str(inp.file_path) if inp.file_path is not None else None,
        }

    def write_media_stage(self, media_stage: MediaStage) -> Path:
        payload = {
            "media_path": str(media_stage.media_path),
            "duration_seconds": media_stage.duration_seconds,
            "wav_path": str(media_stage.wav_path),
            "input_kind": media_stage.input_kind,
        }
        return self._write_json("media.json", payload)

    def load_media_stage(self) -> dict[str, Any] | None:
        return self.read_json("media.json")

    def write_transcript_stage(self, transcript_stage: TranscriptStage) -> Path:
        payload = {
            "transcriber": transcript_stage.transcriber_name,
            "asr_model": transcript_stage.asr_model,
            "full_transcript": transcript_stage.full_transcript,
            "segments": [_segment_to_dict(segment) for segment in transcript_stage.segments],
        }
        return self._write_json("transcript.json", payload)

    def load_transcript_stage(self) -> dict[str, Any] | None:
        return self.read_json("transcript.json")

    def write_window_stage(self, window_stage: WindowStage, *, window_sec: float) -> Path:
        payload = {
            "speaker_aware": window_stage.speaker_aware,
            "vision_enabled": window_stage.vision_enabled,
            "window_sec": window_sec,
            "windows": [_window_to_dict(window) for window in window_stage.windows],
        }
        return self._write_json("windows.json", payload)

    def load_window_stage(self) -> dict[str, Any] | None:
        return self.read_json("windows.json")

    def write_timeline_result(self, result: TimelineResult) -> Path:
        return self._write_json("timeline.json", result.model_dump())

    def write_run_summary(self, summary: dict[str, Any]) -> Path:
        return self._write_json("run_summary.json", summary)

    def load_run_summary(self) -> dict[str, Any] | None:
        return self.read_json("run_summary.json")

    def write_batch_result(
        self,
        batch_index: int,
        *,
        dry_run: bool,
        window_start: int,
        window_end: int,
        result: TimelineResult,
    ) -> Path:
        payload = {
            "batch_index": batch_index,
            "dry_run": dry_run,
            "window_start": window_start,
            "window_end": window_end,
            "result": result.model_dump(),
        }
        return self._write_batch_json(batch_index, payload)

    def load_batch_result(self, batch_index: int) -> dict[str, Any] | None:
        path = self.batch_path(batch_index)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def batch_path(self, batch_index: int) -> Path:
        return self.batch_root / f"batch_{batch_index:04d}.json"

    def read_json(self, name: str) -> dict[str, Any] | list[Any] | None:
        path = self.root / name
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_json(self, name: str, payload: Any) -> Path:
        path = self.root / name
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def _write_batch_json(self, batch_index: int, payload: Any) -> Path:
        path = self.batch_path(batch_index)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path


def segment_from_dict(payload: dict[str, Any]) -> TranscriptSegment:
    return TranscriptSegment(
        start=float(payload["start"]),
        end=float(payload["end"]),
        text=str(payload["text"]),
        speaker=payload.get("speaker"),
        segment_id=payload.get("segment_id"),
    )


def window_from_dict(payload: dict[str, Any]) -> TimeWindow:
    return TimeWindow(
        start=float(payload["start"]),
        end=float(payload["end"]),
        text=str(payload["text"]),
        frame_paths=[str(path) for path in payload.get("frame_paths", [])],
        vision_context=str(payload.get("vision_context", "")),
        source_segment_ids=[
            str(seg_id) for seg_id in payload.get("source_segment_ids", [])
        ]
        or None,
    )


def timeline_result_from_dict(payload: dict[str, Any]) -> TimelineResult:
    return TimelineResult.model_validate(payload)


def _segment_to_dict(segment: TranscriptSegment) -> dict[str, Any]:
    return {
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "speaker": segment.speaker,
        "segment_id": segment.segment_id,
    }


def _window_to_dict(window: TimeWindow) -> dict[str, Any]:
    return {
        "start": window.start,
        "end": window.end,
        "text": window.text,
        "frame_paths": list(window.frame_paths),
        "vision_context": window.vision_context,
        "source_segment_ids": list(window.source_segment_ids or []),
    }
