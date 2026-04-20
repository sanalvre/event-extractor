"""End-to-end YouTube/file pipeline for grounded event timelines."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from event_timeline_extractor.artifacts import (
    ArtifactStore,
    segment_from_dict,
    timeline_result_from_dict,
    window_from_dict,
)
from event_timeline_extractor.chunking import TimeWindow, chunk_segments
from event_timeline_extractor.config import Settings, load_settings
from event_timeline_extractor.fetch import run_ytdlp_download
from event_timeline_extractor.ffmpeg_tools import (
    extract_audio_wav_16k_mono,
    extract_frames_every_interval,
    probe_duration_seconds,
)
from event_timeline_extractor.llm.openrouter import TimelineSynthesizer
from event_timeline_extractor.postprocess import (
    filter_low_signal_events,
    merge_adjacent_duplicate_events,
)
from event_timeline_extractor.schema import TimelineResult
from event_timeline_extractor.timefmt import download_section_first_seconds
from event_timeline_extractor.transcription.base import TranscriptSegment
from event_timeline_extractor.transcription.diarization import maybe_apply_diarization
from event_timeline_extractor.transcription.factory import get_transcriber
from event_timeline_extractor.validation import validate_timeline_evidence

logger = logging.getLogger(__name__)
_DEFAULT_EXTRACTION_BATCH_SIZE = 8
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class PipelineInput:
    youtube_url: str | None = None
    file_path: Path | None = None


@dataclass(frozen=True)
class MediaStage:
    media_path: Path
    duration_seconds: float
    wav_path: Path
    input_kind: str


@dataclass(frozen=True)
class TranscriptStage:
    segments: list[TranscriptSegment]
    full_transcript: str
    transcriber_name: str
    asr_model: str


@dataclass(frozen=True)
class WindowStage:
    windows: list[TimeWindow]
    speaker_aware: bool
    vision_enabled: bool


@dataclass(frozen=True)
class ExtractionBatchPlan:
    requested_batch_size: int
    effective_batch_size: int
    total_windows: int
    total_batches: int
    max_batches_target: int
    max_batch_size: int


def resolve_media_path(
    inp: PipelineInput,
    work_dir: Path,
    *,
    max_seconds: float | None = None,
) -> Path:
    if inp.file_path is not None:
        path = inp.file_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(str(path))
        return path
    if inp.youtube_url:
        section = None
        if max_seconds is not None and max_seconds > 0:
            section = download_section_first_seconds(max_seconds)
        return run_ytdlp_download(
            inp.youtube_url,
            work_dir / "download",
            download_sections=section,
        )
    raise ValueError("Provide youtube_url or file_path.")


def _input_kind(inp: PipelineInput) -> str:
    if inp.youtube_url:
        return "youtube"
    if inp.file_path is not None:
        return "file"
    raise ValueError("Provide youtube_url or file_path.")


def _normalize_transcriber_name(settings: Settings) -> str:
    name = (settings.ete_transcriber or "faster_whisper").lower().strip()
    if name in ("faster-whisper", "whisper"):
        return "faster_whisper"
    if name in ("fake", "test"):
        return "stub"
    if settings.ete_use_stub and name == "faster_whisper":
        return "stub"
    return name or "faster_whisper"


def _asr_model_name(settings: Settings, transcriber_name: str) -> str:
    if transcriber_name == "faster_whisper":
        return settings.ete_whisper_model_size
    if transcriber_name == "groq":
        return settings.groq_model
    if transcriber_name == "memories":
        return "memories-whisper-1"
    if transcriber_name == "stub":
        return "stub"
    return transcriber_name


def _assign_segment_ids(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    out: list[TranscriptSegment] = []
    for index, segment in enumerate(segments, start=1):
        seg_id = segment.segment_id or f"seg-{index:06d}"
        out.append(
            TranscriptSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                speaker=segment.speaker,
                segment_id=seg_id,
            )
        )
    return out


def _prepare_media_stage(
    inp: PipelineInput,
    work_dir: Path,
    *,
    max_minutes: float | None,
    max_seconds: float | None,
) -> MediaStage:
    media = resolve_media_path(inp, work_dir, max_seconds=max_seconds)
    duration = probe_duration_seconds(media)
    if max_seconds is not None:
        duration = min(duration, float(max_seconds))
    if max_minutes is not None and duration > max_minutes * 60:
        raise ValueError(f"Media duration {duration:.0f}s exceeds --max-minutes={max_minutes}.")

    wav = work_dir / "audio_16k.wav"
    audio_cap = max_seconds if max_seconds is not None and max_seconds > 0 else None
    extract_audio_wav_16k_mono(media, wav, max_duration_sec=audio_cap)
    return MediaStage(
        media_path=media,
        duration_seconds=duration,
        wav_path=wav,
        input_kind=_input_kind(inp),
    )


def _run_vision_analysis(
    media: Path,
    work_dir: Path,
    settings: Settings,
    duration: float,
) -> dict[float, str]:
    """Extract frames and run memories-s0 vision analysis."""
    from event_timeline_extractor.vision.memories_s0 import MemoriesS0Analyzer  # noqa: PLC0415

    interval = float(settings.ete_vision_frame_interval)
    frames_dir = work_dir / "frames"
    logger.info("Extracting frames every %gs for visual analysis...", interval)
    frame_paths = extract_frames_every_interval(media, frames_dir, interval_sec=interval)
    if not frame_paths:
        logger.warning("No frames extracted; skipping vision analysis.")
        return {}

    timestamps = [min(i * interval, duration) for i in range(len(frame_paths))]
    logger.info("Running memories-s0 on %d frames...", len(frame_paths))
    analyzer = MemoriesS0Analyzer()
    descriptions = analyzer.analyze([str(path) for path in frame_paths], timestamps)
    return {desc.timestamp: desc.description for desc in descriptions}


def _run_transcription_stage(
    media_stage: MediaStage,
    settings: Settings,
) -> TranscriptStage:
    transcriber_name = _normalize_transcriber_name(settings)
    transcriber = get_transcriber(settings)
    segments = transcriber.transcribe(media_stage.wav_path)
    segments = maybe_apply_diarization(settings, media_stage.wav_path, segments)
    segments = _assign_segment_ids(segments)
    full_transcript = " ".join(segment.text for segment in segments)
    return TranscriptStage(
        segments=segments,
        full_transcript=full_transcript,
        transcriber_name=transcriber_name,
        asr_model=_asr_model_name(settings, transcriber_name),
    )


def _run_window_stage(
    media_stage: MediaStage,
    transcript_stage: TranscriptStage,
    work_dir: Path,
    settings: Settings,
    *,
    window_sec: float,
) -> WindowStage:
    vision_map: dict[float, str] | None = None
    if settings.ete_vision_enabled:
        vision_map = _run_vision_analysis(
            media_stage.media_path,
            work_dir,
            settings,
            media_stage.duration_seconds,
        )

    speaker_aware = (
        transcript_stage.transcriber_name == "memories"
        and settings.memories_transcription_speaker
        and any(segment.speaker for segment in transcript_stage.segments)
    )
    windows = chunk_segments(
        transcript_stage.segments,
        window_sec=window_sec,
        vision_map=vision_map,
        speaker_aware=speaker_aware,
    )
    return WindowStage(
        windows=windows,
        speaker_aware=speaker_aware,
        vision_enabled=settings.ete_vision_enabled,
    )


def _merge_meta(base: dict, result: TimelineResult) -> TimelineResult:
    return TimelineResult(events=result.events, meta={**base, **(result.meta or {})})


def _emit_progress(
    progress_callback: ProgressCallback | None,
    stage: str,
    **payload: Any,
) -> None:
    if progress_callback is None:
        return
    progress_callback(stage, payload)


def _load_cached_transcript_stage(
    artifacts: ArtifactStore,
    inp: PipelineInput,
    settings: Settings,
) -> TranscriptStage | None:
    if not artifacts.input_matches(inp):
        return None
    payload = artifacts.load_transcript_stage()
    if not isinstance(payload, dict):
        return None

    transcriber_name = _normalize_transcriber_name(settings)
    asr_model = _asr_model_name(settings, transcriber_name)
    if payload.get("transcriber") != transcriber_name or payload.get("asr_model") != asr_model:
        return None

    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        return None
    segments = [segment_from_dict(segment) for segment in raw_segments]
    return TranscriptStage(
        segments=segments,
        full_transcript=str(payload.get("full_transcript", "")),
        transcriber_name=transcriber_name,
        asr_model=asr_model,
    )


def _load_cached_window_stage(
    artifacts: ArtifactStore,
    inp: PipelineInput,
    settings: Settings,
    *,
    window_sec: float,
) -> WindowStage | None:
    if not artifacts.input_matches(inp):
        return None
    payload = artifacts.load_window_stage()
    if not isinstance(payload, dict):
        return None
    if float(payload.get("window_sec", -1.0)) != float(window_sec):
        return None
    if bool(payload.get("vision_enabled")) != bool(settings.ete_vision_enabled):
        return None

    raw_windows = payload.get("windows")
    if not isinstance(raw_windows, list):
        return None
    windows = [window_from_dict(window) for window in raw_windows]
    return WindowStage(
        windows=windows,
        speaker_aware=bool(payload.get("speaker_aware")),
        vision_enabled=bool(payload.get("vision_enabled")),
    )


def _window_batches(
    windows: list[TimeWindow], *, batch_size: int
) -> list[tuple[int, int, list[TimeWindow]]]:
    batches: list[tuple[int, int, list[TimeWindow]]] = []
    if batch_size <= 0:
        batch_size = len(windows) or 1
    for batch_index, start in enumerate(range(0, len(windows), batch_size)):
        end = min(start + batch_size, len(windows))
        batches.append((batch_index, start, end, windows[start:end]))
    return batches


def _plan_extraction_batches(
    windows: list[TimeWindow],
    settings: Settings,
) -> ExtractionBatchPlan:
    requested = max(int(settings.ete_extraction_batch_size or _DEFAULT_EXTRACTION_BATCH_SIZE), 1)
    max_batches_target = max(int(settings.ete_extraction_max_batches or 1), 1)
    max_batch_size = max(int(settings.ete_extraction_max_batch_size or requested), requested)
    total_windows = len(windows)

    if total_windows <= 0:
        return ExtractionBatchPlan(
            requested_batch_size=requested,
            effective_batch_size=requested,
            total_windows=0,
            total_batches=0,
            max_batches_target=max_batches_target,
            max_batch_size=max_batch_size,
        )

    effective = requested
    if total_windows > requested * max_batches_target:
        effective = max(requested, math.ceil(total_windows / max_batches_target))
    effective = min(max(effective, 1), max_batch_size)
    total_batches = math.ceil(total_windows / effective)

    return ExtractionBatchPlan(
        requested_batch_size=requested,
        effective_batch_size=effective,
        total_windows=total_windows,
        total_batches=total_batches,
        max_batches_target=max_batches_target,
        max_batch_size=max_batch_size,
    )


def _extract_timeline_batches(
    *,
    artifacts: ArtifactStore,
    synth: TimelineSynthesizer,
    windows: list[TimeWindow],
    dry_run: bool,
    batch_plan: ExtractionBatchPlan,
    progress_callback: ProgressCallback | None = None,
) -> tuple[TimelineResult, int]:
    combined_events = []
    combined_meta: dict = {}
    reused_batches = 0

    batches = _window_batches(windows, batch_size=batch_plan.effective_batch_size)
    total_batches = len(batches)
    for batch_index, start, end, batch_windows in batches:
        _emit_progress(
            progress_callback,
            "extracting",
            batch_index=batch_index,
            total_batches=total_batches,
            window_start=start,
            window_end=end,
            batch_size=batch_plan.effective_batch_size,
        )
        cached_payload = artifacts.load_batch_result(batch_index)
        cached_result: TimelineResult | None = None
        if (
            isinstance(cached_payload, dict)
            and cached_payload.get("dry_run") is dry_run
            and cached_payload.get("window_start") == start
            and cached_payload.get("window_end") == end
        ):
            raw_result = cached_payload.get("result")
            if isinstance(raw_result, dict):
                cached_result = timeline_result_from_dict(raw_result)

        if cached_result is not None:
            batch_result = cached_result
            reused_batches += 1
        else:
            batch_result = (
                synth.dry_run(batch_windows)
                if dry_run
                else synth.synthesize(batch_windows)
            )
            artifacts.write_batch_result(
                batch_index,
                dry_run=dry_run,
                window_start=start,
                window_end=end,
                result=batch_result,
            )

        combined_events.extend(batch_result.events)
        if batch_result.meta:
            combined_meta.update(batch_result.meta)

    return TimelineResult(events=combined_events, meta=combined_meta or None), reused_batches


def _build_meta(
    inp: PipelineInput,
    settings: Settings,
    media_stage: MediaStage,
    transcript_stage: TranscriptStage,
    window_stage: WindowStage,
    *,
    max_seconds: float | None,
) -> dict:
    meta: dict = {
        "input_source": media_stage.input_kind,
        "transcriber": transcript_stage.transcriber_name,
        "asr_backend": transcript_stage.transcriber_name,
        "asr_model": transcript_stage.asr_model,
        "diarization": (settings.ete_diarization or "none").lower().strip(),
        "word_timestamps": settings.ete_whisper_word_timestamps,
        "vision": window_stage.vision_enabled,
        "speaker_aware_chunking": window_stage.speaker_aware,
    }
    if inp.youtube_url:
        meta["source_url"] = inp.youtube_url
    if max_seconds is not None:
        meta["processed_seconds_cap"] = max_seconds
    return meta


def run_pipeline(
    inp: PipelineInput,
    *,
    work_dir: Path,
    settings: Settings | None = None,
    window_sec: float = 20.0,
    max_minutes: float | None = None,
    max_seconds: float | None = None,
    dry_run: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> TimelineResult:
    """Run the pipeline from source media to grounded timeline JSON."""
    settings = settings or load_settings()
    work_dir.mkdir(parents=True, exist_ok=True)
    artifacts = ArtifactStore(work_dir)
    prior_input_match = artifacts.input_matches(inp)
    artifacts.write_input_manifest(inp)

    _emit_progress(progress_callback, "preparing_media")
    media_stage = _prepare_media_stage(
        inp,
        work_dir,
        max_minutes=max_minutes,
        max_seconds=max_seconds,
    )
    artifacts.write_media_stage(media_stage)

    transcript_stage = _load_cached_transcript_stage(artifacts, inp, settings)
    reused_transcript = transcript_stage is not None
    if transcript_stage is None:
        _emit_progress(progress_callback, "transcribing", reused=False)
        transcript_stage = _run_transcription_stage(media_stage, settings)
        artifacts.write_transcript_stage(transcript_stage)
    else:
        _emit_progress(progress_callback, "transcribing", reused=True)

    window_stage = _load_cached_window_stage(
        artifacts,
        inp,
        settings,
        window_sec=window_sec,
    )
    reused_windows = window_stage is not None
    if window_stage is None:
        _emit_progress(progress_callback, "windowing", reused=False)
        window_stage = _run_window_stage(
            media_stage,
            transcript_stage,
            work_dir,
            settings,
            window_sec=window_sec,
        )
        artifacts.write_window_stage(window_stage, window_sec=window_sec)
    else:
        _emit_progress(progress_callback, "windowing", reused=True)

    synth = TimelineSynthesizer(settings)
    batch_plan = _plan_extraction_batches(window_stage.windows, settings)
    meta_base = _build_meta(
        inp,
        settings,
        media_stage,
        transcript_stage,
        window_stage,
        max_seconds=max_seconds,
    )
    meta_base["artifacts_dir"] = str(artifacts.root)
    meta_base["reused_artifacts"] = {
        "input_match": prior_input_match,
        "transcript": reused_transcript,
        "windows": reused_windows,
    }
    meta_base["batch_plan"] = {
        "requested_batch_size": batch_plan.requested_batch_size,
        "effective_batch_size": batch_plan.effective_batch_size,
        "total_windows": batch_plan.total_windows,
        "total_batches": batch_plan.total_batches,
        "max_batches_target": batch_plan.max_batches_target,
        "max_batch_size": batch_plan.max_batch_size,
    }
    artifacts.write_run_summary(meta_base)

    batched_result, reused_batches = _extract_timeline_batches(
        artifacts=artifacts,
        synth=synth,
        windows=window_stage.windows,
        dry_run=dry_run,
        batch_plan=batch_plan,
        progress_callback=progress_callback,
    )
    meta_base["extraction_batches"] = batch_plan.total_batches
    meta_base["reused_extraction_batches"] = reused_batches
    artifacts.write_run_summary(meta_base)

    result = batched_result
    result = _merge_meta(meta_base, result)
    if settings.ete_validate_evidence and not dry_run:
        _emit_progress(progress_callback, "validating")
        result = validate_timeline_evidence(
            result,
            transcript_stage.full_transcript,
            transcript_segments=transcript_stage.segments,
        )
    result = merge_adjacent_duplicate_events(result)
    result = filter_low_signal_events(result)
    artifacts.write_timeline_result(result)
    _emit_progress(progress_callback, "completed", events_count=len(result.events))
    return result
