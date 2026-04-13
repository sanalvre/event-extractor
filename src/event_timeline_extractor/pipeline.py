"""End-to-end: resolve input → wav → transcript → windows → timeline JSON."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from event_timeline_extractor.chunking import chunk_segments
from event_timeline_extractor.config import Settings, load_settings
from event_timeline_extractor.fetch import run_ytdlp_download
from event_timeline_extractor.ffmpeg_tools import (
    extract_audio_wav_16k_mono,
    extract_frames_every_interval,
    probe_duration_seconds,
)
from event_timeline_extractor.llm.openrouter import TimelineSynthesizer
from event_timeline_extractor.schema import TimelineResult
from event_timeline_extractor.timefmt import download_section_first_seconds
from event_timeline_extractor.transcription.diarization import maybe_apply_diarization
from event_timeline_extractor.transcription.factory import get_transcriber
from event_timeline_extractor.validation import validate_timeline_evidence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineInput:
    youtube_url: str | None = None
    file_path: Path | None = None


def resolve_media_path(
    inp: PipelineInput,
    work_dir: Path,
    *,
    max_seconds: float | None = None,
) -> Path:
    if inp.file_path is not None:
        p = inp.file_path.resolve()
        if not p.is_file():
            raise FileNotFoundError(str(p))
        return p
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


def _run_vision_analysis(
    media: Path,
    work_dir: Path,
    settings: Settings,
    duration: float,
) -> dict[float, str]:
    """Extract frames and run memories-s0 vision analysis. Returns {timestamp → description}."""
    from event_timeline_extractor.vision.memories_s0 import MemoriesS0Analyzer  # noqa: PLC0415

    interval = float(settings.ete_vision_frame_interval)
    frames_dir = work_dir / "frames"
    logger.info("Extracting frames every %gs for visual analysis…", interval)
    frame_paths = extract_frames_every_interval(media, frames_dir, interval_sec=interval)
    if not frame_paths:
        logger.warning("No frames extracted — skipping vision analysis.")
        return {}

    # Derive timestamps: frame_000001.jpg → t=0, frame_000002.jpg → t=interval, etc.
    timestamps = [(i * interval) for i in range(len(frame_paths))]
    # Clamp last timestamp to duration so it doesn't overshoot.
    timestamps = [min(t, duration) for t in timestamps]

    logger.info("Running memories-s0 on %d frames…", len(frame_paths))
    analyzer = MemoriesS0Analyzer()
    descriptions = analyzer.analyze([str(p) for p in frame_paths], timestamps)
    return {d.timestamp: d.description for d in descriptions}


def _merge_meta(base: dict, result: TimelineResult) -> TimelineResult:
    m = dict(result.meta or {})
    m = {**base, **m}
    return TimelineResult(events=result.events, meta=m)


def run_pipeline(
    inp: PipelineInput,
    *,
    work_dir: Path,
    settings: Settings | None = None,
    window_sec: float = 20.0,
    max_minutes: float | None = None,
    max_seconds: float | None = None,
    dry_run: bool = False,
) -> TimelineResult:
    """Transcribe real audio (faster-whisper) unless ETE_USE_STUB=1 or transcriber=stub.

    ``max_seconds``: only process the first N seconds — YouTube uses yt-dlp
    ``--download-sections`` (smaller download); local files use ffmpeg ``-t`` on WAV extract.
    """
    settings = settings or load_settings()
    work_dir.mkdir(parents=True, exist_ok=True)
    media = resolve_media_path(inp, work_dir, max_seconds=max_seconds)

    duration = probe_duration_seconds(media)
    if max_seconds is not None:
        duration = min(duration, float(max_seconds))
    if max_minutes is not None and duration > max_minutes * 60:
        raise ValueError(
            f"Media duration {duration:.0f}s exceeds --max-minutes={max_minutes}."
        )

    wav = work_dir / "audio_16k.wav"
    audio_cap = max_seconds if max_seconds is not None and max_seconds > 0 else None
    extract_audio_wav_16k_mono(media, wav, max_duration_sec=audio_cap)

    transcriber = get_transcriber(settings)
    segments = transcriber.transcribe(wav)
    segments = maybe_apply_diarization(settings, wav, segments)
    full_transcript = " ".join(s.text for s in segments)

    vision_map: dict[float, str] = {}
    if settings.ete_vision_enabled:
        vision_map = _run_vision_analysis(media, work_dir, settings, duration)

    # Use speaker-aware chunking when the memories.ai transcriber provides diarization,
    # so windows break at natural speaker turns rather than fixed time boundaries.
    speaker_aware = (
        settings.ete_transcriber == "memories"
        and settings.memories_transcription_speaker
        and any(s.speaker for s in segments)
    )
    windows = chunk_segments(
        segments,
        window_sec=window_sec,
        vision_map=vision_map or None,
        speaker_aware=speaker_aware,
    )

    synth = TimelineSynthesizer(settings)
    meta_base: dict = {
        "asr_model": settings.ete_whisper_model_size,
        "transcriber": settings.ete_transcriber,
        "diarization": (settings.ete_diarization or "none").lower().strip(),
        "word_timestamps": settings.ete_whisper_word_timestamps,
        "vision": settings.ete_vision_enabled,
        "speaker_aware_chunking": speaker_aware,
    }
    if max_seconds is not None:
        meta_base["processed_seconds_cap"] = max_seconds

    if dry_run:
        return _merge_meta(meta_base, synth.dry_run(windows))
    result = synth.synthesize(windows)
    result = _merge_meta(meta_base, result)
    if settings.ete_validate_evidence:
        result = validate_timeline_evidence(result, full_transcript)
    return result
