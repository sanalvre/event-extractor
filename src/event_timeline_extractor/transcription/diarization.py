"""Optional speaker assignment by overlap with diarization intervals."""

from __future__ import annotations

import logging
from pathlib import Path

from event_timeline_extractor.config import Settings
from event_timeline_extractor.transcription.base import TranscriptSegment

logger = logging.getLogger(__name__)


def assign_speakers_by_overlap(
    segments: list[TranscriptSegment],
    speaker_intervals: list[tuple[float, float, str]],
) -> list[TranscriptSegment]:
    """Assign each segment the label of the interval with largest time overlap."""
    out: list[TranscriptSegment] = []
    for seg in segments:
        best_label: str | None = None
        best_overlap = 0.0
        for s, e, label in speaker_intervals:
            overlap = max(0.0, min(seg.end, e) - max(seg.start, s))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label
        out.append(
            TranscriptSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                speaker=best_label,
            )
        )
    return out


def maybe_apply_diarization(
    settings: Settings,
    wav_path: Path,
    segments: list[TranscriptSegment],
) -> list[TranscriptSegment]:
    mode = (settings.ete_diarization or "none").lower().strip()
    if mode == "none":
        return segments
    if mode == "pyannote":
        from event_timeline_extractor.transcription.diarize_pyannote import (
            apply_pyannote_speakers,
        )

        return apply_pyannote_speakers(wav_path, segments, settings)
    raise ValueError(f"Unknown ETE_DIARIZATION: {settings.ete_diarization!r}")
