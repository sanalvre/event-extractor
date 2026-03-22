"""Optional pyannote.audio diarization: assign speaker labels to ASR segments."""

from __future__ import annotations

import logging
from pathlib import Path

from event_timeline_extractor.config import Settings
from event_timeline_extractor.transcription.base import TranscriptSegment
from event_timeline_extractor.transcription.diarization import assign_speakers_by_overlap

logger = logging.getLogger(__name__)

_PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"


def _annotation_to_intervals(annotation: object) -> list[tuple[float, float, str]]:
    intervals: list[tuple[float, float, str]] = []
    for segment, _track, label in annotation.itertracks(yield_label=True):  # type: ignore[attr-defined]
        intervals.append((float(segment.start), float(segment.end), str(label)))
    return intervals


def apply_pyannote_speakers(
    wav_path: Path,
    segments: list[TranscriptSegment],
    settings: Settings,
) -> list[TranscriptSegment]:
    token = settings.hf_token_plain()
    if not token:
        raise ValueError(
            "ETE_DIARIZATION=pyannote requires HF_TOKEN (Hugging Face) for pyannote models."
        )
    try:
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise ImportError(
            'Install diarization extras: pip install -e ".[diarize]"'
        ) from e

    pipeline = Pipeline.from_pretrained(_PYANNOTE_MODEL, use_auth_token=token)
    diarization = pipeline(str(wav_path))
    intervals = _annotation_to_intervals(diarization)
    if not intervals:
        logger.warning("pyannote returned no speech turns; leaving speaker unset.")
        return segments
    return assign_speakers_by_overlap(segments, intervals)
