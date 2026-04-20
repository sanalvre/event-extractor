"""Post-validate timeline output against the transcript and source references."""

from __future__ import annotations

from event_timeline_extractor.schema import TimelineEvent, TimelineResult
from event_timeline_extractor.transcription.base import TranscriptSegment


def _normalize_ws(s: str) -> str:
    return " ".join(s.split())


def _format_mmss(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def evidence_in_transcript(evidence: str, full_transcript: str) -> bool:
    ev = _normalize_ws(evidence)
    if not ev:
        return True
    return ev in _normalize_ws(full_transcript)


def _validate_source_references(
    event: TimelineEvent,
    segment_map: dict[str, TranscriptSegment],
) -> str | None:
    source_ids = list(event.source_segment_ids or [])
    if not source_ids:
        return None

    missing = [seg_id for seg_id in source_ids if seg_id not in segment_map]
    if missing:
        return f"Unknown source segment IDs: {', '.join(missing)}"

    segments = [segment_map[seg_id] for seg_id in source_ids]
    expected_start = _format_mmss(min(segment.start for segment in segments))
    # Source references are line-anchored in the prompt, so source_end should match the
    # latest supporting segment timestamp rather than the segment's internal end offset.
    expected_end = _format_mmss(max(segment.start for segment in segments))

    if event.source_start is not None and event.source_start != expected_start:
        return f"source_start mismatch: expected {expected_start!r}"
    if event.source_end is not None and event.source_end != expected_end:
        return f"source_end mismatch: expected {expected_end!r}"
    return None


def validate_timeline_evidence(
    result: TimelineResult,
    full_transcript: str,
    *,
    transcript_segments: list[TranscriptSegment] | None = None,
) -> TimelineResult:
    """Drop events with invalid evidence or invalid source references.

    Evidence uses whitespace-normalized substring matching against the full transcript.
    Source references, when present, must refer to known segment IDs and their source_start /
    source_end values must align with the referenced segment timestamps.
    """
    kept: list[TimelineEvent] = []
    dropped = 0
    warnings: list[str] = []
    segment_map = {
        str(segment.segment_id): segment
        for segment in (transcript_segments or [])
        if segment.segment_id
    }

    for event in result.events:
        if event.evidence is not None and str(event.evidence).strip():
            if not evidence_in_transcript(str(event.evidence), full_transcript):
                dropped += 1
                warnings.append(
                    f"Evidence not found in transcript (dropped): time={event.time!r} "
                    f"event={event.event!r}"
                )
                continue

        source_error = _validate_source_references(event, segment_map)
        if source_error is not None:
            dropped += 1
            warnings.append(
                f"Invalid source references (dropped): time={event.time!r} "
                f"event={event.event!r} reason={source_error}"
            )
            continue

        kept.append(event)

    meta = dict(result.meta or {})
    meta["validation"] = {"dropped_events": dropped}
    if warnings:
        existing = meta.get("warnings")
        if isinstance(existing, list):
            meta["warnings"] = [*existing, *warnings]
        else:
            meta["warnings"] = warnings

    return TimelineResult(events=kept, meta=meta)
