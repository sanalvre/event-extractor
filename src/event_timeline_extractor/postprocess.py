"""Timeline post-processing helpers."""

from __future__ import annotations

import re

from event_timeline_extractor.schema import TimelineEvent, TimelineResult

_WORD_RE = re.compile(r"[a-z0-9]+")
_SEGMENT_ID_RE = re.compile(r"(\d+)$")
_LOW_SIGNAL_PHRASES = {
    "yes",
    "yeah",
    "yep",
    "okay",
    "ok",
    "right",
    "really",
    "oh",
    "oh well",
    "please do",
    "of course",
    "excuse me",
    "boy",
    "good",
    "no",
    "nope",
}


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(_WORD_RE.findall(value.lower()))


def _normalize_speaker(value: str | None) -> str:
    return " ".join((value or "").strip().lower().split())


def _parse_mmss(value: str | None) -> int | None:
    if not value or ":" not in value:
        return None
    minutes_text, seconds_text = value.split(":", 1)
    if not minutes_text.isdigit() or not seconds_text.isdigit():
        return None
    return int(minutes_text) * 60 + int(seconds_text)


def _format_mmss(seconds: int | None) -> str | None:
    if seconds is None:
        return None
    if seconds < 0:
        seconds = 0
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes:02d}:{secs:02d}"


def _segment_ordinals(segment_ids: list[str] | None) -> list[int]:
    ordinals: list[int] = []
    for segment_id in segment_ids or []:
        match = _SEGMENT_ID_RE.search(segment_id)
        if match:
            ordinals.append(int(match.group(1)))
    return ordinals


def _segments_are_adjacent_or_overlapping(
    left_ids: list[str] | None,
    right_ids: list[str] | None,
) -> bool:
    left_set = set(left_ids or [])
    right_set = set(right_ids or [])
    if left_set and right_set and left_set.intersection(right_set):
        return True

    left_ordinals = _segment_ordinals(left_ids)
    right_ordinals = _segment_ordinals(right_ids)
    if left_ordinals and right_ordinals:
        return abs(min(right_ordinals) - max(left_ordinals)) <= 1 or abs(
            min(left_ordinals) - max(right_ordinals)
        ) <= 1
    return False


def _times_are_close_or_overlapping(left: TimelineEvent, right: TimelineEvent) -> bool:
    left_time = _parse_mmss(left.time)
    right_time = _parse_mmss(right.time)
    if left_time is not None and right_time is not None and abs(left_time - right_time) <= 5:
        return True

    left_start = _parse_mmss(left.source_start)
    left_end = _parse_mmss(left.source_end)
    right_start = _parse_mmss(right.source_start)
    right_end = _parse_mmss(right.source_end)
    if None not in (left_start, left_end, right_start, right_end):
        assert left_start is not None
        assert left_end is not None
        assert right_start is not None
        assert right_end is not None
        return left_start <= right_end + 2 and right_start <= left_end + 2
    return False


def _evidence_is_compatible(left: TimelineEvent, right: TimelineEvent) -> bool:
    left_evidence = _normalize_text(left.evidence)
    right_evidence = _normalize_text(right.evidence)
    if not left_evidence or not right_evidence:
        return True
    return (
        left_evidence == right_evidence
        or left_evidence in right_evidence
        or right_evidence in left_evidence
    )


def _should_merge_events(left: TimelineEvent, right: TimelineEvent) -> bool:
    if _normalize_text(left.event) != _normalize_text(right.event):
        return False
    if _normalize_speaker(left.speaker) != _normalize_speaker(right.speaker):
        return False
    if left.event_type != right.event_type:
        return False
    if not _evidence_is_compatible(left, right):
        return False

    return _segments_are_adjacent_or_overlapping(
        left.source_segment_ids,
        right.source_segment_ids,
    ) or _times_are_close_or_overlapping(left, right)


def _merged_segment_ids(left: TimelineEvent, right: TimelineEvent) -> list[str] | None:
    merged: list[str] = []
    seen: set[str] = set()
    for segment_id in [*(left.source_segment_ids or []), *(right.source_segment_ids or [])]:
        if segment_id and segment_id not in seen:
            merged.append(segment_id)
            seen.add(segment_id)
    return merged or None


def _pick_evidence(left: TimelineEvent, right: TimelineEvent) -> str | None:
    left_evidence = (left.evidence or "").strip()
    right_evidence = (right.evidence or "").strip()
    if not left_evidence:
        return right.evidence
    if not right_evidence:
        return left.evidence
    if (
        len(right_evidence) > len(left_evidence)
        and _normalize_text(left_evidence) in _normalize_text(right_evidence)
    ):
        return right.evidence
    return left.evidence


def _is_low_signal_speech(event: TimelineEvent) -> bool:
    if event.event_type != "speech":
        return False
    normalized_event = _normalize_text(event.event)
    normalized_evidence = _normalize_text(event.evidence)
    candidate = normalized_evidence or normalized_event
    if not candidate:
        return False
    words = candidate.split()
    if len(words) > 3:
        return False
    if candidate in _LOW_SIGNAL_PHRASES:
        return True
    if len(words) == 1 and words[0] in _LOW_SIGNAL_PHRASES:
        return True
    return False


def merge_adjacent_duplicate_events(result: TimelineResult) -> TimelineResult:
    """Collapse repeated adjacent events conservatively.

    We only merge neighboring events when they share the same normalized event text and
    speaker, and their transcript grounding is overlapping or immediately adjacent.
    """
    if not result.events:
        meta = dict(result.meta or {})
        meta["postprocess"] = {"merged_duplicate_events": 0}
        return TimelineResult(events=[], meta=meta)

    merged_events: list[TimelineEvent] = []
    merged_count = 0

    for event in result.events:
        current = TimelineEvent.model_validate(event)
        if not merged_events:
            merged_events.append(current)
            continue

        previous = merged_events[-1]
        if not _should_merge_events(previous, current):
            merged_events.append(current)
            continue

        merged_count += 1
        all_times = [
            value
            for value in (
                _parse_mmss(previous.time),
                _parse_mmss(current.time),
            )
            if value is not None
        ]
        merged_ids = _merged_segment_ids(previous, current)
        range_starts = [
            value
            for value in (
                _parse_mmss(previous.source_start),
                _parse_mmss(current.source_start),
            )
            if value is not None
        ]
        range_ends = [
            value
            for value in (
                _parse_mmss(previous.source_end),
                _parse_mmss(current.source_end),
            )
            if value is not None
        ]

        merged_events[-1] = TimelineEvent(
            time=_format_mmss(min(all_times)) if all_times else previous.time,
            event=previous.event if len(previous.event) >= len(current.event) else current.event,
            event_type=previous.event_type,
            confidence=max(
                value
                for value in (
                    previous.confidence if previous.confidence is not None else 0.0,
                    current.confidence if current.confidence is not None else 0.0,
                )
            )
            if previous.confidence is not None or current.confidence is not None
            else None,
            speaker=previous.speaker if previous.speaker is not None else current.speaker,
            evidence=_pick_evidence(previous, current),
            source_segment_ids=merged_ids,
            source_start=_format_mmss(min(range_starts)) if range_starts else previous.source_start,
            source_end=_format_mmss(max(range_ends)) if range_ends else previous.source_end,
        )

    meta = dict(result.meta or {})
    postprocess = dict(meta.get("postprocess") or {})
    postprocess["merged_duplicate_events"] = merged_count
    meta["postprocess"] = postprocess
    return TimelineResult(events=merged_events, meta=meta)


def filter_low_signal_events(result: TimelineResult) -> TimelineResult:
    """Drop obvious low-signal speech acknowledgements from the final timeline."""
    filtered_events: list[TimelineEvent] = []
    dropped = 0
    for event in result.events:
        current = TimelineEvent.model_validate(event)
        if _is_low_signal_speech(current):
            dropped += 1
            continue
        filtered_events.append(current)

    meta = dict(result.meta or {})
    postprocess = dict(meta.get("postprocess") or {})
    postprocess["filtered_low_signal_events"] = dropped
    meta["postprocess"] = postprocess
    return TimelineResult(events=filtered_events, meta=meta)
