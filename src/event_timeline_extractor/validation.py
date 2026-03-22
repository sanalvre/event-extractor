"""Post-validate LLM timeline output against the full transcript."""

from __future__ import annotations

from event_timeline_extractor.schema import TimelineEvent, TimelineResult


def _normalize_ws(s: str) -> str:
    return " ".join(s.split())


def evidence_in_transcript(evidence: str, full_transcript: str) -> bool:
    ev = _normalize_ws(evidence)
    if not ev:
        return True
    return ev in _normalize_ws(full_transcript)


def validate_timeline_evidence(
    result: TimelineResult,
    full_transcript: str,
) -> TimelineResult:
    """Drop events whose ``evidence`` is present but not a substring of ``full_transcript``.

    Whitespace is normalized for the substring check. Records counts and warnings in ``meta``.
    """
    kept: list[TimelineEvent] = []
    dropped = 0
    warnings: list[str] = []

    for e in result.events:
        if e.evidence is None or not str(e.evidence).strip():
            kept.append(e)
            continue
        if evidence_in_transcript(str(e.evidence), full_transcript):
            kept.append(e)
        else:
            dropped += 1
            warnings.append(
                f"Evidence not found in transcript (dropped): time={e.time!r} "
                f"event={e.event!r}"
            )

    meta = dict(result.meta or {})
    meta["validation"] = {"dropped_events": dropped}
    if warnings:
        existing = meta.get("warnings")
        if isinstance(existing, list):
            meta["warnings"] = [*existing, *warnings]
        else:
            meta["warnings"] = warnings

    return TimelineResult(events=kept, meta=meta)
