from event_timeline_extractor.schema import TimelineEvent, TimelineResult
from event_timeline_extractor.transcription.base import TranscriptSegment
from event_timeline_extractor.validation import evidence_in_transcript, validate_timeline_evidence


def test_evidence_in_transcript() -> None:
    assert evidence_in_transcript("hello world", "  hello   world  ")
    assert not evidence_in_transcript("nope", "hello world")


def test_validate_drops_bad_evidence() -> None:
    result = TimelineResult(
        events=[
            TimelineEvent(time="00:01", event="A", speaker=None, evidence="keep me"),
            TimelineEvent(time="00:02", event="B", speaker=None, evidence="not in transcript"),
        ],
        meta={"model": "x"},
    )
    out = validate_timeline_evidence(result, "keep me and more text")
    assert len(out.events) == 1
    assert out.events[0].event == "A"
    assert out.meta and out.meta.get("validation") == {"dropped_events": 1}
    assert out.meta.get("warnings")


def test_validate_keeps_null_evidence() -> None:
    result = TimelineResult(
        events=[TimelineEvent(time="00:00", event="X", speaker=None, evidence=None)],
        meta=None,
    )
    out = validate_timeline_evidence(result, "")
    assert len(out.events) == 1


def test_validate_keeps_valid_source_references() -> None:
    segments = [
        TranscriptSegment(0.0, 2.0, "hello", segment_id="seg-000001"),
        TranscriptSegment(2.0, 4.0, "world", segment_id="seg-000002"),
    ]
    result = TimelineResult(
        events=[
            TimelineEvent(
                time="00:00",
                event="Greeting",
                evidence="hello",
                source_segment_ids=["seg-000001", "seg-000002"],
                source_start="00:00",
                source_end="00:02",
            )
        ],
        meta={},
    )
    out = validate_timeline_evidence(result, "hello world", transcript_segments=segments)
    assert len(out.events) == 1
    assert out.meta and out.meta["validation"] == {"dropped_events": 0}


def test_validate_drops_unknown_source_segment_ids() -> None:
    segments = [TranscriptSegment(0.0, 2.0, "hello", segment_id="seg-000001")]
    result = TimelineResult(
        events=[
            TimelineEvent(
                time="00:00",
                event="Greeting",
                evidence="hello",
                source_segment_ids=["seg-999999"],
                source_start="00:00",
                source_end="00:02",
            )
        ],
        meta={},
    )
    out = validate_timeline_evidence(result, "hello", transcript_segments=segments)
    assert len(out.events) == 0
    assert out.meta and out.meta["validation"] == {"dropped_events": 1}
    assert "Invalid source references" in out.meta["warnings"][0]


def test_validate_drops_source_timestamp_mismatch() -> None:
    segments = [TranscriptSegment(0.0, 2.0, "hello", segment_id="seg-000001")]
    result = TimelineResult(
        events=[
            TimelineEvent(
                time="00:00",
                event="Greeting",
                evidence="hello",
                source_segment_ids=["seg-000001"],
                source_start="00:01",
                source_end="00:02",
            )
        ],
        meta={},
    )
    out = validate_timeline_evidence(result, "hello", transcript_segments=segments)
    assert len(out.events) == 0
    assert out.meta and out.meta["validation"] == {"dropped_events": 1}
