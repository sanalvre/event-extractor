from event_timeline_extractor.schema import TimelineEvent, TimelineResult
from event_timeline_extractor.validation import evidence_in_transcript, validate_timeline_evidence


def test_evidence_in_transcript() -> None:
    assert evidence_in_transcript("hello world", "  hello   world  ")
    assert not evidence_in_transcript("nope", "hello world")


def test_validate_drops_bad_evidence() -> None:
    r = TimelineResult(
        events=[
            TimelineEvent(time="00:01", event="A", speaker=None, evidence="keep me"),
            TimelineEvent(time="00:02", event="B", speaker=None, evidence="not in transcript"),
        ],
        meta={"model": "x"},
    )
    out = validate_timeline_evidence(r, "keep me and more text")
    assert len(out.events) == 1
    assert out.events[0].event == "A"
    assert out.meta and out.meta.get("validation") == {"dropped_events": 1}
    assert out.meta.get("warnings")


def test_validate_keeps_null_evidence() -> None:
    r = TimelineResult(
        events=[TimelineEvent(time="00:00", event="X", speaker=None, evidence=None)],
        meta=None,
    )
    out = validate_timeline_evidence(r, "")
    assert len(out.events) == 1
