from event_timeline_extractor.postprocess import (
    filter_low_signal_events,
    merge_adjacent_duplicate_events,
)
from event_timeline_extractor.schema import TimelineEvent, TimelineResult


def test_merge_adjacent_duplicate_events_merges_overlapping_grounding() -> None:
    result = TimelineResult(
        events=[
            TimelineEvent(
                time="00:10",
                event="Host introduces the interview topic.",
                event_type="speech",
                confidence=0.62,
                speaker="Host",
                evidence="Let's talk about the product roadmap.",
                source_segment_ids=["seg-000010", "seg-000011"],
                source_start="00:10",
                source_end="00:14",
            ),
            TimelineEvent(
                time="00:12",
                event="Host introduces the interview topic",
                event_type="speech",
                confidence=0.84,
                speaker="Host",
                evidence="Let's talk about the product roadmap.",
                source_segment_ids=["seg-000011", "seg-000012"],
                source_start="00:12",
                source_end="00:16",
            ),
        ],
        meta={},
    )

    out = merge_adjacent_duplicate_events(result)

    assert len(out.events) == 1
    assert out.events[0].event_type == "speech"
    assert out.events[0].confidence == 0.84
    assert out.events[0].source_segment_ids == ["seg-000010", "seg-000011", "seg-000012"]
    assert out.events[0].source_start == "00:10"
    assert out.events[0].source_end == "00:16"
    assert out.meta is not None
    assert out.meta["postprocess"] == {"merged_duplicate_events": 1}


def test_merge_adjacent_duplicate_events_keeps_separate_distinct_moments() -> None:
    result = TimelineResult(
        events=[
            TimelineEvent(
                time="00:10",
                event="Crowd applauds loudly.",
                event_type="action",
                speaker=None,
                evidence="The audience breaks into applause.",
                source_segment_ids=["seg-000010"],
                source_start="00:10",
                source_end="00:11",
            ),
            TimelineEvent(
                time="03:40",
                event="Crowd applauds loudly.",
                event_type="action",
                speaker=None,
                evidence="The audience breaks into applause.",
                source_segment_ids=["seg-000220"],
                source_start="03:40",
                source_end="03:41",
            ),
        ],
        meta={},
    )

    out = merge_adjacent_duplicate_events(result)

    assert len(out.events) == 2
    assert out.meta is not None
    assert out.meta["postprocess"] == {"merged_duplicate_events": 0}


def test_merge_adjacent_duplicate_events_requires_matching_speaker() -> None:
    result = TimelineResult(
        events=[
            TimelineEvent(
                time="00:10",
                event="Question about launch timing.",
                event_type="speech",
                speaker="Host",
                evidence="When does it ship?",
                source_segment_ids=["seg-000010"],
                source_start="00:10",
                source_end="00:12",
            ),
            TimelineEvent(
                time="00:12",
                event="Question about launch timing.",
                event_type="speech",
                speaker="Guest",
                evidence="When does it ship?",
                source_segment_ids=["seg-000011"],
                source_start="00:12",
                source_end="00:14",
            ),
        ],
        meta={},
    )

    out = merge_adjacent_duplicate_events(result)

    assert len(out.events) == 2
    assert out.meta is not None
    assert out.meta["postprocess"] == {"merged_duplicate_events": 0}


def test_merge_adjacent_duplicate_events_requires_matching_event_type() -> None:
    result = TimelineResult(
        events=[
            TimelineEvent(
                time="00:10",
                event="Team agrees on the launch date.",
                event_type="decision",
                speaker="Host",
                evidence="Let's ship next Friday.",
                source_segment_ids=["seg-000010"],
                source_start="00:10",
                source_end="00:12",
            ),
            TimelineEvent(
                time="00:11",
                event="Team agrees on the launch date.",
                event_type="speech",
                speaker="Host",
                evidence="Let's ship next Friday.",
                source_segment_ids=["seg-000011"],
                source_start="00:11",
                source_end="00:13",
            ),
        ],
        meta={},
    )

    out = merge_adjacent_duplicate_events(result)

    assert len(out.events) == 2
    assert out.meta is not None
    assert out.meta["postprocess"] == {"merged_duplicate_events": 0}


def test_filter_low_signal_events_drops_brief_acknowledgements() -> None:
    result = TimelineResult(
        events=[
            TimelineEvent(
                time="00:10",
                event="Right.",
                event_type="speech",
                confidence=1.0,
                evidence="Right.",
                source_segment_ids=["seg-000010"],
                source_start="00:10",
                source_end="00:10",
            ),
            TimelineEvent(
                time="00:11",
                event="Host explains the new finding.",
                event_type="speech",
                confidence=0.9,
                evidence="We discovered a new chamber under the site.",
                source_segment_ids=["seg-000011"],
                source_start="00:11",
                source_end="00:11",
            ),
        ],
        meta={},
    )

    out = filter_low_signal_events(result)

    assert len(out.events) == 1
    assert out.events[0].event == "Host explains the new finding."
    assert out.meta is not None
    assert out.meta["postprocess"] == {"filtered_low_signal_events": 1}


def test_filter_low_signal_events_keeps_non_speech_events() -> None:
    result = TimelineResult(
        events=[
            TimelineEvent(
                time="00:10",
                event="Camera cuts to excavation site.",
                event_type="transition",
                confidence=0.7,
                evidence=None,
                source_segment_ids=["seg-000010"],
                source_start="00:10",
                source_end="00:10",
            )
        ],
        meta={},
    )

    out = filter_low_signal_events(result)

    assert len(out.events) == 1
    assert out.meta is not None
    assert out.meta["postprocess"] == {"filtered_low_signal_events": 0}
