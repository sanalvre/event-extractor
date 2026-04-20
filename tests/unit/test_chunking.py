from event_timeline_extractor.chunking import (
    attach_frames_to_timeline,
    chunk_segments,
    format_mmss,
    format_segment_line,
)
from event_timeline_extractor.transcription.base import TranscriptSegment


def test_format_mmss() -> None:
    assert format_mmss(0) == "00:00"
    assert format_mmss(65.4) == "01:05"


def test_chunk_segments_empty() -> None:
    assert chunk_segments([]) == []


def test_format_segment_line_speaker() -> None:
    line = format_segment_line(
        TranscriptSegment(65.0, 70.0, "hello", speaker="SPEAKER_00", segment_id="seg-000001")
    )
    assert line == "[01:05] seg-000001 SPEAKER_00: hello"


def test_chunk_segments_merges() -> None:
    segs = [
        TranscriptSegment(0, 5, "a", segment_id="seg-000001"),
        TranscriptSegment(5, 10, "b", segment_id="seg-000002"),
    ]
    w = chunk_segments(segs, window_sec=100)
    assert len(w) == 1
    assert "[00:00] seg-000001 a" in w[0].text and "[00:05] seg-000002 b" in w[0].text
    assert w[0].source_segment_ids == ["seg-000001", "seg-000002"]


def test_attach_frames() -> None:
    m = attach_frames_to_timeline(["a.jpg", "b.jpg"], 10.0)
    assert len(m) == 2


def test_chunk_respects_window() -> None:
    segs = [
        TranscriptSegment(0, 2, "one"),
        TranscriptSegment(25, 27, "two"),
    ]
    w = chunk_segments(segs, window_sec=10)
    assert len(w) >= 2
