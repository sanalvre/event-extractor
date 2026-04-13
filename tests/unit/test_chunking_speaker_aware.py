"""Tests for speaker_aware chunking in chunk_segments()."""

from __future__ import annotations

from event_timeline_extractor.chunking import TimeWindow, chunk_segments
from event_timeline_extractor.transcription.base import TranscriptSegment


def _seg(start: float, end: float, text: str, speaker: str | None = None) -> TranscriptSegment:
    return TranscriptSegment(start=start, end=end, text=text, speaker=speaker)


# ---------------------------------------------------------------------------
# speaker_aware=False (default) — existing behaviour unchanged
# ---------------------------------------------------------------------------


def test_speaker_aware_false_does_not_break_on_speaker_change() -> None:
    segs = [
        _seg(0.0, 5.0, "Hello", "SPEAKER_00"),
        _seg(5.0, 10.0, "Hi there", "SPEAKER_01"),  # different speaker
    ]
    windows = chunk_segments(segs, window_sec=20.0, speaker_aware=False)
    assert len(windows) == 1, "Should NOT split on speaker change when speaker_aware=False"


# ---------------------------------------------------------------------------
# speaker_aware=True — breaks at speaker turns
# ---------------------------------------------------------------------------


def test_speaker_aware_splits_on_speaker_change() -> None:
    segs = [
        _seg(0.0, 3.0, "Officer: Stop!", "SPEAKER_00"),
        _seg(3.0, 6.0, "Driver: Okay.", "SPEAKER_01"),
        _seg(6.0, 9.0, "Officer: Exit.", "SPEAKER_00"),
    ]
    windows = chunk_segments(segs, window_sec=30.0, speaker_aware=True)
    assert len(windows) == 3, "Each speaker turn should be its own window"
    assert "Officer: Stop!" in windows[0].text
    assert "Driver: Okay." in windows[1].text
    assert "Officer: Exit." in windows[2].text


def test_speaker_aware_merges_same_speaker_consecutive_segs() -> None:
    segs = [
        _seg(0.0, 3.0, "First line.", "SPEAKER_00"),
        _seg(3.0, 6.0, "Second line.", "SPEAKER_00"),  # same speaker — no split
        _seg(6.0, 9.0, "Response.", "SPEAKER_01"),
    ]
    windows = chunk_segments(segs, window_sec=30.0, speaker_aware=True)
    assert len(windows) == 2
    assert "First line." in windows[0].text
    assert "Second line." in windows[0].text


def test_speaker_aware_still_breaks_on_time_limit() -> None:
    """Long monologues split at window_sec even if speaker doesn't change."""
    segs = [_seg(float(i), float(i + 1), f"Word {i}.", "SPEAKER_00") for i in range(30)]
    windows = chunk_segments(segs, window_sec=10.0, speaker_aware=True)
    assert len(windows) > 1, "Long same-speaker segment must still split by time"


def test_speaker_aware_graceful_when_no_speakers_in_segments() -> None:
    """Falls back to time-based chunking when speaker labels are absent."""
    segs = [_seg(0.0, 5.0, "Hello"), _seg(5.0, 10.0, "World")]
    windows = chunk_segments(segs, window_sec=20.0, speaker_aware=True)
    assert len(windows) == 1  # no speaker info → no speaker-based splits


def test_speaker_aware_ignores_none_speaker_for_break_decision() -> None:
    """A None speaker segment should not trigger a break."""
    segs = [
        _seg(0.0, 3.0, "Known speaker.", "SPEAKER_00"),
        _seg(3.0, 6.0, "Unknown speaker.", None),   # no speaker label → no break
        _seg(6.0, 9.0, "Known again.", "SPEAKER_00"),
    ]
    windows = chunk_segments(segs, window_sec=30.0, speaker_aware=True)
    assert len(windows) == 1


# ---------------------------------------------------------------------------
# vision_context preserved through speaker-aware chunking
# ---------------------------------------------------------------------------


def test_speaker_aware_preserves_vision_context() -> None:
    segs = [
        _seg(0.0, 5.0, "Officer speaks.", "SPEAKER_00"),
        _seg(5.0, 10.0, "Driver responds.", "SPEAKER_01"),
    ]
    vision_map = {0.0: "Officer at window.", 5.0: "Driver looks up."}
    windows = chunk_segments(segs, window_sec=30.0, speaker_aware=True, vision_map=vision_map)
    assert len(windows) == 2
    assert "Officer at window." in windows[0].vision_context
    assert "Driver looks up." in windows[1].vision_context
