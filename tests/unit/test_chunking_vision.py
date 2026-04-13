"""Tests for vision_map / vision_context additions to chunking."""

from __future__ import annotations

from event_timeline_extractor.chunking import (
    TimeWindow,
    _vision_context_in_range,
    chunk_segments,
)
from event_timeline_extractor.transcription.base import TranscriptSegment


def _seg(start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(start=start, end=end, text=text, speaker=None)


# ---------------------------------------------------------------------------
# TimeWindow.vision_context default
# ---------------------------------------------------------------------------


def test_time_window_vision_context_defaults_to_empty_string() -> None:
    w = TimeWindow(start=0.0, end=10.0, text="hello", frame_paths=[])
    assert w.vision_context == ""


def test_time_window_vision_context_can_be_set() -> None:
    w = TimeWindow(start=0.0, end=10.0, text="hi", frame_paths=[], vision_context="Some scene.")
    assert w.vision_context == "Some scene."


# ---------------------------------------------------------------------------
# _vision_context_in_range helper
# ---------------------------------------------------------------------------


def test_vision_context_in_range_basic() -> None:
    vision_map = {0.0: "Empty road.", 10.0: "Car arrives.", 25.0: "Driver exits."}
    result = _vision_context_in_range(vision_map, 0.0, 15.0)
    assert "00:00 — Empty road." in result
    assert "00:10 — Car arrives." in result
    assert "Driver exits." not in result


def test_vision_context_in_range_empty_map_returns_empty_string() -> None:
    assert _vision_context_in_range({}, 0.0, 30.0) == ""


def test_vision_context_in_range_sorted_by_timestamp() -> None:
    vision_map = {20.0: "C", 0.0: "A", 10.0: "B"}
    result = _vision_context_in_range(vision_map, 0.0, 30.0)
    lines = result.splitlines()
    assert lines[0].endswith("A")
    assert lines[1].endswith("B")
    assert lines[2].endswith("C")


# ---------------------------------------------------------------------------
# chunk_segments — vision_map integration
# ---------------------------------------------------------------------------


def test_chunk_segments_populates_vision_context() -> None:
    segs = [_seg(0.0, 5.0, "Hello"), _seg(5.0, 10.0, "World")]
    vision_map = {0.0: "Person at desk.", 5.0: "Person stands up."}
    windows = chunk_segments(segs, window_sec=20.0, vision_map=vision_map)
    assert len(windows) == 1
    assert "Person at desk." in windows[0].vision_context
    assert "Person stands up." in windows[0].vision_context


def test_chunk_segments_no_vision_map_gives_empty_context() -> None:
    segs = [_seg(0.0, 5.0, "Hello")]
    windows = chunk_segments(segs, window_sec=20.0)
    assert windows[0].vision_context == ""


def test_chunk_segments_vision_context_scoped_to_window() -> None:
    """Frames outside a window's time range must not bleed into it."""
    segs = [
        _seg(0.0, 20.0, "First window"),
        _seg(20.0, 40.0, "Second window"),
    ]
    vision_map = {
        5.0: "In first window.",
        30.0: "In second window.",
    }
    windows = chunk_segments(segs, window_sec=20.0, vision_map=vision_map)
    assert len(windows) == 2
    assert "In first window." in windows[0].vision_context
    assert "In second window." not in windows[0].vision_context
    assert "In second window." in windows[1].vision_context
    assert "In first window." not in windows[1].vision_context
