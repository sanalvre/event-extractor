"""Unit tests for MemoriesS0Analyzer (memories-s0 VLM wrapper)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from event_timeline_extractor.vision.memories_s0 import FrameDescription, MemoriesS0Analyzer

# ---------------------------------------------------------------------------
# FrameDescription dataclass
# ---------------------------------------------------------------------------


def test_frame_description_fields() -> None:
    fd = FrameDescription(timestamp=5.0, description="A car approaches the gate.")
    assert fd.timestamp == 5.0
    assert fd.description == "A car approaches the gate."


def test_frame_description_equality() -> None:
    assert FrameDescription(1.0, "X") == FrameDescription(1.0, "X")
    assert FrameDescription(1.0, "X") != FrameDescription(2.0, "X")


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def test_resolve_device_explicit_cpu() -> None:
    assert MemoriesS0Analyzer._resolve_device("cpu") == "cpu"


def test_resolve_device_explicit_cuda() -> None:
    assert MemoriesS0Analyzer._resolve_device("cuda") == "cuda"


def test_resolve_device_auto_falls_back_to_cpu_when_no_cuda() -> None:
    """When torch reports no CUDA, device should fall back to 'cpu' with a warning."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    with patch.dict(sys.modules, {"torch": mock_torch}):
        with pytest.warns(RuntimeWarning, match="CPU"):
            device = MemoriesS0Analyzer._resolve_device(None)
    assert device == "cpu"


# ---------------------------------------------------------------------------
# analyze() — core logic (model mocked out)
# ---------------------------------------------------------------------------


def _make_analyzer(device: str = "cpu") -> MemoriesS0Analyzer:
    """Create an analyzer with a pre-loaded (mocked) model so no HF download occurs."""
    analyzer = MemoriesS0Analyzer.__new__(MemoriesS0Analyzer)
    analyzer._device = device
    analyzer._model = MagicMock()
    analyzer._processor = MagicMock()
    return analyzer


def test_analyze_empty_inputs_returns_empty() -> None:
    analyzer = _make_analyzer()
    assert analyzer.analyze([], []) == []


def test_analyze_mismatched_lengths_raises() -> None:
    analyzer = _make_analyzer()
    with pytest.raises(ValueError, match="must match"):
        analyzer.analyze(["frame.jpg"], [1.0, 2.0])


def test_analyze_returns_correct_frame_descriptions(tmp_path: Path) -> None:
    frames = [tmp_path / f"frame_{i:06d}.jpg" for i in range(3)]
    for f in frames:
        f.write_bytes(b"fake-jpeg")
    timestamps = [0.0, 10.0, 20.0]

    analyzer = _make_analyzer()
    descriptions = ["Scene A.", "Scene B.", "Scene C."]

    with patch.object(analyzer, "_ensure_loaded"), patch.object(
        analyzer, "_describe_frame", side_effect=descriptions
    ):
        result = analyzer.analyze([str(f) for f in frames], timestamps)

    assert result == [
        FrameDescription(timestamp=0.0, description="Scene A."),
        FrameDescription(timestamp=10.0, description="Scene B."),
        FrameDescription(timestamp=20.0, description="Scene C."),
    ]


def test_analyze_calls_describe_frame_for_each_path(tmp_path: Path) -> None:
    frames = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    for f in frames:
        f.write_bytes(b"x")

    analyzer = _make_analyzer()
    describe_mock = MagicMock(return_value="desc")

    with patch.object(analyzer, "_ensure_loaded"), patch.object(
        analyzer, "_describe_frame", describe_mock
    ):
        analyzer.analyze([str(f) for f in frames], [0.0, 5.0])

    assert describe_mock.call_count == 2
    describe_mock.assert_any_call(frames[0])
    describe_mock.assert_any_call(frames[1])


# ---------------------------------------------------------------------------
# _ensure_loaded() — import-error path
# ---------------------------------------------------------------------------


def test_ensure_loaded_raises_helpful_error_without_transformers() -> None:
    """If transformers is absent, _ensure_loaded should raise ImportError with install hint."""
    analyzer = MemoriesS0Analyzer.__new__(MemoriesS0Analyzer)
    analyzer._device = "cpu"
    analyzer._model = None
    analyzer._processor = None

    with patch.dict(sys.modules, {"transformers": None}):
        with pytest.raises((ImportError, SystemError)):
            analyzer._ensure_loaded()
