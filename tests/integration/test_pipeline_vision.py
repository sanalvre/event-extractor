"""Integration tests for vision analysis in the pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from event_timeline_extractor.config import Settings
from event_timeline_extractor.pipeline import PipelineInput, _run_vision_analysis, run_pipeline
from tests.conftest import make_tiny_mp4

# ---------------------------------------------------------------------------
# _run_vision_analysis unit-style test (mocked VLM)
# ---------------------------------------------------------------------------


def test_run_vision_analysis_returns_timestamp_map(tmp_path: Path, ffmpeg_available: bool) -> None:
    """_run_vision_analysis extracts frames and returns {timestamp → description}."""
    if not ffmpeg_available:
        pytest.skip("ffmpeg not available")

    video = tmp_path / "vid.mp4"
    make_tiny_mp4(video, duration_sec=2.0)

    settings = Settings(ete_vision_frame_interval=1)

    from event_timeline_extractor.vision.memories_s0 import FrameDescription

    fake_descriptions = [
        FrameDescription(timestamp=0.0, description="Empty scene."),
        FrameDescription(timestamp=1.0, description="Person enters."),
    ]

    with patch(
        "event_timeline_extractor.vision.memories_s0.MemoriesS0Analyzer"
    ) as mock_analyzer_cls:
        mock_instance = MagicMock()
        mock_instance.analyze.return_value = fake_descriptions
        mock_analyzer_cls.return_value = mock_instance

        result = _run_vision_analysis(video, tmp_path / "work", settings, duration=2.0)

    assert result == {0.0: "Empty scene.", 1.0: "Person enters."}
    mock_instance.analyze.assert_called_once()


# ---------------------------------------------------------------------------
# Full pipeline — vision disabled (default)
# ---------------------------------------------------------------------------


def test_pipeline_vision_disabled_skips_vision_call(
    tmp_path: Path, ffmpeg_available: bool, monkeypatch
) -> None:
    """When ETE_VISION_ENABLED=false, _run_vision_analysis is never called."""
    if not ffmpeg_available:
        pytest.skip("ffmpeg not available")

    monkeypatch.setenv("ETE_TRANSCRIBER", "stub")
    monkeypatch.setenv("ETE_VISION_ENABLED", "false")

    video = tmp_path / "vid.mp4"
    make_tiny_mp4(video, duration_sec=1.0)

    with patch("event_timeline_extractor.pipeline._run_vision_analysis") as mock_vis:
        run_pipeline(PipelineInput(file_path=video), work_dir=tmp_path / "work", dry_run=True)

    mock_vis.assert_not_called()


def test_pipeline_meta_vision_false_by_default(
    tmp_path: Path, ffmpeg_available: bool, monkeypatch
) -> None:
    """meta['vision'] should be False when vision is not enabled."""
    if not ffmpeg_available:
        pytest.skip("ffmpeg not available")

    monkeypatch.setenv("ETE_TRANSCRIBER", "stub")
    video = tmp_path / "vid.mp4"
    make_tiny_mp4(video, duration_sec=1.0)

    result = run_pipeline(PipelineInput(file_path=video), work_dir=tmp_path / "work", dry_run=True)
    assert result.meta is not None
    assert result.meta.get("vision") is False


# ---------------------------------------------------------------------------
# Full pipeline — vision enabled (VLM mocked)
# ---------------------------------------------------------------------------


def test_pipeline_vision_enabled_calls_analyzer_and_sets_meta(
    tmp_path: Path, ffmpeg_available: bool, monkeypatch
) -> None:
    """When ETE_VISION_ENABLED=true, _run_vision_analysis is called and meta reflects it."""
    if not ffmpeg_available:
        pytest.skip("ffmpeg not available")

    monkeypatch.setenv("ETE_TRANSCRIBER", "stub")
    monkeypatch.setenv("ETE_VISION_ENABLED", "true")
    monkeypatch.setenv("ETE_VISION_FRAME_INTERVAL", "1")

    video = tmp_path / "vid.mp4"
    make_tiny_mp4(video, duration_sec=2.0)

    with patch(
        "event_timeline_extractor.pipeline._run_vision_analysis", return_value={}
    ) as mock_vis:
        result = run_pipeline(
            PipelineInput(file_path=video), work_dir=tmp_path / "work", dry_run=True
        )

    mock_vis.assert_called_once()
    assert result.meta is not None
    assert result.meta.get("vision") is True


def test_pipeline_vision_context_flows_into_windows(
    tmp_path: Path, ffmpeg_available: bool, monkeypatch
) -> None:
    """vision_map returned by _run_vision_analysis is used to populate window vision_context."""
    if not ffmpeg_available:
        pytest.skip("ffmpeg not available")

    monkeypatch.setenv("ETE_TRANSCRIBER", "stub")
    monkeypatch.setenv("ETE_VISION_ENABLED", "true")

    video = tmp_path / "vid.mp4"
    make_tiny_mp4(video, duration_sec=2.0)

    captured_windows: list = []

    def fake_synthesize_dry_run(windows):
        captured_windows.extend(windows)
        from event_timeline_extractor.schema import TimelineResult
        return TimelineResult(events=[], meta={"dry_run": True})

    with patch("event_timeline_extractor.pipeline._run_vision_analysis",
               return_value={0.0: "Officer exits vehicle."}), \
         patch("event_timeline_extractor.pipeline.TimelineSynthesizer") as mock_synth_cls:
        mock_instance = MagicMock()
        mock_instance.dry_run.side_effect = fake_synthesize_dry_run
        mock_synth_cls.return_value = mock_instance

        run_pipeline(PipelineInput(file_path=video), work_dir=tmp_path / "work", dry_run=True)

    assert captured_windows, "synthesizer should have received windows"
    combined = " ".join(w.vision_context for w in captured_windows)
    assert "Officer exits vehicle." in combined
