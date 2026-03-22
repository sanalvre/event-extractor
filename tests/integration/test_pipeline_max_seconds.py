"""Pipeline with max_seconds on local file (stub ASR; no network)."""

import pytest

from event_timeline_extractor.config import Settings
from event_timeline_extractor.pipeline import PipelineInput, run_pipeline
from tests.conftest import make_tiny_mp4


def test_pipeline_respects_max_seconds_local_stub(tmp_path, ffmpeg_available, monkeypatch) -> None:
    if not ffmpeg_available:
        pytest.skip("ffmpeg/ffprobe not on PATH")
    vid = tmp_path / "clip.mp4"
    make_tiny_mp4(vid, duration_sec=3.0)

    monkeypatch.setenv("ETE_USE_STUB", "1")
    settings = Settings(ete_use_stub=True, ete_transcriber="stub")

    result = run_pipeline(
        PipelineInput(file_path=vid),
        work_dir=tmp_path / "w",
        settings=settings,
        window_sec=30.0,
        max_seconds=2.0,
        dry_run=True,
    )
    assert result.meta and result.meta.get("processed_seconds_cap") == 2.0
    assert result.events
