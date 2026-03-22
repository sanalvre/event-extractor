"""Full pipeline with stub transcriber + dry-run LLM (no network)."""

import pytest

from event_timeline_extractor.config import Settings
from event_timeline_extractor.pipeline import PipelineInput, run_pipeline
from tests.conftest import make_tiny_mp4


def test_pipeline_local_file_dry_run(tmp_path, ffmpeg_available, monkeypatch) -> None:
    if not ffmpeg_available:
        pytest.skip("ffmpeg/ffprobe not on PATH")
    vid = tmp_path / "clip.mp4"
    make_tiny_mp4(vid, duration_sec=1.0)

    monkeypatch.setenv("ETE_USE_STUB", "1")
    settings = Settings(ete_use_stub=True, ete_transcriber="stub")

    result = run_pipeline(
        PipelineInput(file_path=vid),
        work_dir=tmp_path / "w",
        settings=settings,
        window_sec=30.0,
        dry_run=True,
    )
    assert result.events
    assert result.meta and result.meta.get("dry_run") is True
