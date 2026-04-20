"""Process a *middle* slice of a longer local file (CLI workflow; UI is YouTube-only)."""

from __future__ import annotations

import subprocess

import pytest

from event_timeline_extractor.config import Settings
from event_timeline_extractor.pipeline import PipelineInput, run_pipeline
from tests.conftest import make_tiny_mp4


def test_pipeline_on_ffmpeg_extracted_middle_segment(
    tmp_path, ffmpeg_available, monkeypatch
) -> None:
    """A \"different part\" of the same synthetic video: full clip → extract seconds 4–7 → pipeline.

    The web UI only downloads **from the start** of a YouTube video. For a **mid-file** cut,
    use ffmpeg (same as a user would) then ``ete run --file`` on the excerpt.
    """
    if not ffmpeg_available:
        pytest.skip("ffmpeg/ffprobe not on PATH")

    long_clip = tmp_path / "long.mp4"
    make_tiny_mp4(long_clip, duration_sec=10.0)

    # Middle 3 seconds of the 10s clip (not from t=0).
    segment = tmp_path / "segment_middle.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(long_clip),
            "-ss",
            "4",
            "-t",
            "3",
            "-c",
            "copy",
            str(segment),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )

    monkeypatch.setenv("ETE_USE_STUB", "1")
    settings = Settings(ete_use_stub=True, ete_transcriber="stub")

    result = run_pipeline(
        PipelineInput(file_path=segment),
        work_dir=tmp_path / "work",
        settings=settings,
        window_sec=30.0,
        dry_run=True,
    )
    assert result.events
    assert result.meta
    assert result.meta.get("input_source") == "file"
    assert result.meta.get("transcriber") == "stub"
    assert result.meta.get("asr_model") == "stub"
    assert result.meta.get("diarization") == "none"
    assert result.meta.get("dry_run") is True
