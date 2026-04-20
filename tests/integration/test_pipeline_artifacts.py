"""Artifact persistence for pipeline runs."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from event_timeline_extractor.chunking import TimeWindow
from event_timeline_extractor.config import Settings
from event_timeline_extractor.pipeline import (
    MediaStage,
    PipelineInput,
    TranscriptStage,
    WindowStage,
    _plan_extraction_batches,
    run_pipeline,
)
from event_timeline_extractor.schema import TimelineResult
from tests.conftest import make_tiny_mp4


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_pipeline_persists_stage_artifacts_for_dry_run(
    tmp_path, ffmpeg_available, monkeypatch
) -> None:
    if not ffmpeg_available:
        pytest.skip("ffmpeg/ffprobe not on PATH")

    video = tmp_path / "clip.mp4"
    make_tiny_mp4(video, duration_sec=1.0)

    monkeypatch.setenv("ETE_USE_STUB", "1")
    settings = Settings(ete_use_stub=True, ete_transcriber="stub")
    work_dir = tmp_path / "work"

    result = run_pipeline(
        PipelineInput(file_path=video),
        work_dir=work_dir,
        settings=settings,
        window_sec=30.0,
        dry_run=True,
    )

    artifacts_dir = work_dir / "artifacts"
    assert artifacts_dir.is_dir()
    assert result.meta is not None
    assert result.meta.get("artifacts_dir") == str(artifacts_dir)
    assert result.meta.get("extraction_batches") == 1

    input_payload = _read_json(artifacts_dir / "input.json")
    media_payload = _read_json(artifacts_dir / "media.json")
    transcript_payload = _read_json(artifacts_dir / "transcript.json")
    windows_payload = _read_json(artifacts_dir / "windows.json")
    summary_payload = _read_json(artifacts_dir / "run_summary.json")
    timeline_payload = _read_json(artifacts_dir / "timeline.json")
    batch_payload = _read_json(artifacts_dir / "batches" / "batch_0000.json")

    assert input_payload["file_path"] == str(video)
    assert input_payload["youtube_url"] is None

    assert media_payload["input_kind"] == "file"
    assert media_payload["wav_path"].endswith("audio_16k.wav")

    assert transcript_payload["transcriber"] == "stub"
    assert transcript_payload["asr_model"] == "stub"
    assert transcript_payload["segments"]
    assert transcript_payload["segments"][0]["segment_id"].startswith("seg-")
    assert transcript_payload["full_transcript"]

    assert windows_payload["windows"]
    assert isinstance(windows_payload["speaker_aware"], bool)
    assert windows_payload["window_sec"] == 30.0
    assert windows_payload["windows"][0]["source_segment_ids"]
    assert batch_payload["dry_run"] is True
    assert batch_payload["window_start"] == 0
    assert batch_payload["result"]["events"][0]["source_segment_ids"]
    assert timeline_payload["events"] == result.model_dump()["events"]
    assert summary_payload["artifacts_dir"] == str(artifacts_dir)


def test_pipeline_persists_source_url_for_youtube_input(tmp_path, monkeypatch) -> None:
    settings = Settings(ete_use_stub=True, ete_transcriber="stub")
    work_dir = tmp_path / "work"
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    media_stage = tmp_path / "media.mp4"
    wav_path = work_dir / "audio_16k.wav"

    monkeypatch.setattr(
        "event_timeline_extractor.pipeline._prepare_media_stage",
        lambda *args, **kwargs: MediaStage(
            media_path=media_stage,
            duration_seconds=12.0,
            wav_path=wav_path,
            input_kind="youtube",
        ),
    )
    monkeypatch.setattr(
        "event_timeline_extractor.pipeline._run_transcription_stage",
        lambda *args, **kwargs: TranscriptStage(
            segments=[],
            full_transcript="",
            transcriber_name="stub",
            asr_model="stub",
        ),
    )
    monkeypatch.setattr(
        "event_timeline_extractor.pipeline._run_window_stage",
        lambda *args, **kwargs: WindowStage(
            windows=[],
            speaker_aware=False,
            vision_enabled=False,
        ),
    )

    class _FakeSynth:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def dry_run(self, _windows):
            return TimelineResult(events=[], meta={"dry_run": True})

    monkeypatch.setattr("event_timeline_extractor.pipeline.TimelineSynthesizer", _FakeSynth)

    result = run_pipeline(
        PipelineInput(youtube_url=youtube_url),
        work_dir=work_dir,
        settings=settings,
        dry_run=True,
    )

    summary_payload = _read_json(work_dir / "artifacts" / "run_summary.json")
    assert result.meta is not None
    assert result.meta.get("source_url") == youtube_url
    assert summary_payload["source_url"] == youtube_url


def test_pipeline_reuses_cached_transcript_and_windows(
    tmp_path, ffmpeg_available, monkeypatch
) -> None:
    if not ffmpeg_available:
        pytest.skip("ffmpeg/ffprobe not on PATH")

    video = tmp_path / "clip.mp4"
    make_tiny_mp4(video, duration_sec=1.0)

    monkeypatch.setenv("ETE_USE_STUB", "1")
    settings = Settings(ete_use_stub=True, ete_transcriber="stub")
    work_dir = tmp_path / "work"

    first = run_pipeline(
        PipelineInput(file_path=video),
        work_dir=work_dir,
        settings=settings,
        window_sec=30.0,
        dry_run=True,
    )

    with patch(
        "event_timeline_extractor.pipeline._run_transcription_stage",
        side_effect=AssertionError("transcription should be reused"),
    ), patch(
        "event_timeline_extractor.pipeline._run_window_stage",
        side_effect=AssertionError("windows should be reused"),
    ):
        second = run_pipeline(
            PipelineInput(file_path=video),
            work_dir=work_dir,
            settings=settings,
            window_sec=30.0,
            dry_run=True,
        )

    assert first.events == second.events
    assert second.meta is not None
    assert second.meta["reused_artifacts"]["transcript"] is True
    assert second.meta["reused_artifacts"]["windows"] is True
    assert second.meta["reused_extraction_batches"] == second.meta["extraction_batches"]


def test_pipeline_reuses_saved_extraction_batches(tmp_path, monkeypatch) -> None:
    settings = Settings(openrouter_api_key="sk-test", ete_transcriber="stub", ete_use_stub=True)
    work_dir = tmp_path / "work"
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    media_stage = MediaStage(
        media_path=tmp_path / "media.mp4",
        duration_seconds=120.0,
        wav_path=work_dir / "audio_16k.wav",
        input_kind="youtube",
    )
    transcript_stage = TranscriptStage(
        segments=[],
        full_transcript="",
        transcriber_name="stub",
        asr_model="stub",
    )
    windows = [
        TimeWindow(start=float(i), end=float(i + 1), text=f"[00:0{i}] event {i}", frame_paths=[])
        for i in range(10)
    ]
    window_stage = WindowStage(windows=windows, speaker_aware=False, vision_enabled=False)

    monkeypatch.setattr(
        "event_timeline_extractor.pipeline._prepare_media_stage",
        lambda *args, **kwargs: media_stage,
    )
    monkeypatch.setattr(
        "event_timeline_extractor.pipeline._run_transcription_stage",
        lambda *args, **kwargs: transcript_stage,
    )
    monkeypatch.setattr(
        "event_timeline_extractor.pipeline._run_window_stage",
        lambda *args, **kwargs: window_stage,
    )

    class _FakeSynth:
        calls = 0

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def synthesize(self, batch_windows):
            _FakeSynth.calls += 1
            return TimelineResult(
                events=[
                    {
                        "time": "00:00",
                        "event": f"batch-{_FakeSynth.calls}-size-{len(batch_windows)}",
                        "speaker": None,
                        "evidence": None,
                    }
                ],
                meta={"model": "fake"},
            )

    monkeypatch.setattr("event_timeline_extractor.pipeline.TimelineSynthesizer", _FakeSynth)

    first = run_pipeline(
        PipelineInput(youtube_url=youtube_url),
        work_dir=work_dir,
        settings=settings,
        dry_run=False,
    )
    assert _FakeSynth.calls == 2

    class _NoCallSynth:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def synthesize(self, _batch_windows):
            raise AssertionError("saved extraction batches should be reused")

    monkeypatch.setattr("event_timeline_extractor.pipeline.TimelineSynthesizer", _NoCallSynth)

    second = run_pipeline(
        PipelineInput(youtube_url=youtube_url),
        work_dir=work_dir,
        settings=settings,
        dry_run=False,
    )

    assert first.events == second.events
    assert second.meta is not None
    assert second.meta["reused_extraction_batches"] == 2
    assert (work_dir / "artifacts" / "batches" / "batch_0000.json").is_file()
    assert (work_dir / "artifacts" / "batches" / "batch_0001.json").is_file()


def test_pipeline_reports_stage_progress(tmp_path, ffmpeg_available, monkeypatch) -> None:
    if not ffmpeg_available:
        pytest.skip("ffmpeg/ffprobe not on PATH")

    video = tmp_path / "clip.mp4"
    make_tiny_mp4(video, duration_sec=1.0)

    monkeypatch.setenv("ETE_USE_STUB", "1")
    settings = Settings(ete_use_stub=True, ete_transcriber="stub")
    seen: list[tuple[str, dict]] = []

    run_pipeline(
        PipelineInput(file_path=video),
        work_dir=tmp_path / "work",
        settings=settings,
        window_sec=30.0,
        dry_run=True,
        progress_callback=lambda stage, payload: seen.append((stage, payload)),
    )

    stage_names = [stage for stage, _payload in seen]
    assert stage_names[0] == "preparing_media"
    assert "transcribing" in stage_names
    assert "windowing" in stage_names
    assert "extracting" in stage_names
    assert stage_names[-1] == "completed"
    extracting_payloads = [payload for stage, payload in seen if stage == "extracting"]
    assert extracting_payloads
    assert extracting_payloads[0]["total_batches"] >= 1


def test_pipeline_merges_duplicate_events_across_batches(tmp_path, monkeypatch) -> None:
    settings = Settings(
        openrouter_api_key="sk-test",
        ete_transcriber="stub",
        ete_use_stub=True,
        ete_validate_evidence=False,
    )
    work_dir = tmp_path / "work"
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    media_stage = MediaStage(
        media_path=tmp_path / "media.mp4",
        duration_seconds=120.0,
        wav_path=work_dir / "audio_16k.wav",
        input_kind="youtube",
    )
    transcript_stage = TranscriptStage(
        segments=[],
        full_transcript="",
        transcriber_name="stub",
        asr_model="stub",
    )
    windows = [
        TimeWindow(start=float(i), end=float(i + 1), text=f"[00:0{i}] event {i}", frame_paths=[])
        for i in range(10)
    ]
    window_stage = WindowStage(windows=windows, speaker_aware=False, vision_enabled=False)

    monkeypatch.setattr(
        "event_timeline_extractor.pipeline._prepare_media_stage",
        lambda *args, **kwargs: media_stage,
    )
    monkeypatch.setattr(
        "event_timeline_extractor.pipeline._run_transcription_stage",
        lambda *args, **kwargs: transcript_stage,
    )
    monkeypatch.setattr(
        "event_timeline_extractor.pipeline._run_window_stage",
        lambda *args, **kwargs: window_stage,
    )

    class _FakeSynth:
        calls = 0

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def synthesize(self, batch_windows):
            _FakeSynth.calls += 1
            event = {
                "time": "00:00",
                "event": "Host introduces the episode topic.",
                "speaker": "Host",
                "evidence": "Welcome back to the show.",
                "source_segment_ids": [f"seg-00000{_FakeSynth.calls}"],
                "source_start": "00:00",
                "source_end": "00:05",
            }
            return TimelineResult(events=[event], meta={"model": f"fake-{len(batch_windows)}"})

    monkeypatch.setattr("event_timeline_extractor.pipeline.TimelineSynthesizer", _FakeSynth)

    result = run_pipeline(
        PipelineInput(youtube_url=youtube_url),
        work_dir=work_dir,
        settings=settings,
        dry_run=False,
    )

    assert len(result.events) == 1
    assert result.events[0].source_segment_ids == ["seg-000001", "seg-000002"]
    assert result.meta is not None
    assert result.meta["postprocess"] == {
        "merged_duplicate_events": 1,
        "filtered_low_signal_events": 0,
    }


def test_plan_extraction_batches_scales_for_long_runs() -> None:
    settings = Settings(
        ete_extraction_batch_size=4,
        ete_extraction_max_batches=3,
        ete_extraction_max_batch_size=10,
    )
    windows = [
        TimeWindow(start=float(i), end=float(i + 1), text=f"window {i}", frame_paths=[])
        for i in range(25)
    ]

    plan = _plan_extraction_batches(windows, settings)

    assert plan.requested_batch_size == 4
    assert plan.effective_batch_size == 9
    assert plan.total_windows == 25
    assert plan.total_batches == 3


def test_pipeline_records_dynamic_batch_plan(tmp_path, monkeypatch) -> None:
    settings = Settings(
        openrouter_api_key="sk-test",
        ete_transcriber="stub",
        ete_use_stub=True,
        ete_validate_evidence=False,
        ete_extraction_batch_size=3,
        ete_extraction_max_batches=2,
        ete_extraction_max_batch_size=6,
    )
    work_dir = tmp_path / "work"
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    media_stage = MediaStage(
        media_path=tmp_path / "media.mp4",
        duration_seconds=120.0,
        wav_path=work_dir / "audio_16k.wav",
        input_kind="youtube",
    )
    transcript_stage = TranscriptStage(
        segments=[],
        full_transcript="",
        transcriber_name="stub",
        asr_model="stub",
    )
    windows = [
        TimeWindow(start=float(i), end=float(i + 1), text=f"[00:0{i}] event {i}", frame_paths=[])
        for i in range(10)
    ]
    window_stage = WindowStage(windows=windows, speaker_aware=False, vision_enabled=False)

    monkeypatch.setattr(
        "event_timeline_extractor.pipeline._prepare_media_stage",
        lambda *args, **kwargs: media_stage,
    )
    monkeypatch.setattr(
        "event_timeline_extractor.pipeline._run_transcription_stage",
        lambda *args, **kwargs: transcript_stage,
    )
    monkeypatch.setattr(
        "event_timeline_extractor.pipeline._run_window_stage",
        lambda *args, **kwargs: window_stage,
    )

    class _FakeSynth:
        calls = 0

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def synthesize(self, batch_windows):
            _FakeSynth.calls += 1
            return TimelineResult(
                events=[
                    {
                        "time": "00:00",
                        "event": f"batch-{_FakeSynth.calls}-size-{len(batch_windows)}",
                        "speaker": None,
                        "evidence": None,
                    }
                ],
                meta={"model": "fake"},
            )

    monkeypatch.setattr("event_timeline_extractor.pipeline.TimelineSynthesizer", _FakeSynth)

    result = run_pipeline(
        PipelineInput(youtube_url=youtube_url),
        work_dir=work_dir,
        settings=settings,
        dry_run=False,
    )

    assert _FakeSynth.calls == 2
    assert result.meta is not None
    assert result.meta["batch_plan"]["requested_batch_size"] == 3
    assert result.meta["batch_plan"]["effective_batch_size"] == 5
    assert result.meta["batch_plan"]["total_windows"] == 10
    assert result.meta["batch_plan"]["total_batches"] == 2
    assert result.meta["extraction_batches"] == 2
