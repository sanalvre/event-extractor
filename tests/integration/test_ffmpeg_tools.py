import pytest

from event_timeline_extractor.ffmpeg_tools import (
    extract_audio_wav_16k_mono,
    extract_frames_every_interval,
    probe_duration_seconds,
)
from tests.conftest import make_tiny_mp4


def test_probe_and_extract_wav_and_frames(tmp_path, ffmpeg_available) -> None:
    if not ffmpeg_available:
        pytest.skip("ffmpeg/ffprobe not on PATH")
    vid = tmp_path / "t.mp4"
    make_tiny_mp4(vid, duration_sec=1.0)
    d = probe_duration_seconds(vid)
    assert 0.5 <= d <= 2.0

    wav = tmp_path / "a.wav"
    extract_audio_wav_16k_mono(vid, wav)
    assert wav.is_file() and wav.stat().st_size > 0

    frames = extract_frames_every_interval(vid, tmp_path / "frames", interval_sec=0.5)
    assert len(frames) >= 1

    wav_short = tmp_path / "short.wav"
    extract_audio_wav_16k_mono(vid, wav_short, max_duration_sec=0.3)
    d_short = probe_duration_seconds(wav_short)
    assert d_short <= 0.45
