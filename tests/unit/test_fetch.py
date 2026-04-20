import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from event_timeline_extractor.fetch import (
    FetchError,
    build_ytdlp_argv,
    find_downloaded_media,
    is_probably_youtube_url,
    run_ytdlp_download,
    validate_youtube_url,
)


def test_is_probably_youtube_url() -> None:
    assert is_probably_youtube_url("https://www.youtube.com/watch?v=abc")
    assert is_probably_youtube_url("https://youtu.be/abc")
    assert not is_probably_youtube_url("https://example.com")
    assert not is_probably_youtube_url("ftp://youtube.com/x")


def test_validate_youtube_url_raises() -> None:
    with pytest.raises(FetchError):
        validate_youtube_url("https://example.com")


def test_build_ytdlp_argv_contains_url_and_output() -> None:
    out = Path("/tmp/w")
    argv = build_ytdlp_argv("https://youtu.be/abc", out, ytdlp_invocation=["yt-dlp"])
    assert argv[0] == "yt-dlp"
    assert "--no-playlist" in argv
    assert "https://youtu.be/abc" in argv


def test_build_ytdlp_argv_download_sections() -> None:
    out = Path("/tmp/w")
    argv = build_ytdlp_argv(
        "https://youtu.be/abc",
        out,
        ytdlp_invocation=["yt-dlp"],
        download_sections="*0:00-0:20",
    )
    assert argv[argv.index("--download-sections") + 1] == "*0:00-0:20"


def test_find_downloaded_media_prefers_video_ext(tmp_path: Path) -> None:
    (tmp_path / "source.mp4").write_bytes(b"x")
    p = find_downloaded_media(tmp_path)
    assert p.suffix == ".mp4"


@patch("event_timeline_extractor.fetch.subprocess.run")
def test_run_ytdlp_download_success(mock_run, tmp_path: Path) -> None:
    mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
    (tmp_path / "source.mp4").write_bytes(b"x")
    p = run_ytdlp_download("https://youtu.be/x", tmp_path)
    assert p.name == "source.mp4"


@patch("event_timeline_extractor.fetch.subprocess.run")
def test_run_ytdlp_download_retries_timeout(mock_run, tmp_path: Path) -> None:
    mock_run.side_effect = [
        subprocess.TimeoutExpired(cmd=["yt-dlp"], timeout=10),
        subprocess.CompletedProcess(args=[], returncode=0),
    ]
    (tmp_path / "source.mp4").write_bytes(b"x")

    p = run_ytdlp_download("https://youtu.be/x", tmp_path, timeout_sec=10)

    assert p.name == "source.mp4"
    assert mock_run.call_count == 2


@patch("event_timeline_extractor.fetch.subprocess.run")
def test_run_ytdlp_download_timeout_message_after_retries(mock_run, tmp_path: Path) -> None:
    mock_run.side_effect = subprocess.TimeoutExpired(cmd=["yt-dlp"], timeout=7)

    with pytest.raises(FetchError, match="timed out after 7s"):
        run_ytdlp_download("https://youtu.be/x", tmp_path, timeout_sec=7)


@patch("event_timeline_extractor.fetch.subprocess.run", side_effect=FileNotFoundError)
def test_run_ytdlp_missing_binary(_, tmp_path: Path) -> None:
    with pytest.raises(FetchError, match="Could not run yt-dlp"):
        run_ytdlp_download(
            "https://youtu.be/x",
            tmp_path,
            ytdlp_invocation=["/nonexistent/yt-dlp"],
        )
