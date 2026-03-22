import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from event_timeline_extractor.web.app import create_app


def _has(cmd: str) -> bool:
    return shutil.which(cmd) is not None


@pytest.fixture
def ffmpeg_available() -> bool:
    return _has("ffmpeg") and _has("ffprobe")


@pytest.fixture
def ytdlp_available() -> bool:
    return _has("yt-dlp")


@pytest.fixture
def web_client() -> TestClient:
    """Shared FastAPI test client (same app factory as integration tests)."""
    return TestClient(create_app())


def make_tiny_mp4(path, *, duration_sec: float = 1.0) -> None:
    """Create a minimal mp4 with silent mono audio (video-only breaks WAV extract)."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=320x240:d={duration_sec}",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=16000:cl=mono",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
