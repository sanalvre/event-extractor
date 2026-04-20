"""ffmpeg wrappers for audio + frame extraction."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from event_timeline_extractor.resilience import retry_call


class FFmpegError(RuntimeError):
    pass


_FFMPEG_ATTEMPTS = 2
_FFMPEG_RETRY_DELAY_SEC = 0.5


def _run_ffmpeg(argv: list[str], *, timeout_sec: int = 7200) -> None:
    try:
        retry_call(
            lambda: subprocess.run(
                argv,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            ),
            attempts=_FFMPEG_ATTEMPTS,
            delay_seconds=_FFMPEG_RETRY_DELAY_SEC,
            should_retry=lambda exc: isinstance(exc, subprocess.TimeoutExpired),
        )
    except FileNotFoundError as e:
        raise FFmpegError(
            "ffmpeg not found on PATH. Install ffmpeg and add it to PATH."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise FFmpegError(
            f"ffmpeg timed out after {timeout_sec}s (retried {_FFMPEG_ATTEMPTS} times)."
        ) from e
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or "").strip()
        raise FFmpegError(f"ffmpeg failed: {err[:4000]}") from e


def probe_duration_seconds(media_path: Path, *, ffprobe_bin: str = "ffprobe") -> float:
    """Return duration in seconds (float)."""
    argv = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(media_path),
    ]
    try:
        p = retry_call(
            lambda: subprocess.run(
                argv,
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            ),
            attempts=_FFMPEG_ATTEMPTS,
            delay_seconds=_FFMPEG_RETRY_DELAY_SEC,
            should_retry=lambda exc: isinstance(exc, subprocess.TimeoutExpired),
        )
    except FileNotFoundError as e:
        raise FFmpegError("ffprobe not found on PATH.") from e
    except subprocess.TimeoutExpired as e:
        raise FFmpegError(
            f"ffprobe timed out after 120s (retried {_FFMPEG_ATTEMPTS} times)."
        ) from e
    except subprocess.CalledProcessError as e:
        raise FFmpegError(f"ffprobe failed: {(e.stderr or '')[:2000]}") from e
    data = json.loads(p.stdout)
    dur = float(data["format"]["duration"])
    return dur


def extract_audio_wav_16k_mono(
    media_path: Path,
    wav_out: Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
    max_duration_sec: float | None = None,
) -> Path:
    """Extract 16 kHz mono WAV for speech models.

    If ``max_duration_sec`` is set, only the first N seconds of audio are decoded
    (limits work for local files and is a safety cap after partial downloads).
    """
    wav_out.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(media_path),
    ]
    if max_duration_sec is not None:
        if max_duration_sec <= 0:
            raise ValueError("max_duration_sec must be positive")
        argv.extend(["-t", str(max_duration_sec)])
    argv.extend(
        [
            "-ac",
            "1",
            "-ar",
            "16000",
            "-vn",
            str(wav_out),
        ]
    )
    _run_ffmpeg(argv)
    return wav_out


def extract_frames_every_interval(
    media_path: Path,
    frames_dir: Path,
    *,
    interval_sec: float = 2.0,
    ffmpeg_bin: str = "ffmpeg",
) -> list[Path]:
    """Extract one JPEG per interval (floor). Returns sorted frame paths."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(frames_dir / "frame_%06d.jpg")
    # fps filter: 1/interval frames per second
    fps = 1.0 / interval_sec if interval_sec > 0 else 1.0
    argv = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(media_path),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "3",
        pattern,
    ]
    _run_ffmpeg(argv)
    paths = sorted(frames_dir.glob("frame_*.jpg"))
    return paths
