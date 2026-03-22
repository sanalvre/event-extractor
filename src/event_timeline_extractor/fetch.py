"""Download media with yt-dlp."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

_YOUTUBE_HOSTS = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtu.be",
        "www.youtu.be",
    }
)


class FetchError(RuntimeError):
    pass


def resolve_ytdlp_invocation() -> list[str]:
    """How to invoke yt-dlp: console script next to ``sys.executable``, else ``python -m yt_dlp``."""
    scripts = Path(sys.executable).resolve().parent
    for name in ("yt-dlp.exe", "yt-dlp"):
        cand = scripts / name
        if cand.is_file():
            return [str(cand)]
    return [sys.executable, "-m", "yt_dlp"]


def is_probably_youtube_url(url: str) -> bool:
    u = url.strip()
    if not u.startswith(("http://", "https://")):
        return False
    try:
        host = urlparse(u).hostname or ""
    except ValueError:
        return False
    host = host.lower()
    if host in _YOUTUBE_HOSTS:
        return True
    return host.endswith(".youtube.com")


def validate_youtube_url(url: str) -> str:
    u = url.strip()
    if not is_probably_youtube_url(u):
        raise FetchError(
            "Expected a YouTube http(s) URL (youtube.com, youtu.be, or *.youtube.com)."
        )
    return u


def build_ytdlp_argv(
    url: str,
    output_dir: Path,
    *,
    ytdlp_invocation: list[str] | None = None,
    output_template: str = "source.%(ext)s",
    download_sections: str | None = None,
) -> list[str]:
    """Argv for: one video file into output_dir/output_template."""
    validate_youtube_url(url)
    inv = ytdlp_invocation if ytdlp_invocation is not None else resolve_ytdlp_invocation()
    out = output_dir / output_template
    argv: list[str] = [
        *inv,
        "--no-playlist",
        "--no-warnings",
        "-o",
        str(out),
    ]
    if download_sections:
        argv.extend(["--download-sections", download_sections])
    argv.append(url)
    return argv


def _first_file_matching(dir_path: Path, pattern: re.Pattern[str]) -> Path | None:
    for p in sorted(dir_path.iterdir()):
        if p.is_file() and pattern.search(p.name):
            return p
    return None


def find_downloaded_media(output_dir: Path) -> Path:
    """After yt-dlp, pick the main media file in output_dir."""
    video = _first_file_matching(output_dir, re.compile(r"\.(mp4|webm|mkv|mov|m4a|opus)$", re.I))
    if video:
        return video
    files = [p for p in output_dir.iterdir() if p.is_file()]
    if not files:
        raise FetchError("yt-dlp produced no files in the output directory.")
    return max(files, key=lambda p: p.stat().st_mtime)


def run_ytdlp_download(
    url: str,
    output_dir: Path,
    *,
    ytdlp_invocation: list[str] | None = None,
    download_sections: str | None = None,
    timeout_sec: int = 3600,
) -> Path:
    """Download best single-file format into output_dir; return path to media."""
    output_dir.mkdir(parents=True, exist_ok=True)
    inv = ytdlp_invocation if ytdlp_invocation is not None else resolve_ytdlp_invocation()
    argv = build_ytdlp_argv(
        url,
        output_dir,
        ytdlp_invocation=inv,
        download_sections=download_sections,
    )
    try:
        subprocess.run(
            argv,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except FileNotFoundError as e:
        hint = (
            "Install the project dependencies so yt-dlp is available: "
            "pip install -e ."
        )
        raise FetchError(
            f"Could not run yt-dlp ({inv[0]!r}). {hint}"
        ) from e
    except subprocess.TimeoutExpired as e:
        raise FetchError("yt-dlp timed out.") from e
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or "").strip()
        raise FetchError(f"yt-dlp failed (exit {e.returncode}): {err[:2000]}") from e
    return find_downloaded_media(output_dir)
