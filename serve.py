"""Launcher that sets KMP_DUPLICATE_LIB_OK before uvicorn starts.

Fixes the OpenMP conflict between faster-whisper (CTranslate2) and other
libraries that each bundle their own OpenMP runtime (libiomp5md.dll vs
libomp140.x86_64.dll).  Without this flag the server crashes on first
transcription request.

Usage:
    python serve.py
    python serve.py --port 8765 --host 127.0.0.1
"""

import os
import sys

# Must be set before any OpenMP-linked library is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse

import uvicorn  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Event Timeline Extractor web server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes (dev).")
    args = parser.parse_args()

    uvicorn.run(
        "event_timeline_extractor.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
