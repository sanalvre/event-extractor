"""Resolve Transcriber from settings."""

from __future__ import annotations

import logging

from event_timeline_extractor.config import Settings
from event_timeline_extractor.transcription.base import Transcriber
from event_timeline_extractor.transcription.faster_whisper_backend import FasterWhisperTranscriber
from event_timeline_extractor.transcription.stub import StubTranscriber

logger = logging.getLogger(__name__)


def get_transcriber(settings: Settings) -> Transcriber:
    """Resolve ASR backend.

    ``ETE_TRANSCRIBER=faster_whisper`` always wins over a stray ``ETE_USE_STUB=1`` in the
    process environment (system env vars can override ``.env`` in pydantic-settings, which
    previously forced the stub even when real ASR was intended).
    """
    name = (settings.ete_transcriber or "faster_whisper").lower().strip()
    if name in ("stub", "fake", "test"):
        logger.debug("Using StubTranscriber (ETE_TRANSCRIBER=stub).")
        return StubTranscriber()
    if name in ("faster_whisper", "faster-whisper", "whisper"):
        return FasterWhisperTranscriber(settings)
    if settings.ete_use_stub:
        logger.debug("Using StubTranscriber (ETE_USE_STUB=1).")
        return StubTranscriber()
    raise ValueError(f"Unknown ETE_TRANSCRIBER: {settings.ete_transcriber!r}")
