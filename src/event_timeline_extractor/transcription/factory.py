"""Resolve Transcriber from settings."""

from __future__ import annotations

import logging

from event_timeline_extractor.config import Settings
from event_timeline_extractor.transcription.base import Transcriber
from event_timeline_extractor.transcription.faster_whisper_backend import FasterWhisperTranscriber
from event_timeline_extractor.transcription.stub import StubTranscriber

logger = logging.getLogger(__name__)

_MEMORIES_BASE_URL = "https://mavi-backend.memories.ai/serve/api/v2"


def get_transcriber(settings: Settings) -> Transcriber:
    """Resolve ASR backend.

    ``ETE_TRANSCRIBER=faster_whisper`` always wins over a stray ``ETE_USE_STUB=1`` in the
    process environment (system env vars can override ``.env`` in pydantic-settings, which
    previously forced the stub even when real ASR was intended).

    Supported values for ETE_TRANSCRIBER:
    - ``faster_whisper`` (default) — local Whisper via faster-whisper
    - ``memories``                 — memories.ai cloud API with built-in diarization
                                      (requires MEMORIES_API_KEY)
    - ``groq``                     — Groq cloud API (189× real-time, requires GROQ_API_KEY)
    - ``stub`` / ``fake`` / ``test`` — deterministic placeholder for tests
    """
    name = (settings.ete_transcriber or "faster_whisper").lower().strip()
    if name in ("stub", "fake", "test"):
        logger.debug("Using StubTranscriber (ETE_TRANSCRIBER=stub).")
        return StubTranscriber()
    if name in ("faster_whisper", "faster-whisper", "whisper"):
        return FasterWhisperTranscriber(settings)
    if name == "memories":
        from event_timeline_extractor.transcription.memories_backend import (
            MemoriesTranscriber,  # noqa: PLC0415
        )

        key = settings.memories_key_plain()
        if not key:
            raise ValueError(
                "ETE_TRANSCRIBER=memories requires MEMORIES_API_KEY. "
                "Get a key at https://api-platform.memories.ai and add it to .env."
            )
        logger.info(
            "Using MemoriesTranscriber (speaker=%s).", settings.memories_transcription_speaker
        )
        return MemoriesTranscriber(
            api_key=key,
            speaker=settings.memories_transcription_speaker,
            base_url=_MEMORIES_BASE_URL,
        )
    if name == "groq":
        from event_timeline_extractor.transcription.groq_backend import (
            GroqTranscriber,  # noqa: PLC0415
        )

        key = settings.groq_key_plain()
        if not key:
            raise ValueError(
                "ETE_TRANSCRIBER=groq requires GROQ_API_KEY. "
                "Get a free key at https://console.groq.com and add it to .env."
            )
        logger.info("Using GroqTranscriber (model=%s).", settings.groq_model)
        return GroqTranscriber(
            api_key=key,
            model=settings.groq_model,
            base_url=settings.groq_base_url,
        )
    if settings.ete_use_stub:
        logger.debug("Using StubTranscriber (ETE_USE_STUB=1).")
        return StubTranscriber()
    raise ValueError(f"Unknown ETE_TRANSCRIBER: {settings.ete_transcriber!r}")
