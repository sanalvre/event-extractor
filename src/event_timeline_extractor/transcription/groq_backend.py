"""Groq cloud transcription backend.

Uses Groq's Whisper Large v3 API (OpenAI-compatible) to transcribe audio at
~189× real-time speed.  No extra packages needed — uses httpx which is already
a core dependency.

Setup
-----
1. Get a free API key at https://console.groq.com
2. Add to .env:
       ETE_TRANSCRIBER=groq
       GROQ_API_KEY=gsk_...
3. Optionally change the model:
       GROQ_MODEL=whisper-large-v3-turbo   # faster, slightly lower accuracy

File size limit
---------------
Groq accepts audio files up to 25 MB.  At 16 kHz mono 16-bit WAV that is
roughly 13 minutes.  For longer recordings use ``--max-seconds`` to cap the
clip, or the pipeline will raise a clear error before uploading.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

from event_timeline_extractor.transcription.base import TranscriptSegment

logger = logging.getLogger(__name__)

_GROQ_SIZE_LIMIT_BYTES = 25 * 1024 * 1024  # 25 MB
_GROQ_TIMEOUT_SEC = 300.0


class GroqTranscriber:
    """Transcriber backed by Groq's cloud Whisper API.

    Implements the :class:`~event_timeline_extractor.transcription.base.Transcriber`
    protocol — drop-in replacement for :class:`FasterWhisperTranscriber`.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "whisper-large-v3",
        base_url: str = "https://api.groq.com/openai/v1",
    ) -> None:
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Get a free key at https://console.groq.com and add it to .env."
            )
        self._api_key = api_key
        self._model = model
        self._endpoint = base_url.rstrip("/") + "/audio/transcriptions"

    # ------------------------------------------------------------------
    # Transcriber protocol
    # ------------------------------------------------------------------

    def transcribe(self, wav_path: str | Path) -> list[TranscriptSegment]:
        """Upload *wav_path* to Groq and return timestamped segments.

        Args:
            wav_path: Path to a 16 kHz mono WAV file (produced by the pipeline).

        Returns:
            List of :class:`TranscriptSegment` objects in chronological order.

        Raises:
            ValueError: Audio file exceeds Groq's 25 MB upload limit.
            RuntimeError: Groq API returned an HTTP error.
        """
        wav_path = Path(wav_path)
        self._check_file_size(wav_path)

        logger.info(
            "Groq transcription: uploading %s (%.1f MB) with model %s…",
            wav_path.name,
            wav_path.stat().st_size / 1_048_576,
            self._model,
        )

        with wav_path.open("rb") as audio_file:
            with httpx.Client(timeout=_GROQ_TIMEOUT_SEC) as client:
                response = client.post(
                    self._endpoint,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    files={"file": (wav_path.name, audio_file, "audio/wav")},
                    data={
                        "model": self._model,
                        "response_format": "verbose_json",
                        "language": "en",
                    },
                )

        if response.status_code >= 400:
            snippet = response.text[:600]
            logger.error("Groq API error %s: %s", response.status_code, snippet)
            raise RuntimeError(
                f"Groq API returned HTTP {response.status_code}. "
                f"Details: {snippet}"
            )

        data = response.json()
        segments = self._parse_segments(data)
        logger.info("Groq returned %d segments.", len(segments))
        return segments

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_file_size(self, path: Path) -> None:
        size = path.stat().st_size
        if size > _GROQ_SIZE_LIMIT_BYTES:
            mb = size / 1_048_576
            raise ValueError(
                f"Audio file is {mb:.1f} MB — Groq's limit is 25 MB (~13 min at 16 kHz mono). "
                "Use --max-seconds to cap the clip length."
            )

    @staticmethod
    def _parse_segments(data: dict) -> list[TranscriptSegment]:
        """Map Groq's verbose_json response to TranscriptSegment objects."""
        raw_segments = data.get("segments", [])
        out: list[TranscriptSegment] = []
        for seg in raw_segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            out.append(
                TranscriptSegment(
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    text=text,
                    speaker=None,  # Groq does not provide speaker labels
                )
            )
        return out
