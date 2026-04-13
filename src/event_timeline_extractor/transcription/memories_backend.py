"""memories.ai transcription backend.

Uses memories.ai's synchronous transcription API which returns timestamped
segments with optional built-in speaker diarization — collapsing Whisper +
pyannote into a single cloud call.

API docs: https://api-tools.memories.ai
Base URL:  https://mavi-backend.memories.ai/serve/api/v2

Setup
-----
1. Get an API key at https://api-platform.memories.ai (free tier available)
2. Add to .env:
       ETE_TRANSCRIBER=memories
       MEMORIES_API_KEY=sk-mai-...
3. Speaker diarization is on by default. To disable:
       MEMORIES_TRANSCRIPTION_SPEAKER=false

Flow
----
1. Upload WAV  →  POST /upload  →  asset_id
2. Transcribe  →  POST /transcriptions/sync-generate-audio  →  segments + speakers
3. Map to TranscriptSegment list (protocol-compatible with Whisper backend)

Pricing (at time of writing)
-----------------------------
- Without speaker: $0.001 / second of audio
- With speaker:    $0.002 / second of audio
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

from event_timeline_extractor.transcription.base import TranscriptSegment

logger = logging.getLogger(__name__)

_BASE = "https://mavi-backend.memories.ai/serve/api/v2"
_UPLOAD_TIMEOUT = 300.0   # seconds — large files can take a while
_TRANSCRIBE_TIMEOUT = 300.0


class MemoriesTranscriber:
    """Transcriber backed by the memories.ai cloud API.

    Implements the :class:`~event_timeline_extractor.transcription.base.Transcriber`
    protocol.  When *speaker=True* (default), the returned segments already carry
    speaker labels so pyannote diarization is not needed.
    """

    def __init__(
        self,
        api_key: str,
        *,
        speaker: bool = True,
        base_url: str = _BASE,
    ) -> None:
        if not api_key:
            raise ValueError(
                "MEMORIES_API_KEY is not set. "
                "Get a key at https://api-platform.memories.ai and add it to .env."
            )
        self._api_key = api_key
        self._speaker = speaker
        self._base = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Transcriber protocol
    # ------------------------------------------------------------------

    def transcribe(self, wav_path: str | Path) -> list[TranscriptSegment]:
        """Upload *wav_path* and return timestamped (+ optionally speaker-labelled) segments.

        Args:
            wav_path: Path to a 16 kHz mono WAV file produced by the pipeline.

        Returns:
            List of :class:`TranscriptSegment` objects in chronological order.
            When ``speaker=True``, each segment's ``.speaker`` field is set to
            the memories.ai speaker label (e.g. ``SPEAKER_00``).

        Raises:
            RuntimeError: API returned an HTTP error at any step.
        """
        path = Path(wav_path)
        asset_id = self._upload(path)
        items = self._transcribe(asset_id)
        segments = self._to_segments(items)
        logger.info(
            "memories.ai transcription complete: %d segments, speaker=%s",
            len(segments),
            self._speaker,
        )
        return segments

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self, *, json: bool = False) -> dict[str, str]:
        h = {"Authorization": self._api_key}
        if json:
            h["Content-Type"] = "application/json"
        return h

    def _upload(self, path: Path) -> str:
        """Upload audio file and return the memories.ai asset_id."""
        size_mb = path.stat().st_size / 1_048_576
        logger.info("Uploading %.1f MB to memories.ai…", size_mb)

        with path.open("rb") as fh:
            with httpx.Client(timeout=_UPLOAD_TIMEOUT) as client:
                r = client.post(
                    f"{self._base}/upload",
                    headers=self._headers(),
                    files={"file": (path.name, fh, "audio/wav")},
                )

        _raise_for_status(r, "upload")
        asset_id: str = r.json()["data"]["asset_id"]
        logger.info("Upload complete — asset_id: %s", asset_id)
        return asset_id

    def _transcribe(self, asset_id: str) -> list[dict]:
        """Run synchronous transcription and return raw item list."""
        logger.info(
            "Transcribing asset %s (speaker=%s)…", asset_id, self._speaker
        )
        with httpx.Client(timeout=_TRANSCRIBE_TIMEOUT) as client:
            r = client.post(
                f"{self._base}/transcriptions/sync-generate-audio",
                headers=self._headers(json=True),
                json={
                    "asset_id": asset_id,
                    "model": "whisper-1",
                    "speaker": self._speaker,
                },
            )
        _raise_for_status(r, "transcription")
        items: list[dict] = r.json()["data"]["items"]
        logger.info("Transcription returned %d raw items.", len(items))
        return items

    @staticmethod
    def _to_segments(items: list[dict]) -> list[TranscriptSegment]:
        """Map memories.ai response items to TranscriptSegment objects."""
        out: list[TranscriptSegment] = []
        for item in items:
            text = (item.get("text") or "").strip()
            if not text:
                continue
            out.append(
                TranscriptSegment(
                    start=float(item.get("start_time", 0.0)),
                    end=float(item.get("end_time", 0.0)),
                    text=text,
                    speaker=item.get("speaker") or None,
                )
            )
        return out


def _raise_for_status(r: httpx.Response, step: str) -> None:
    if r.status_code >= 400:
        snippet = r.text[:500]
        logger.error("memories.ai %s error %s: %s", step, r.status_code, snippet)
        raise RuntimeError(
            f"memories.ai {step} returned HTTP {r.status_code}. Details: {snippet}"
        )
