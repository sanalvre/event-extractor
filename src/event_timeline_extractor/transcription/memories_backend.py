"""memories.ai transcription backend."""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

from event_timeline_extractor.resilience import retry_call
from event_timeline_extractor.transcription.base import TranscriptSegment

logger = logging.getLogger(__name__)

_BASE = "https://mavi-backend.memories.ai/serve/api/v2"
_UPLOAD_TIMEOUT = 300.0
_TRANSCRIBE_TIMEOUT = 300.0
_MEMORIES_ATTEMPTS = 3
_MEMORIES_RETRY_DELAY_SEC = 1.0


class MemoriesTranscriber:
    """Transcriber backed by the memories.ai cloud API."""

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

    def transcribe(self, wav_path: str | Path) -> list[TranscriptSegment]:
        """Upload *wav_path* and return timestamped segments."""
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

    def _headers(self, *, json: bool = False) -> dict[str, str]:
        headers = {"Authorization": self._api_key}
        if json:
            headers["Content-Type"] = "application/json"
        return headers

    def _upload(self, path: Path) -> str:
        size_mb = path.stat().st_size / 1_048_576
        logger.info("Uploading %.1f MB to memories.ai...", size_mb)

        with path.open("rb") as fh:
            def _post() -> httpx.Response:
                fh.seek(0)
                with httpx.Client(timeout=_UPLOAD_TIMEOUT) as client:
                    return client.post(
                        f"{self._base}/upload",
                        headers=self._headers(),
                        files={"file": (path.name, fh, "audio/wav")},
                    )

            try:
                response = retry_call(
                    _post,
                    attempts=_MEMORIES_ATTEMPTS,
                    delay_seconds=_MEMORIES_RETRY_DELAY_SEC,
                    should_retry=lambda exc: isinstance(
                        exc,
                        (httpx.TimeoutException, httpx.NetworkError),
                    ),
                )
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                raise RuntimeError(
                    f"memories.ai upload failed after {_MEMORIES_ATTEMPTS} attempt(s): {exc}"
                ) from exc

        _raise_for_status(response, "upload")
        asset_id: str = response.json()["data"]["asset_id"]
        logger.info("Upload complete, asset_id: %s", asset_id)
        return asset_id

    def _transcribe(self, asset_id: str) -> list[dict]:
        logger.info("Transcribing asset %s (speaker=%s)...", asset_id, self._speaker)

        def _post() -> httpx.Response:
            with httpx.Client(timeout=_TRANSCRIBE_TIMEOUT) as client:
                return client.post(
                    f"{self._base}/transcriptions/sync-generate-audio",
                    headers=self._headers(json=True),
                    json={
                        "asset_id": asset_id,
                        "model": "whisper-1",
                        "speaker": self._speaker,
                    },
                )

        try:
            response = retry_call(
                _post,
                attempts=_MEMORIES_ATTEMPTS,
                delay_seconds=_MEMORIES_RETRY_DELAY_SEC,
                should_retry=lambda exc: isinstance(
                    exc,
                    (httpx.TimeoutException, httpx.NetworkError),
                ),
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise RuntimeError(
                f"memories.ai transcription failed after {_MEMORIES_ATTEMPTS} attempt(s): {exc}"
            ) from exc

        _raise_for_status(response, "transcription")
        items: list[dict] = response.json()["data"]["items"]
        logger.info("Transcription returned %d raw items.", len(items))
        return items

    @staticmethod
    def _to_segments(items: list[dict]) -> list[TranscriptSegment]:
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


def _raise_for_status(response: httpx.Response, step: str) -> None:
    if response.status_code >= 400:
        snippet = response.text[:500]
        logger.error("memories.ai %s error %s: %s", step, response.status_code, snippet)
        raise RuntimeError(
            f"memories.ai {step} returned HTTP {response.status_code}. Details: {snippet}"
        )
