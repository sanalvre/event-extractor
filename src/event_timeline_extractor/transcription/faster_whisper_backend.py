"""faster-whisper backend with model cache and fast defaults."""

from __future__ import annotations

import threading
from pathlib import Path

from event_timeline_extractor.config import Settings
from event_timeline_extractor.transcription.base import TranscriptSegment

_cache: dict[tuple[str, str, str], object] = {}
_lock = threading.Lock()


def _device_and_compute_type() -> tuple[str, str]:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def _get_model(settings: Settings):
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "faster-whisper is not installed. Run: pip install -e ."
        ) from e

    device, ctype = _device_and_compute_type()
    key = (settings.ete_whisper_model_size, device, ctype)
    with _lock:
        if key not in _cache:
            _cache[key] = WhisperModel(
                settings.ete_whisper_model_size,
                device=device,
                compute_type=ctype,
            )
        return _cache[key]


class FasterWhisperTranscriber:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = _get_model(settings)

    def transcribe(self, wav_path: str | Path) -> list[TranscriptSegment]:
        path = Path(wav_path)
        segments, _info = self._model.transcribe(
            str(path),
            beam_size=self._settings.ete_whisper_beam_size,
            vad_filter=self._settings.ete_whisper_vad,
            word_timestamps=self._settings.ete_whisper_word_timestamps,
        )
        out: list[TranscriptSegment] = []
        for s in segments:
            seg = TranscriptSegment(
                start=float(s.start),
                end=float(s.end),
                text=(s.text or "").strip(),
                speaker=None,
            )
            if seg.text:
                out.append(seg)
        return out
