from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None
    segment_id: str | None = None


@runtime_checkable
class Transcriber(Protocol):
    def transcribe(self, wav_path: str | Path) -> list[TranscriptSegment]: ...
