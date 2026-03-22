"""Deterministic fake transcript for tests and dry runs."""

from __future__ import annotations

from pathlib import Path

from event_timeline_extractor.transcription.base import TranscriptSegment


class StubTranscriber:
    def transcribe(self, wav_path: str | Path) -> list[TranscriptSegment]:
        _ = Path(wav_path)
        return [
            TranscriptSegment(0.0, 5.0, "Unit one: approach the vehicle.", speaker="A"),
            TranscriptSegment(5.0, 12.0, "Step out of the car please.", speaker="A"),
            TranscriptSegment(12.0, 20.0, "What is going on?", speaker="B"),
        ]
