from event_timeline_extractor.transcription.base import TranscriptSegment
from event_timeline_extractor.transcription.diarization import assign_speakers_by_overlap


def test_assign_speakers_by_overlap() -> None:
    segs = [
        TranscriptSegment(0.0, 2.0, "a"),
        TranscriptSegment(3.0, 5.0, "b"),
    ]
    intervals = [
        (0.0, 1.5, "SPEAKER_00"),
        (2.5, 5.0, "SPEAKER_01"),
    ]
    out = assign_speakers_by_overlap(segs, intervals)
    assert out[0].speaker == "SPEAKER_00"
    assert out[1].speaker == "SPEAKER_01"
