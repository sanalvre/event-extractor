"""Group transcript segments into time windows for the LLM."""

from __future__ import annotations

from dataclasses import dataclass

from event_timeline_extractor.transcription.base import TranscriptSegment


@dataclass(frozen=True)
class TimeWindow:
    start: float
    end: float
    text: str
    frame_paths: list[str]
    vision_context: str = ""


def format_mmss(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def format_segment_line(seg: TranscriptSegment) -> str:
    """One transcript line per segment: ``[MM:SS] SPEAKER: text`` (speaker optional)."""
    ts = format_mmss(seg.start)
    prefix = f"{seg.speaker}: " if seg.speaker else ""
    return f"[{ts}] {prefix}{seg.text.strip()}"


def chunk_segments(
    segments: list[TranscriptSegment],
    *,
    window_sec: float = 20.0,
    frame_paths_by_time: dict[float, str] | None = None,
    vision_map: dict[float, str] | None = None,
    speaker_aware: bool = False,
) -> list[TimeWindow]:
    """Merge transcript segments into windows of roughly `window_sec` seconds.

    Args:
        segments: Transcript segments from ASR.
        window_sec: Target window length in seconds.
        frame_paths_by_time: Optional map of {timestamp → frame_path} for attaching frames.
        vision_map: Optional map of {timestamp → description} from visual frame analysis.
        speaker_aware: When True, also break windows at speaker changes (natural scene
            boundaries).  Requires segments to carry ``.speaker`` labels.  Long
            monologues are still split at ``window_sec`` to keep LLM context bounded.
    """
    if not segments:
        return []
    frame_paths_by_time = frame_paths_by_time or {}
    vision_map = vision_map or {}
    sorted_seg = sorted(segments, key=lambda s: s.start)
    windows: list[TimeWindow] = []
    cur_start = sorted_seg[0].start
    buf: list[TranscriptSegment] = []
    buf_end = cur_start

    def flush() -> None:
        nonlocal cur_start, buf, buf_end
        if not buf:
            return
        text = "\n".join(format_segment_line(s) for s in buf).strip()
        frames = _frames_in_range(frame_paths_by_time, cur_start, buf_end)
        vis_ctx = _vision_context_in_range(vision_map, cur_start, buf_end)
        windows.append(
            TimeWindow(start=cur_start, end=buf_end, text=text, frame_paths=frames, vision_context=vis_ctx)
        )
        buf = []

    for seg in sorted_seg:
        if not buf:
            cur_start = seg.start
            buf_end = seg.start
        potential_end = max(buf_end, seg.end)
        # Break on time limit OR (optionally) on speaker change.
        speaker_changed = (
            speaker_aware
            and buf
            and seg.speaker is not None
            and buf[-1].speaker is not None
            and seg.speaker != buf[-1].speaker
        )
        if buf and (potential_end - cur_start > window_sec or speaker_changed):
            flush()
            cur_start = seg.start
            buf_end = seg.start
            buf = []
        buf.append(seg)
        buf_end = max(buf_end, seg.end)
        if buf_end - cur_start >= window_sec:
            flush()
    flush()
    return windows


def _vision_context_in_range(
    vision_map: dict[float, str],
    start: float,
    end: float,
) -> str:
    """Build a formatted vision context string for timestamps within [start, end]."""
    lines: list[str] = []
    for t, desc in sorted(vision_map.items()):
        if start - 1e-6 <= t <= end + 1e-6:
            lines.append(f"{format_mmss(t)} — {desc}")
    return "\n".join(lines)


def _frames_in_range(
    by_time: dict[float, str],
    start: float,
    end: float,
) -> list[str]:
    out: list[str] = []
    for t, p in sorted(by_time.items()):
        if start - 1e-6 <= t <= end + 1e-6:
            out.append(p)
    return out


def attach_frames_to_timeline(
    frames: list[str],
    duration_sec: float,
) -> dict[float, str]:
    """Map approximate timestamp -> frame path (evenly spaced)."""
    if not frames or duration_sec <= 0:
        return {}
    n = len(frames)
    out: dict[float, str] = {}
    for i, p in enumerate(frames):
        t = (i / max(n - 1, 1)) * duration_sec
        out[round(t, 3)] = p
    return out
