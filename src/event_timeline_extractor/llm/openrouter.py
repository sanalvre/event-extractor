"""OpenRouter (OpenAI-compatible) client for timeline JSON."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from event_timeline_extractor.chunking import TimeWindow, format_mmss
from event_timeline_extractor.config import Settings
from event_timeline_extractor.schema import TimelineEvent, TimelineResult

logger = logging.getLogger(__name__)


def _redact_for_logs(text: str) -> str:
    t = text
    t = re.sub(r"Bearer\s+sk-[A-Za-z0-9_-]+", "Bearer <redacted>", t)
    t = re.sub(r"sk-[A-Za-z0-9_-]{10,}", "sk-<redacted>", t)
    return t


class TimelineSynthesizer:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base = settings.openrouter_base_url.rstrip("/")
        self._model = settings.openrouter_model

    def synthesize(self, windows: list[TimeWindow]) -> TimelineResult:
        key = self._settings.openrouter_key_plain()
        if not key:
            raise ValueError("OPENROUTER_API_KEY is not set.")

        system = (
            "You are a precise assistant. Output ONLY valid JSON, no markdown. "
            "The JSON must match this shape: "
            '{"events":[{"time":"MM:SS","event":"string","speaker":null,"evidence":null}]}'
            "\n\nRules:\n"
            "- 'evidence': If present, MUST be copied verbatim from the transcript below "
            "(a contiguous substring of one or more lines). Do NOT correct spelling, names, "
            "or ASR/word errors; quote exactly what appears.\n"
            "- 'time': MUST be the MM:SS timestamp from the bracket at the start of the line "
            "that this event refers to (the segment start), not a default of 00:00 unless "
            "that line truly starts at 00:00.\n"
            "- 'speaker': Use ONLY neutral labels exactly as shown in the transcript "
            "(e.g. SPEAKER_00, SPEAKER_01, or A:, B: prefixes). If the transcript does not "
            "label speakers per line, use null. Do NOT guess roles like driver or officer.\n"
            "- List events in chronological order; prefer one event per distinct utterance "
            "when the transcript breaks into separate lines."
        )
        user_lines: list[str] = []
        for w in windows:
            user_lines.append(
                f"--- Window [{format_mmss(w.start)}–{format_mmss(w.end)}] ---\n{w.text}"
            )
        user = (
            "Transcript windows follow. Each line is one segment: [MM:SS] optional_speaker: text.\n"
            "Extract salient events.\n\n" + "\n\n".join(user_lines)
        )

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": float(self._settings.ete_openrouter_temperature),
        }

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/local/event-timeline-extractor",
            "X-Title": "Event Timeline Extractor",
        }

        url = f"{self._base}/chat/completions"
        logger.debug("POST %s (body keys: %s)", url, list(payload.keys()))

        with httpx.Client(timeout=120.0) as client:
            r = client.post(url, headers=headers, json=payload)

        raw = _redact_for_logs(r.text[:8000])
        if r.status_code >= 400:
            logger.error("OpenRouter error %s: %s", r.status_code, raw)
            raise RuntimeError(f"OpenRouter HTTP {r.status_code}")

        data = r.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.error("Unexpected response shape: %s", _redact_for_logs(raw))
            raise RuntimeError("LLM response missing choices[0].message.content") from e

        if not isinstance(content, str):
            raise RuntimeError("LLM content is not a string")

        events = _parse_events_json(content)
        meta = {
            "model": self._model,
            "llm_temperature": float(self._settings.ete_openrouter_temperature),
        }
        return TimelineResult(events=events, meta=meta)

    def dry_run(self, windows: list[TimeWindow]) -> TimelineResult:
        """Deterministic placeholder without network."""
        ev: list[TimelineEvent] = []
        for w in windows:
            ev.append(
                TimelineEvent(
                    time=format_mmss(w.start),
                    event="(dry-run) " + (w.text[:80] + ("…" if len(w.text) > 80 else "")),
                    speaker=None,
                    evidence=w.text[:200],
                )
            )
        return TimelineResult(events=ev, meta={"dry_run": True})


def _parse_events_json(content: str) -> list[TimelineEvent]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    obj = json.loads(text)
    events_raw = obj.get("events") if isinstance(obj, dict) else None
    if not isinstance(events_raw, list):
        raise ValueError("JSON must contain an 'events' array.")
    out: list[TimelineEvent] = []
    for e in events_raw:
        if not isinstance(e, dict):
            continue
        out.append(TimelineEvent.model_validate(e))
    return out
