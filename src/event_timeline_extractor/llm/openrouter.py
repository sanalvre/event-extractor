"""OpenRouter (OpenAI-compatible) client for timeline JSON."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from event_timeline_extractor.chunking import TimeWindow, format_mmss
from event_timeline_extractor.config import Settings
from event_timeline_extractor.resilience import retry_call
from event_timeline_extractor.schema import TimelineEvent, TimelineResult

logger = logging.getLogger(__name__)
_OPENROUTER_TIMEOUT_SEC = 120.0
_OPENROUTER_ATTEMPTS = 3
_OPENROUTER_RETRY_DELAY_SEC = 1.0


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
            '{"events":[{"time":"MM:SS","event":"string","event_type":"other","confidence":null,'
            '"speaker":null,"evidence":null,"source_segment_ids":[],"source_start":null,"source_end":null}]}'
            "\n\nRules:\n"
            "- This is a timeline extraction task, not transcript rewriting. Return only the most "
            "salient moments a reviewer would care about.\n"
            "- Aim for roughly 2 to 4 meaningful events per minute for conversational material.\n"
            "- Within each ~20 second window, prefer 0 to 2 strong events.\n"
            "- It is better to return fewer strong events than many tiny utterances.\n"
            "- Do not skip clear requests, refusals, notable questions, claims, discoveries, "
            "or decisions.\n"
            "- DO NOT emit low-signal backchannels or filler turns such as yes, yeah, okay, right, "
            "please do, of course, really, oh, or short acknowledgements unless they clearly "
            "change the direction of the conversation.\n"
            "- Prefer events that capture a claim, discovery, request, refusal, explanation, "
            "decision, transition, reveal, or notable action.\n"
            "- If several adjacent lines belong to one moment, combine them into one event "
            "instead of listing each line separately.\n"
            "- 'event_type': choose exactly one of speech, action, transition, decision, "
            "incident, or other.\n"
            "- 'confidence': optional decimal from 0.0 to 1.0. Use higher confidence when the "
            "event is directly and clearly stated in the transcript. Use null if uncertain.\n"
            "- 'evidence': If present, MUST be copied verbatim from the transcript below "
            "(a contiguous substring of one or more lines). Do NOT correct spelling, names, "
            "or ASR or word errors; quote exactly what appears.\n"
            "- 'time': MUST be the MM:SS timestamp from the bracket at the start of the line "
            "that this event refers to, not a default of 00:00 unless that line truly starts "
            "at 00:00.\n"
            "- 'source_segment_ids': MUST list the exact segment IDs from the transcript lines "
            "that support this event, such as seg-000001. Use an empty array only if no segment "
            "ID is available.\n"
            "- 'source_start' and 'source_end': copy the earliest and latest MM:SS timestamps from "
            "the supporting transcript lines.\n"
            "- 'speaker': Use ONLY neutral labels exactly as shown in the transcript "
            "(for example SPEAKER_00 or SPEAKER_01). If the transcript does not label speakers "
            "per line, use null. Do NOT guess roles like driver or officer.\n"
            "- List events in chronological order.\n"
            "- Keep event descriptions short but meaningful. Summarize the moment; do not just "
            "repeat a trivial one-line utterance unless that utterance itself is the key event.\n"
            "- Good timeline events often look like: a request is made, a refusal is given, "
            "a claim is asserted, a discovery is described, a live demonstration is proposed, "
            "or the topic clearly shifts."
        )

        has_vision = any(window.vision_context for window in windows)
        user_lines: list[str] = []
        for window in windows:
            block = (
                f"--- Window [{format_mmss(window.start)}-{format_mmss(window.end)}] ---\n"
                f"{window.text}"
            )
            if window.vision_context:
                block += f"\n[VISUAL CONTEXT]\n{window.vision_context}"
            user_lines.append(block)
        vision_hint = (
            "Where present, [VISUAL CONTEXT] provides frame-level scene descriptions; "
            "use them to enrich event descriptions when relevant.\n"
            if has_vision
            else ""
        )
        user = (
            "Transcript windows follow. Each line is one segment:\n"
            "[MM:SS] seg-XXXXXX optional_speaker: text.\n"
            + vision_hint
            + "Extract only the strongest timeline events.\n\n"
            + "\n\n".join(user_lines)
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

        def _post() -> httpx.Response:
            with httpx.Client(timeout=_OPENROUTER_TIMEOUT_SEC) as client:
                return client.post(url, headers=headers, json=payload)

        try:
            response = retry_call(
                _post,
                attempts=_OPENROUTER_ATTEMPTS,
                delay_seconds=_OPENROUTER_RETRY_DELAY_SEC,
                should_retry=lambda exc: isinstance(
                    exc,
                    (httpx.TimeoutException, httpx.NetworkError),
                ),
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise RuntimeError(
                "OpenRouter request failed after "
                f"{_OPENROUTER_ATTEMPTS} attempt(s): {exc}"
            ) from exc

        raw = _redact_for_logs(response.text[:8000])
        if response.status_code >= 400:
            logger.error("OpenRouter error %s: %s", response.status_code, raw)
            raise RuntimeError(f"OpenRouter HTTP {response.status_code}")

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            logger.error("Unexpected response shape: %s", _redact_for_logs(raw))
            raise RuntimeError("LLM response missing choices[0].message.content") from exc

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
        events: list[TimelineEvent] = []
        for window in windows:
            events.append(
                TimelineEvent(
                    time=format_mmss(window.start),
                    event="(dry-run) "
                    + (window.text[:80] + ("..." if len(window.text) > 80 else "")),
                    event_type="other",
                    confidence=0.25,
                    speaker=None,
                    evidence=window.text[:200],
                    source_segment_ids=list(window.source_segment_ids or []),
                    source_start=format_mmss(window.start),
                    source_end=format_mmss(window.end),
                )
            )
        return TimelineResult(events=events, meta={"dry_run": True})


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
    for event in events_raw:
        if not isinstance(event, dict):
            continue
        out.append(TimelineEvent.model_validate(event))
    return out
