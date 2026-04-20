"""Structured timeline output."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

EventType = Literal["speech", "action", "transition", "decision", "incident", "other"]
_EVENT_TYPE_ALIASES = {
    "request": "decision",
    "refusal": "decision",
    "agreement": "decision",
    "question": "speech",
    "answer": "speech",
    "claim": "speech",
    "reveal": "incident",
    "discovery": "incident",
}
_ALLOWED_EVENT_TYPES = {"speech", "action", "transition", "decision", "incident", "other"}


class TimelineEvent(BaseModel):
    time: str = Field(..., description="Timestamp MM:SS from clip start.")
    event: str = Field(..., description="Short event description.")
    event_type: EventType = Field(
        default="other",
        description="High-level event category for filtering and review.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional model confidence from 0.0 to 1.0.",
    )
    speaker: str | None = None
    evidence: str | None = Field(None, description="Supporting transcript snippet.")
    source_segment_ids: list[str] | None = Field(
        None,
        description="Transcript segment IDs that support this event.",
    )
    source_start: str | None = Field(
        None,
        description="Earliest source timestamp for this event in MM:SS.",
    )
    source_end: str | None = Field(
        None,
        description="Latest source timestamp for this event in MM:SS.",
    )

    @field_validator("event_type", mode="before")
    @classmethod
    def _normalize_event_type(cls, value: str | None) -> str:
        normalized = str(value or "other").strip().lower()
        normalized = _EVENT_TYPE_ALIASES.get(normalized, normalized or "other")
        if normalized not in _ALLOWED_EVENT_TYPES:
            return "other"
        return normalized


class TimelineResult(BaseModel):
    events: list[TimelineEvent]
    meta: dict | None = None
