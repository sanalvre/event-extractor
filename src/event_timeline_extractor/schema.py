"""Structured timeline output."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TimelineEvent(BaseModel):
    time: str = Field(..., description="Timestamp MM:SS from clip start.")
    event: str = Field(..., description="Short event description.")
    speaker: str | None = None
    evidence: str | None = Field(None, description="Supporting transcript snippet.")


class TimelineResult(BaseModel):
    events: list[TimelineEvent]
    meta: dict | None = None
