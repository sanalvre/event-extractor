"""Tests that vision_context is correctly injected into the LLM prompt."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from event_timeline_extractor.chunking import TimeWindow
from event_timeline_extractor.config import Settings
from event_timeline_extractor.llm.openrouter import TimelineSynthesizer


def _make_settings() -> Settings:
    return Settings(openrouter_api_key="sk-test-key-for-mock")


def _mock_client_returning(content: str) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = content
    mock_resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    mock_client.post.return_value = mock_resp
    return mock_client


_EMPTY_EVENTS = json.dumps({"events": []})


@patch("event_timeline_extractor.llm.openrouter.httpx.Client")
def test_prompt_includes_visual_context_when_present(mock_cls) -> None:
    mock_cls.return_value = _mock_client_returning(_EMPTY_EVENTS)

    windows = [
        TimeWindow(
            start=0.0,
            end=10.0,
            text="[00:00] Officer exits vehicle.",
            frame_paths=[],
            vision_context="00:00 — Officer approaches vehicle.\n00:05 — Driver exits.",
        )
    ]
    TimelineSynthesizer(_make_settings()).synthesize(windows)

    payload = mock_cls.return_value.post.call_args[1]["json"]
    user_content = payload["messages"][1]["content"]
    assert "[VISUAL CONTEXT]" in user_content
    assert "Officer approaches vehicle." in user_content
    assert "Driver exits." in user_content


@patch("event_timeline_extractor.llm.openrouter.httpx.Client")
def test_prompt_omits_visual_context_block_when_empty(mock_cls) -> None:
    mock_cls.return_value = _mock_client_returning(_EMPTY_EVENTS)

    windows = [
        TimeWindow(
            start=0.0,
            end=10.0,
            text="[00:00] Hello there.",
            frame_paths=[],
            vision_context="",
        )
    ]
    TimelineSynthesizer(_make_settings()).synthesize(windows)

    payload = mock_cls.return_value.post.call_args[1]["json"]
    user_content = payload["messages"][1]["content"]
    assert "[VISUAL CONTEXT]" not in user_content


@patch("event_timeline_extractor.llm.openrouter.httpx.Client")
def test_prompt_includes_vision_hint_in_instructions_when_context_present(mock_cls) -> None:
    """When at least one window has vision_context, the preamble should include the hint."""
    mock_cls.return_value = _mock_client_returning(_EMPTY_EVENTS)

    windows = [
        TimeWindow(0.0, 5.0, "[00:00] Text.", [], vision_context="00:00 — Scene here.")
    ]
    TimelineSynthesizer(_make_settings()).synthesize(windows)

    payload = mock_cls.return_value.post.call_args[1]["json"]
    user_content = payload["messages"][1]["content"]
    # Hint text should appear in the preamble alongside the [VISUAL CONTEXT] block
    assert "frame-level scene descriptions" in user_content


@patch("event_timeline_extractor.llm.openrouter.httpx.Client")
def test_prompt_omits_vision_hint_when_no_context(mock_cls) -> None:
    """When no windows have vision_context, the preamble should NOT include the vision hint."""
    mock_cls.return_value = _mock_client_returning(_EMPTY_EVENTS)

    windows = [TimeWindow(0.0, 5.0, "[00:00] Text.", [], vision_context="")]
    TimelineSynthesizer(_make_settings()).synthesize(windows)

    payload = mock_cls.return_value.post.call_args[1]["json"]
    user_content = payload["messages"][1]["content"]
    assert "frame-level scene descriptions" not in user_content
